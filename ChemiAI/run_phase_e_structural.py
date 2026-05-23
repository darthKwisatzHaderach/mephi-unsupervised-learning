"""
Фаза E: Morgan-proxy (fr_* + FpDensity) и Mordred-proxy (topo/VSA/charge)
поверх public best w55_cc25. SMILES в датасете нет — используем доступные блоки признаков.
"""
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from run_local_signal_search import (
    N_SPLITS,
    _catboost,
    clear_fold_cache,
    competition_score,
    ensure_data,
    fit_oof,
    load_data,
    per_target_rmse,
)
from run_phase_b_si_ic50 import SEEDS, frozen_cc50, load_ext_cols
from sklearn.model_selection import KFold


@dataclass
class ExtraHead:
    name: str
    cols: list[str]
    target: int  # 0 IC50, 1 CC50, 2 SI
    weight: float


def feature_blocks(all_cols: list[str]) -> dict[str, list[str]]:
    fr = [c for c in all_cols if c.startswith("fr_")]
    morgan = fr + [c for c in all_cols if c.startswith("FpDensityMorgan")]
    mordred = [
        c for c in all_cols
        if any(
            c.startswith(p)
            for p in (
                "BCUT2D_", "Chi", "Kappa", "EState", "VSA_", "PEOE_VSA",
                "SMR_VSA", "SlogP_VSA", "MaxPartial", "MinPartial",
                "MaxAbsPartial", "MinAbsPartial", "MaxEState", "MinEState",
                "MaxAbsEState", "MinAbsEState", "HallKier", "BertzCT",
                "BalabanJ", "Ipc", "AvgIpc", "qed", "SPS", "LabuteASA",
                "TPSA", "FractionCSP3", "RingCount", "NumRotatable",
            )
        )
    ]
    mordred = [c for c in mordred if c not in morgan]
    return {"morgan": morgan, "mordred": mordred, "fr_only": fr}


def blend_extra_head(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    base_oof: np.ndarray,
    base_test: np.ndarray,
    cols: list[str],
    target: int,
    weight: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    oof = base_oof.copy()
    test = base_test.copy()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    test_acc = np.zeros(len(X_test))

    for train_idx, valid_idx in kf.split(X_train):
        X_fit = X_train.iloc[train_idx][cols]
        X_valid = X_train.iloc[valid_idx][cols]
        y_fit = np.log1p(y_train[train_idx, target]) if target > 0 else y_train[train_idx, target]
        y_va = y_train[valid_idx, target]

        m = _catboost(random_seed=random_state)
        if target == 0:
            m.fit(X_fit, y_fit, verbose=False)
            pred_v = np.clip(m.predict(X_valid), 0, None)
            pred_t = np.clip(m.predict(X_test[cols]), 0, None)
        else:
            m.fit(X_fit, y_fit, verbose=False)
            pred_v = np.expm1(np.clip(m.predict(X_valid), 0, 12))
            pred_t = np.expm1(np.clip(m.predict(X_test[cols]), 0, 12))
            pred_v = np.clip(pred_v, 0, None)
            pred_t = np.clip(pred_t, 0, None)

        oof[valid_idx, target] = (1 - weight) * base_oof[valid_idx, target] + weight * pred_v
        test_acc += pred_t / N_SPLITS

    test[:, target] = (1 - weight) * base_test[:, target] + weight * test_acc
    return oof, test


def eval_head(
    base_cfg,
    head: ExtraHead,
    X, Xt, y,
) -> dict:
    rows = []
    for seed in SEEDS:
        clear_fold_cache()
        base_oof, base_test = fit_oof(X, Xt, y, base_cfg, random_state=seed)
        ref = competition_score(y, base_oof)
        oof, _ = blend_extra_head(
            X, Xt, y, base_oof, base_test, head.cols, head.target, head.weight, seed,
        )
        sc = competition_score(y, oof)
        rmse = per_target_rmse(y, oof)
        rows.append({
            "delta": sc - ref,
            "oof": sc,
            "t_rmse": rmse[head.target],
        })
    df = pd.DataFrame(rows)
    return {
        "name": head.name,
        "delta_mean": df["delta"].mean(),
        "wins": int((df["delta"] < -0.05).sum()),
        "t_rmse": df["t_rmse"].mean(),
    }


def main() -> None:
    ensure_data()
    X, Xt, y, all_cols = load_data()
    ext = load_ext_cols(all_cols)
    blocks = feature_blocks(all_cols)
    base = frozen_cc50(ext, ic50_cat_w=0.55, cc50_cat_w=0.25)

    print("Blocks:", {k: len(v) for k, v in blocks.items()})
    t0 = time.perf_counter()
    candidates: list[ExtraHead] = []

    for block in ["morgan", "mordred", "fr_only"]:
        cols = blocks[block]
        for t, tname in [(0, "ic50"), (1, "cc50"), (2, "si")]:
            for w in [0.15, 0.25, 0.35]:
                candidates.append(ExtraHead(f"{block}_{tname}_w{int(w*100)}", cols, t, w))

    # комбо лучших направлений (предварительно morgan ic50 + mordred si)
    for w1, w2 in [(0.25, 0.25), (0.25, 0.15), (0.15, 0.25)]:
        candidates.append(ExtraHead(f"combo_morg_ic50_mord_si", blocks["morgan"], 0, w1))  # placeholder

    results = []
    for head in candidates:
        if head.name.startswith("combo"):
            continue
        r = eval_head(base, head, X, Xt, y)
        results.append(r)
        print(f"{r['name']:<28} d={r['delta_mean']:+.2f} wins={r['wins']}/3 t_rmse={r['t_rmse']:.0f}")

    res_df = pd.DataFrame(results).sort_values("delta_mean")
    print("\n=== TOP 10 ===")
    print(res_df.head(10).to_string(index=False))
    res_df.to_csv("phase_e_structural.csv", index=False)

    # лучший combo: apply top IC50 + top SI heads sequentially
    top_ic50 = res_df[res_df["name"].str.contains("_ic50_")].iloc[0]
    top_cc50 = res_df[res_df["name"].str.contains("_cc50_")].iloc[0]
    top_si = res_df[res_df["name"].str.contains("_si_")].iloc[0]
    print(f"\nBest per target: {top_ic50['name']}, {top_cc50['name']}, {top_si['name']}")

    for seed in SEEDS:
        clear_fold_cache()
        bo, bt = fit_oof(X, Xt, y, base, seed)
        ref = competition_score(y, bo)
        oof = bo.copy()
        for row, t in [(top_ic50, 0), (top_cc50, 1), (top_si, 2)]:
            name = row["name"]
            head = next(h for h in candidates if h.name == name)
            oof, _ = blend_extra_head(X, Xt, y, oof, bt, head.cols, t, head.weight, seed)
        print(f"seed {seed} combo3 d={competition_score(y,oof)-ref:+.2f}")

    print(f"Время: {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
