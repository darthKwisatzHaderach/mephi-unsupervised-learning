"""
Phase J — vsa_w15 и LGBM IC50-head поверх Phase I (full_cb25, public ~272.17).

Запуск:
  python run_phase_j.py
  python run_phase_j.py --quick
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from run_local_signal_search import (
    N_SPLITS,
    clear_fold_cache,
    competition_score,
    ensure_data,
    load_data,
    per_target_rmse,
)
from run_phase_b_si_ic50 import SEEDS, load_ext_cols
from run_phase_e_structural import blend_extra_head, feature_blocks
from run_phase_i_kaggle_ideas import blend_ic50_full, fit_phase_h, vsa_block

FULL_CB_W = 0.25

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def fit_phase_i(
    X: pd.DataFrame,
    Xt: pd.DataFrame,
    y: np.ndarray,
    ext: list[str],
    blocks: dict[str, list[str]],
    seed: int,
    *,
    full_w: float = FULL_CB_W,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase H + CatBoost(all 192) IC50 blend."""
    oof, test = fit_phase_h(X, Xt, y, ext, blocks, seed)
    return blend_ic50_full(X, Xt, y, oof, test, full_w, seed)


def blend_ic50_lgb(
    X: pd.DataFrame,
    Xt: pd.DataFrame,
    y: np.ndarray,
    base_oof: np.ndarray,
    base_test: np.ndarray,
    weight: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """LightGBM на все 192 признака, только IC50."""
    oof = base_oof.copy()
    test = base_test.copy()
    kf = KFold(N_SPLITS, shuffle=True, random_state=seed)
    test_acc = np.zeros(len(Xt))

    for train_idx, valid_idx in kf.split(X):
        X_fit = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_tr = y[train_idx, 0]

        m = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        m.fit(X_fit, y_tr)
        pred_v = np.clip(m.predict(X_valid), 0, None)
        pred_t = np.clip(m.predict(Xt), 0, None)
        oof[valid_idx, 0] = (1 - weight) * base_oof[valid_idx, 0] + weight * pred_v
        test_acc += pred_t / N_SPLITS

    test[:, 0] = (1 - weight) * base_test[:, 0] + weight * test_acc
    return oof, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    seeds = [42] if args.quick else list(SEEDS)

    if not HAS_LGB:
        print("WARNING: lightgbm не установлен — LGB-кандидаты пропущены")

    ensure_data()
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)
    vsa_cols = vsa_block(cols)
    print(f"Phase J vs Phase I (full_cb25). VSA cols: {len(vsa_cols)}")
    t0 = time.perf_counter()

    ref_scores: dict[int, float] = {}
    for seed in seeds:
        clear_fold_cache()
        oof, _ = fit_phase_i(X, Xt, y, ext, blocks, seed)
        ref_scores[seed] = competition_score(y, oof)
        rmse = per_target_rmse(y, oof)
        print(
            f"REF phase_i @ {seed}: OOF={ref_scores[seed]:.2f} "
            f"(IC50={rmse[0]:.1f} CC50={rmse[1]:.1f} SI={rmse[2]:.1f})"
        )

    candidates: list[tuple[str, object]] = []

    # VSA поверх Phase I
    if vsa_cols:
        for w in [0.15, 0.20]:
            def make_vsa_i(w=w):
                def fn(X, Xt, y, ext, blocks, seed):
                    clear_fold_cache()
                    oof, test = fit_phase_i(X, Xt, y, ext, blocks, seed)
                    oof, _ = blend_extra_head(X, Xt, y, oof, test, vsa_cols, 0, w, seed)
                    return oof
                return fn
            candidates.append((f"i_vsa_w{int(w*100)}", make_vsa_i()))

        # VSA поверх Phase H (референс из Phase I grid)
        for w in [0.15]:
            def make_vsa_h(w=w):
                def fn(X, Xt, y, ext, blocks, seed):
                    clear_fold_cache()
                    oof, test = fit_phase_h(X, Xt, y, ext, blocks, seed)
                    oof, _ = blend_extra_head(X, Xt, y, oof, test, vsa_cols, 0, w, seed)
                    return oof
                return fn
            candidates.append((f"h_vsa_w{int(w*100)}", make_vsa_h()))

    # LGBM IC50-head поверх Phase I
    if HAS_LGB:
        for w in [0.15, 0.20, 0.25]:
            def make_lgb(w=w):
                def fn(X, Xt, y, ext, blocks, seed):
                    clear_fold_cache()
                    oof, test = fit_phase_i(X, Xt, y, ext, blocks, seed)
                    oof, _ = blend_ic50_lgb(X, Xt, y, oof, test, w, seed)
                    return oof
                return fn
            candidates.append((f"lgb_ic{int(w*100)}", make_lgb()))

    all_rows: list[dict] = []
    for name, fn in candidates:
        print(f"  {name}...", flush=True, end=" ")
        for seed in seeds:
            clear_fold_cache()
            oof = fn(X, Xt, y, ext, blocks, seed)
            sc = competition_score(y, oof)
            rmse = per_target_rmse(y, oof)
            d = sc - ref_scores[seed]
            all_rows.append({
                "name": name,
                "seed": seed,
                "oof": sc,
                "delta": d,
                "ic50": rmse[0],
                "cc50": rmse[1],
                "si": rmse[2],
            })
            print(f"s{seed}={sc:.2f}({d:+.2f})", end=" ")
        print()

    df = pd.DataFrame(all_rows)
    sm = (
        df.groupby("name")
        .agg(
            oof_mean=("oof", "mean"),
            delta_mean=("delta", "mean"),
            min_delta=("delta", "min"),
            wins=("delta", lambda s: int((s < -0.05).sum())),
            ic50=("ic50", "mean"),
            si=("si", "mean"),
        )
        .reset_index()
        .sort_values("delta_mean")
    )
    sm.to_csv("phase_j.csv", index=False)
    df.to_csv("phase_j.detail.csv", index=False)

    print(f"\n=== Phase J ({time.perf_counter()-t0:.0f}s) ===")
    print(sm.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    better = sm[sm["delta_mean"] < 0].head(5)
    if not better.empty:
        print("\n★ Лучше ref Phase I по mean OOF:")
        for _, r in better.iterrows():
            print(f"  {r['name']}: Δ={r['delta_mean']:+.2f}, wins={r['wins']}")
        print("\n  python make_submission_phase_j.py <variant>")


if __name__ == "__main__":
    main()
