"""
Phase I — идеи из Kaggle-QSAR поверх Phase H best (public 272.44).

Кандидаты (0 сабмитов):
  1. blend_ic50_full — CatBoost(all 192) только IC50
  2. pic50_ext — pIC50 для ext CatBoost в base
  3. pic50_fr — pIC50 для fr-head
  4. vsa_head — VSA-блок IC50
  5. ext_multiseed — 4 seeds для ext CatBoost
  6. combo: pic50 + full_cb25

Запуск:
  .venv/bin/python run_phase_i_kaggle_ideas.py
  .venv/bin/python run_phase_i_kaggle_ideas.py --quick
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from run_local_signal_search import (
    N_SPLITS,
    _catboost,
    clear_fold_cache,
    competition_score,
    ensure_data,
    fit_oof,
    ic50_to_pic50,
    load_data,
    per_target_rmse,
    pic50_to_ic50_mm,
)
from run_phase_b_si_ic50 import SEEDS, frozen_cc50, load_ext_cols
from run_phase_e_structural import blend_extra_head, feature_blocks

# Phase H best (solution_best.ipynb)
IC50_W = 0.65
CC50_W = 0.25
FR_W = 0.42
MORD_W = 0.55
MORGAN_W = 0.25
CAT_SEEDS = (42, 7, 2024, 1337)


def vsa_block(all_cols: list[str]) -> list[str]:
    """VSA / EState_VSA — отдельно от полного mordred-блока."""
    used = set(feature_blocks(all_cols)["mordred"])
    return [
        c for c in all_cols
        if (
            "VSA" in c or c.startswith("EState_VSA") or c.startswith("VSA_EState")
        )
        and c not in used
    ] or [
        c for c in all_cols
        if "VSA" in c or c.startswith("EState_VSA") or c.startswith("VSA_EState")
    ]


def fit_phase_h(
    X: pd.DataFrame,
    Xt: pd.DataFrame,
    y: np.ndarray,
    ext: list[str],
    blocks: dict[str, list[str]],
    seed: int,
    *,
    ic50_w: float = IC50_W,
    fr_w: float = FR_W,
    mord_w: float = MORD_W,
    morgan_w: float = MORGAN_W,
    ic50_pic50: bool = False,
    ic50_cat_seeds: tuple[int, ...] | None = None,
    X_trans: pd.DataFrame | None = None,
    Xt_trans: pd.DataFrame | None = None,
    cc50_trans_k: int | None = None,
    cc50_blend_w: float | None = None,
    cc50_cat_w: float | None = None,
    cc50_pca_n: int | None = None,
    si_alpha: float | None = None,
    ic50_trans_w: float | None = None,
    X_clust: pd.DataFrame | None = None,
    Xt_clust: pd.DataFrame | None = None,
    X_cc50_cb: pd.DataFrame | None = None,
    Xt_cc50_cb: pd.DataFrame | None = None,
    cc50_cb_lgb: bool = False,
    cc50_trans_target_only: bool = False,
    si_meta_blend: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Текущий best: base + fr + mordred + morgan."""
    cc50_kw: dict = {}
    if cc50_trans_k is not None:
        cc50_kw["cc50_trans_k"] = cc50_trans_k
    if cc50_blend_w is not None:
        cc50_kw["cc50_blend_w"] = cc50_blend_w
    if cc50_pca_n is not None:
        cc50_kw["cc50_pca_n"] = cc50_pca_n
    if si_alpha is not None and not si_meta_blend:
        cc50_kw["si_alpha"] = si_alpha
    if ic50_trans_w is not None:
        cc50_kw["ic50_trans_w"] = ic50_trans_w
    if cc50_cb_lgb:
        cc50_kw["cc50_cb_lgb"] = True
    if cc50_trans_target_only:
        cc50_kw["cc50_trans_target_only"] = True
    if si_meta_blend:
        cc50_kw["si_meta_blend"] = True
    base = frozen_cc50(
        ext,
        ic50_cat_w=ic50_w,
        cc50_cat_w=cc50_cat_w if cc50_cat_w is not None else CC50_W,
        ic50_pic50=ic50_pic50,
        ic50_cat_seeds=ic50_cat_seeds,
        **cc50_kw,
    )
    oof, test = fit_oof(
        X, Xt, y, base, random_state=seed,
        X_transductive=X_trans, X_test_transductive=Xt_trans,
        X_clustering=X_clust, X_test_clustering=Xt_clust,
        X_cc50_cb=X_cc50_cb, X_test_cc50_cb=Xt_cc50_cb,
    )
    oof, test = blend_extra_head(X, Xt, y, oof, test, blocks["fr_only"], 0, fr_w, seed)
    oof, test = blend_extra_head(X, Xt, y, oof, test, blocks["mordred"], 0, mord_w, seed)
    oof, test = blend_extra_head(X, Xt, y, oof, test, blocks["morgan"], 0, morgan_w, seed)
    return oof, test


def blend_ic50_full(
    X: pd.DataFrame,
    Xt: pd.DataFrame,
    y: np.ndarray,
    base_oof: np.ndarray,
    base_test: np.ndarray,
    weight: float,
    seed: int,
    *,
    pic50: bool = False,
    cat_seeds: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """CatBoost на все 192 признака, только IC50."""
    oof = base_oof.copy()
    test = base_test.copy()
    kf = KFold(N_SPLITS, shuffle=True, random_state=seed)
    test_acc = np.zeros(len(Xt))
    seeds = list(cat_seeds or [seed])

    for train_idx, valid_idx in kf.split(X):
        X_fit = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_tr = ic50_to_pic50(y[train_idx, 0]) if pic50 else y[train_idx, 0]

        preds_v, preds_t = [], []
        for cb_seed in seeds:
            m = _catboost(random_seed=cb_seed)
            m.fit(X_fit, y_tr, verbose=False)
            pv = m.predict(X_valid)
            pt = m.predict(Xt)
            if pic50:
                pv = pic50_to_ic50_mm(pv)
                pt = pic50_to_ic50_mm(pt)
            preds_v.append(np.clip(pv, 0, None))
            preds_t.append(np.clip(pt, 0, None))

        pred_v = np.mean(preds_v, axis=0)
        pred_t = np.mean(preds_t, axis=0)
        oof[valid_idx, 0] = (1 - weight) * base_oof[valid_idx, 0] + weight * pred_v
        test_acc += pred_t / N_SPLITS

    test[:, 0] = (1 - weight) * base_test[:, 0] + weight * test_acc
    return oof, test


def blend_ic50_pic50_fr(
    X: pd.DataFrame,
    Xt: pd.DataFrame,
    y: np.ndarray,
    base_oof: np.ndarray,
    base_test: np.ndarray,
    cols: list[str],
    weight: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """fr-head с pIC50 таргетом."""
    oof = base_oof.copy()
    test = base_test.copy()
    kf = KFold(N_SPLITS, shuffle=True, random_state=seed)
    test_acc = np.zeros(len(Xt))

    for train_idx, valid_idx in kf.split(X):
        X_fit = X.iloc[train_idx][cols]
        X_valid = X.iloc[valid_idx][cols]
        y_pic = ic50_to_pic50(y[train_idx, 0])
        m = _catboost(random_seed=seed)
        m.fit(X_fit, y_pic, verbose=False)
        pred_v = np.clip(pic50_to_ic50_mm(m.predict(X_valid)), 0, None)
        pred_t = np.clip(pic50_to_ic50_mm(m.predict(Xt[cols])), 0, None)
        oof[valid_idx, 0] = (1 - weight) * base_oof[valid_idx, 0] + weight * pred_v
        test_acc += pred_t / N_SPLITS

    test[:, 0] = (1 - weight) * base_test[:, 0] + weight * test_acc
    return oof, test


def eval_candidate(
    name: str,
    fn,
    X,
    Xt,
    y,
    ext,
    blocks,
    ref_scores: dict[int, float],
) -> list[dict]:
    rows = []
    for seed in SEEDS if len(ref_scores) > 1 else [42]:
        clear_fold_cache()
        oof = fn(X, Xt, y, ext, blocks, seed)
        sc = competition_score(y, oof)
        rmse = per_target_rmse(y, oof)
        rows.append({
            "name": name,
            "seed": seed,
            "oof": sc,
            "delta": sc - ref_scores[seed],
            "ic50": rmse[0],
            "cc50": rmse[1],
            "si": rmse[2],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    seeds = [42] if args.quick else list(SEEDS)

    ensure_data()
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)
    vsa_cols = vsa_block(cols)
    print(f"Phase I vs Phase H (public 272.44). VSA cols: {len(vsa_cols)}")
    t0 = time.perf_counter()

    ref_scores: dict[int, float] = {}
    for seed in seeds:
        clear_fold_cache()
        oof, _ = fit_phase_h(X, Xt, y, ext, blocks, seed)
        ref_scores[seed] = competition_score(y, oof)
        rmse = per_target_rmse(y, oof)
        print(
            f"REF phase_h @ {seed}: OOF={ref_scores[seed]:.2f} "
            f"(IC50={rmse[0]:.1f} CC50={rmse[1]:.1f} SI={rmse[2]:.1f})"
        )

    candidates: list[tuple[str, object]] = []

    # 1. full CatBoost IC50 blend
    for w in [0.15, 0.20, 0.25]:
        def make_full(w=w):
            def fn(X, Xt, y, ext, blocks, seed):
                clear_fold_cache()
                oof, test = fit_phase_h(X, Xt, y, ext, blocks, seed)
                oof, _ = blend_ic50_full(X, Xt, y, oof, test, w, seed)
                return oof
            return fn
        candidates.append((f"full_cb{int(w*100)}", make_full()))

    # 2. pIC50 ext in base
    def pic50_ext_fn(X, Xt, y, ext, blocks, seed):
        clear_fold_cache()
        return fit_phase_h(X, Xt, y, ext, blocks, seed, ic50_pic50=True)[0]
    candidates.append(("pic50_ext", pic50_ext_fn))

    # 3. ext multiseed
    def ext_ms_fn(X, Xt, y, ext, blocks, seed):
        clear_fold_cache()
        return fit_phase_h(
            X, Xt, y, ext, blocks, seed, ic50_cat_seeds=CAT_SEEDS,
        )[0]
    candidates.append(("ext_multiseed", ext_ms_fn))

    # 4. pIC50 fr-head (переобучаем fr поверх base без fr)
    def pic50_fr_fn(X, Xt, y, ext, blocks, seed):
        clear_fold_cache()
        base = frozen_cc50(ext, ic50_cat_w=IC50_W, cc50_cat_w=CC50_W)
        oof, test = fit_oof(X, Xt, y, base, seed)
        oof, test = blend_ic50_pic50_fr(
            X, Xt, y, oof, test, blocks["fr_only"], FR_W, seed,
        )
        oof, test = blend_extra_head(X, Xt, y, oof, test, blocks["mordred"], 0, MORD_W, seed)
        oof, test = blend_extra_head(X, Xt, y, oof, test, blocks["morgan"], 0, MORGAN_W, seed)
        return oof
    candidates.append(("pic50_fr", pic50_fr_fn))

    # 5. VSA head
    if vsa_cols:
        for w in [0.15, 0.20]:
            def make_vsa(w=w):
                def fn(X, Xt, y, ext, blocks, seed):
                    clear_fold_cache()
                    oof, test = fit_phase_h(X, Xt, y, ext, blocks, seed)
                    oof, _ = blend_extra_head(X, Xt, y, oof, test, vsa_cols, 0, w, seed)
                    return oof
                return fn
            candidates.append((f"vsa_w{int(w*100)}", make_vsa()))

    # 6. combo: pic50_ext + full_cb20
    def combo_fn(X, Xt, y, ext, blocks, seed):
        clear_fold_cache()
        oof, test = fit_phase_h(X, Xt, y, ext, blocks, seed, ic50_pic50=True)
        oof, _ = blend_ic50_full(X, Xt, y, oof, test, 0.20, seed)
        return oof
    candidates.append(("pic50_ext_full20", combo_fn))

    # 7. full_cb25 + pic50 training
    def full_pic50_fn(X, Xt, y, ext, blocks, seed):
        clear_fold_cache()
        oof, test = fit_phase_h(X, Xt, y, ext, blocks, seed)
        oof, _ = blend_ic50_full(X, Xt, y, oof, test, 0.25, seed, pic50=True)
        return oof
    candidates.append(("full25_pic50", full_pic50_fn))

    all_rows: list[dict] = []
    for name, fn in candidates:
        print(f"  {name}...", flush=True, end=" ")
        rows = []
        for seed in seeds:
            clear_fold_cache()
            oof = fn(X, Xt, y, ext, blocks, seed)
            sc = competition_score(y, oof)
            rmse = per_target_rmse(y, oof)
            d = sc - ref_scores[seed]
            rows.append({"name": name, "seed": seed, "oof": sc, "delta": d,
                         "ic50": rmse[0], "cc50": rmse[1], "si": rmse[2]})
            print(f"s{seed}={sc:.2f}({d:+.2f})", end=" ")
        all_rows.extend(rows)
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
    sm.to_csv("phase_i_kaggle_ideas.csv", index=False)
    df.to_csv("phase_i_kaggle_ideas.detail.csv", index=False)

    print(f"\n=== Phase I ({time.perf_counter()-t0:.0f}s) ===")
    print(sm.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    top = sm[sm["delta_mean"] < -0.05].head(5)
    if not top.empty:
        print("\n★ Кандидаты для сабмита:")
        for _, r in top.iterrows():
            print(f"  {r['name']}: mean Δ={r['delta_mean']:+.2f}, wins={r['wins']}")
        print("\n  .venv/bin/python make_submission_phase_i.py <variant>")
    else:
        print("\n→ Нет улучшений >0.05 OOF. Phase H остаётся best.")


if __name__ == "__main__":
    main()
