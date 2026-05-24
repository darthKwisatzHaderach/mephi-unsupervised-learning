"""
Phase L — гипотезы 2.1 (LGB в structural head) и 2.2 (pair interactions).

Пропущено: 2.3 adaptive SI α, 2.4 CC50 KNN, 2.5 quantile post-process (закрытые/рискованные ветки).

Запуск:
  python run_phase_l2.py --quick
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

from phase_k_fe import engineer_features
from run_local_signal_search import (
    N_SPLITS,
    clear_fold_cache,
    competition_score,
    ensure_data,
    load_data,
    per_target_rmse,
)
from run_phase_b_si_ic50 import SEEDS, frozen_cc50, load_ext_cols
from run_phase_e_structural import blend_extra_head, feature_blocks
from run_phase_i_kaggle_ideas import (
    CC50_W,
    FR_W,
    IC50_W,
    MORD_W,
    MORGAN_W,
    blend_ic50_full,
    fit_phase_h,
)
from run_phase_j import FULL_CB_W, blend_ic50_lgb

LGB_W = 0.45


@dataclass
class FeCfg:
    ratios: bool = True
    interactions: bool = False


def _lgb_ic50():
    return LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        verbose=-1,
    )


def blend_extra_head_lgb(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    base_oof: np.ndarray,
    base_test: np.ndarray,
    cols: list[str],
    weight: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """LGBM IC50-head на блоке признаков (гипотеза 2.1)."""
    oof = base_oof.copy()
    test = base_test.copy()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    test_acc = np.zeros(len(X_test))

    for train_idx, valid_idx in kf.split(X_train):
        X_fit = X_train.iloc[train_idx][cols]
        X_valid = X_train.iloc[valid_idx][cols]
        y_fit = y_train[train_idx, 0]

        m = _lgb_ic50()
        m.set_params(random_state=random_state)
        m.fit(X_fit, y_fit)
        pred_v = np.clip(m.predict(X_valid), 0, None)
        pred_t = np.clip(m.predict(X_test[cols]), 0, None)

        oof[valid_idx, 0] = (1 - weight) * base_oof[valid_idx, 0] + weight * pred_v
        test_acc += pred_t / N_SPLITS

    test[:, 0] = (1 - weight) * base_test[:, 0] + weight * test_acc
    return oof, test


def fit_phase_h_lgb_morgan(
    X: pd.DataFrame,
    Xt: pd.DataFrame,
    y: np.ndarray,
    ext: list[str],
    blocks: dict[str, list[str]],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase H, но morgan-head = LightGBM вместо CatBoost."""
    base = frozen_cc50(ext, ic50_cat_w=IC50_W, cc50_cat_w=CC50_W)
    from run_local_signal_search import fit_oof

    oof, test = fit_oof(X, Xt, y, base, random_state=seed)
    oof, test = blend_extra_head(X, Xt, y, oof, test, blocks["fr_only"], 0, FR_W, seed)
    oof, test = blend_extra_head(X, Xt, y, oof, test, blocks["mordred"], 0, MORD_W, seed)
    oof, test = blend_extra_head_lgb(
        X, Xt, y, oof, test, blocks["morgan"], MORGAN_W, seed,
    )
    return oof, test


def fit_ratio_lgb_pipeline(
    X: pd.DataFrame,
    Xt: pd.DataFrame,
    y: np.ndarray,
    ext: list[str],
    blocks: dict[str, list[str]],
    seed: int,
    fe: FeCfg,
    *,
    lgb_w: float = LGB_W,
    lgb_morgan: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Best Phase K: ratios + full_cb + LGB; опционально LGB morgan-head."""
    Xh = engineer_features(X, ratios=fe.ratios, interactions=fe.interactions)
    Xth = engineer_features(Xt, ratios=fe.ratios, interactions=fe.interactions)

    if lgb_morgan:
        oof, test = fit_phase_h_lgb_morgan(X, Xt, y, ext, blocks, seed)
    else:
        oof, test = fit_phase_h(X, Xt, y, ext, blocks, seed)

    oof, test = blend_ic50_full(X, Xt, y, oof, test, FULL_CB_W, seed)
    oof, test = blend_ic50_lgb(Xh, Xth, y, oof, test, lgb_w, seed)
    return oof, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    seeds = [42] if args.quick else list(SEEDS)

    ensure_data()
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)
    t0 = time.perf_counter()

    candidates: list[tuple[str, FeCfg, float, bool]] = [
        ("ratio_lgb45", FeCfg(), 0.45, False),
        ("inter_lgb45", FeCfg(interactions=True), 0.45, False),
        ("inter_lgb48", FeCfg(interactions=True), 0.48, False),
        ("lgb_morgan_lgb45", FeCfg(), 0.45, True),
        ("inter_lgb_morgan45", FeCfg(interactions=True), 0.45, True),
    ]

    ref_scores: dict[int, float] = {}
    for seed in seeds:
        clear_fold_cache()
        oof, _ = fit_ratio_lgb_pipeline(
            X, Xt, y, ext, blocks, seed, FeCfg(), lgb_w=0.45,
        )
        ref_scores[seed] = competition_score(y, oof)
    print("REF ratio_lgb45: " + ", ".join(f"s{s}={ref_scores[s]:.2f}" for s in seeds))

    rows: list[dict] = []
    for name, fe, lgb_w, lgb_morgan in candidates:
        if name == "ratio_lgb45":
            continue
        print(f"  {name}...", flush=True, end=" ")
        for seed in seeds:
            clear_fold_cache()
            oof, _ = fit_ratio_lgb_pipeline(
                X, Xt, y, ext, blocks, seed, fe,
                lgb_w=lgb_w, lgb_morgan=lgb_morgan,
            )
            sc = competition_score(y, oof)
            rmse = per_target_rmse(y, oof)
            d = sc - ref_scores[seed]
            rows.append({
                "name": name,
                "seed": seed,
                "oof": sc,
                "delta": d,
                "ic50": rmse[0],
            })
            print(f"s{seed}={sc:.2f}({d:+.2f})", end=" ")
        print()

    df = pd.DataFrame(rows)
    sm = (
        df.groupby("name")
        .agg(oof_mean=("oof", "mean"), delta_mean=("delta", "mean"), ic50=("ic50", "mean"))
        .reset_index()
        .sort_values("delta_mean")
    )
    sm.to_csv("phase_l2.csv", index=False)
    df.to_csv("phase_l2.detail.csv", index=False)

    print(f"\n=== Phase L2 ({time.perf_counter()-t0:.0f}s) ===")
    print(sm.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print("\nPublic ref: ratio_lgb45 = 269.21")
    print("  python make_submission_phase_l2.py inter_lgb48")


if __name__ == "__main__":
    main()
