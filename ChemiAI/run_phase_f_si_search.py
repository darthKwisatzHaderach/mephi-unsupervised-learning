"""
Фаза F: массовый SI-поиск поверх public best fr42_ic65 (274.76).

Стратегия:
  1) Кэшируем base+fr IC50 один раз на seed.
  2) Быстрые SI-головы (ExtraHead, transductive, log-ratio, alt loss).
  3) Медленная сетка PipelineConfig (si_alpha, si_robust_w) с полным refit.

Ref: fr42_ic65 OOF ~528.03 (seed42 mean over 3 seeds ~528.03).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from run_local_signal_search import (
    N_SPLITS,
    _catboost,
    clear_fold_cache,
    competition_score,
    ensure_data,
    fit_oof,
    load_data,
    make_transductive_pca_features,
    per_target_rmse,
)
from run_phase_b_si_ic50 import SEEDS, frozen_cc50, load_ext_cols
from run_phase_e_structural import ExtraHead, blend_extra_head, feature_blocks

BEST_IC50_W = 0.65
BEST_CC50_W = 0.25
BEST_FR_W = 0.42


def best_base_cfg(ext: list[str]):
    return frozen_cc50(ext, ic50_cat_w=BEST_IC50_W, cc50_cat_w=BEST_CC50_W)


@dataclass
class CachedBase:
    oof: np.ndarray
    test: np.ndarray


def fit_best_base(
    X, Xt, y, ext, blocks, seed: int,
) -> CachedBase:
    clear_fold_cache()
    base = best_base_cfg(ext)
    oof, test = fit_oof(X, Xt, y, base, random_state=seed)
    oof, test = blend_extra_head(
        X, Xt, y, oof, test, blocks["fr_only"], 0, BEST_FR_W, seed,
    )
    return CachedBase(oof=oof, test=test)


def eval_on_base(
    name: str,
    base: CachedBase,
    oof_new: np.ndarray,
    ref_scores: dict[int, float],
    seed: int,
    y: np.ndarray,
) -> dict:
    score = competition_score(y, oof_new)
    rmse = per_target_rmse(y, oof_new)
    return {
        "name": name,
        "seed": seed,
        "oof": score,
        "delta": score - ref_scores[seed],
        "si_rmse": rmse[2],
        "ic50_rmse": rmse[0],
        "cc50_rmse": rmse[1],
    }


def blend_si_transductive_head(
    X_train,
    X_test,
    y_train,
    base_oof: np.ndarray,
    base_test: np.ndarray,
    trans_k: int,
    weight: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    oof = base_oof.copy()
    test = base_test.copy()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    test_acc = np.zeros(len(X_test))

    for train_idx, valid_idx in kf.split(X_train):
        X_fit = X_train.iloc[train_idx]
        X_valid = X_train.iloc[valid_idx]
        y_fit = y_train[train_idx]

        X_fit_pca, X_valid_pca, X_test_pca = make_transductive_pca_features(
            X_fit, X_valid, X_test, random_state=random_state,
        )
        from sklearn.neighbors import KNeighborsRegressor

        knn = KNeighborsRegressor(
            n_neighbors=trans_k, weights="distance", n_jobs=-1,
        )
        knn.fit(X_fit_pca, np.log1p(y_fit[:, 2]))
        pred_v = np.clip(np.expm1(knn.predict(X_valid_pca)), 0, None)
        pred_t = np.clip(np.expm1(knn.predict(X_test_pca)), 0, None)

        oof[valid_idx, 2] = (1 - weight) * base_oof[valid_idx, 2] + weight * pred_v
        test_acc += pred_t / N_SPLITS

    test[:, 2] = (1 - weight) * base_test[:, 2] + weight * test_acc
    return oof, test


def blend_si_logratio_head(
    X_train,
    X_test,
    y_train,
    base_oof: np.ndarray,
    base_test: np.ndarray,
    cols: list[str],
    weight: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """CatBoost на log1p(CC50/IC50) как proxy SI."""
    oof = base_oof.copy()
    test = base_test.copy()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    test_acc = np.zeros(len(X_test))
    eps = 1e-3

    for train_idx, valid_idx in kf.split(X_train):
        X_fit = X_train.iloc[train_idx][cols]
        X_valid = X_train.iloc[valid_idx][cols]
        ratio = y_train[train_idx, 1] / np.clip(y_train[train_idx, 0], eps, None)
        y_ratio = np.log1p(np.clip(ratio, 0, None))

        model = _catboost(random_seed=random_state)
        model.fit(X_fit, y_ratio, verbose=False)
        pred_v = np.expm1(np.clip(model.predict(X_valid), 0, 12))
        pred_t = np.expm1(np.clip(model.predict(X_test[cols]), 0, 12))
        pred_v = np.clip(pred_v, 0, None)
        pred_t = np.clip(pred_t, 0, None)

        oof[valid_idx, 2] = (1 - weight) * base_oof[valid_idx, 2] + weight * pred_v
        test_acc += pred_t / N_SPLITS

    test[:, 2] = (1 - weight) * base_test[:, 2] + weight * test_acc
    return oof, test


def blend_si_alt_robust_head(
    X_train,
    X_test,
    y_train,
    base_oof: np.ndarray,
    base_test: np.ndarray,
    weight: float,
    random_state: int,
    loss_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Альтернативный SI-head: Huber/Quantile GBR на cluster-признаках."""
    from run_local_signal_search import build_clustering_features

    oof = base_oof.copy()
    test = base_test.copy()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    test_acc = np.zeros(len(X_test))

    for train_idx, valid_idx in kf.split(X_train):
        X_fit = X_train.iloc[train_idx]
        X_valid = X_train.iloc[valid_idx]
        y_si = np.log1p(y_train[train_idx, 2])

        X_fit_aug, X_valid_aug = build_clustering_features(
            X_fit, X_valid, random_state=random_state,
        )
        _, X_test_aug = build_clustering_features(
            X_fit, X_test, random_state=random_state,
        )

        if loss_kind == "huber":
            model = GradientBoostingRegressor(
                loss="huber",
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                random_state=random_state,
            )
        elif loss_kind == "quantile":
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=0.5,
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                random_state=random_state,
            )
        else:
            model = HistGradientBoostingRegressor(
                max_depth=5,
                learning_rate=0.05,
                max_iter=500,
                loss="absolute_error",
                random_state=random_state,
            )

        model.fit(X_fit_aug, y_si)
        pred_v = np.expm1(np.clip(model.predict(X_valid_aug), 0, 12))
        pred_t = np.expm1(np.clip(model.predict(X_test_aug), 0, 12))
        pred_v = np.clip(pred_v, 0, None)
        pred_t = np.clip(pred_t, 0, None)

        oof[valid_idx, 2] = (1 - weight) * base_oof[valid_idx, 2] + weight * pred_v
        test_acc += pred_t / N_SPLITS

    test[:, 2] = (1 - weight) * base_test[:, 2] + weight * test_acc
    return oof, test


def run_fast_heads(
    X, Xt, y, blocks, bases: dict[int, CachedBase], ref_scores: dict[int, float],
) -> pd.DataFrame:
    rows = []

    def run_head(name: str, seed: int, base: CachedBase) -> np.ndarray:
        if name.startswith("si_fr_only_"):
            w = int(name.split("_w")[1]) / 100
            oof, _ = blend_extra_head(
                X, Xt, y, base.oof, base.test, blocks["fr_only"], 2, w, seed,
            )
            return oof
        if name.startswith("si_mordred_") and "ratio" not in name:
            w = int(name.split("_w")[1]) / 100
            oof, _ = blend_extra_head(
                X, Xt, y, base.oof, base.test, blocks["mordred"], 2, w, seed,
            )
            return oof
        if name.startswith("si_morgan_"):
            w = int(name.split("_w")[1]) / 100
            oof, _ = blend_extra_head(
                X, Xt, y, base.oof, base.test, blocks["morgan"], 2, w, seed,
            )
            return oof
        if name.startswith("si_trans_"):
            parts = name.split("_")
            k = int(parts[2][1:])
            w = int(parts[3][1:]) / 100
            oof, _ = blend_si_transductive_head(
                X, Xt, y, base.oof, base.test, k, w, seed,
            )
            return oof
        if name.startswith("si_ratio_"):
            rest = name[len("si_ratio_"):]
            block, wpart = rest.rsplit("_w", 1)
            w = int(wpart) / 100
            oof, _ = blend_si_logratio_head(
                X, Xt, y, base.oof, base.test, blocks[block], w, seed,
            )
            return oof
        if name.startswith("si_huber_") or name.startswith("si_quantile_"):
            kind = "huber" if "huber" in name else "quantile"
            w = int(name.split("_w")[1]) / 100
            oof, _ = blend_si_alt_robust_head(
                X, Xt, y, base.oof, base.test, w, seed, kind,
            )
            return oof
        if name.startswith("si_abs_extra_"):
            w = int(name.split("_w")[1]) / 100
            oof, _ = blend_si_alt_robust_head(
                X, Xt, y, base.oof, base.test, w, seed, "abs_extra",
            )
            return oof
        if name.startswith("si_mord_w") and "_ratio" in name:
            w_mord = int(name.split("_w")[1].split("_")[0]) / 100
            w_ratio = int(name.split("ratio")[1]) / 100
            return _combo_mordred_ratio(X, Xt, y, base, blocks["mordred"], w_mord, w_ratio, seed)
        raise ValueError(name)

    names: list[str] = []
    for block in ["fr_only", "mordred", "morgan"]:
        prefix = {"fr_only": "si_fr_only", "mordred": "si_mordred", "morgan": "si_morgan"}[block]
        for w in [0.15, 0.20, 0.25, 0.30, 0.35]:
            names.append(f"{prefix}_w{int(w*100)}")

    for k in [3, 5, 8]:
        for w in [0.15, 0.20, 0.25, 0.30]:
            names.append(f"si_trans_k{k}_w{int(w*100)}")

    for block in ["fr_only", "mordred", "morgan"]:
        for w in [0.15, 0.20, 0.25, 0.30]:
            names.append(f"si_ratio_{block}_w{int(w*100)}")

    for kind in ["huber", "quantile", "abs_extra"]:
        for w in [0.20, 0.30, 0.40]:
            names.append(f"si_{kind}_w{int(w*100)}")

    for w in [0.20, 0.25]:
        names.append(f"si_mord_w{int(w*100)}_ratio15")

    for hi, name in enumerate(names):
        for seed in SEEDS:
            base = bases[seed]
            oof_new = run_head(name, seed, base)
            rows.append(eval_on_base(name, base, oof_new, ref_scores, seed, y))
        if (hi + 1) % 10 == 0:
            print(f"  fast heads: {hi+1}/{len(names)}")

    return pd.DataFrame(rows)


def _combo_mordred_ratio(X, Xt, y, base, cols, w_mord, w_ratio, seed):
    oof1, _ = blend_extra_head(X, Xt, y, base.oof, base.test, cols, 2, w_mord, seed)
    oof2, _ = blend_si_logratio_head(
        X, Xt, y, base.oof, base.test, cols, w_ratio, seed,
    )
    out = base.oof.copy()
    out[:, 2] = 0.5 * oof1[:, 2] + 0.5 * oof2[:, 2]
    return out


def run_config_grid(
    X, Xt, y, ext, blocks, ref_scores: dict[int, float],
) -> pd.DataFrame:
    rows = []
    configs: list[tuple[str, dict]] = []

    for alpha in [0.28, 0.30, 0.32, 0.33, 0.35, 0.37, 0.40, 0.42, 0.45]:
        configs.append((f"si_a{int(alpha*100)}", dict(si_alpha=alpha)))

    for rw in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
        configs.append((f"si_rw{int(rw*100)}", dict(si_robust_w=rw)))

    for alpha, rw in [(0.32, 0.25), (0.33, 0.25), (0.37, 0.35), (0.40, 0.30)]:
        configs.append((
            f"si_a{int(alpha*100)}_rw{int(rw*100)}",
            dict(si_alpha=alpha, si_robust_w=rw),
        ))

    for k in [60, 90, 120]:
        for w in [0.25, 0.35, 0.45]:
            configs.append((
                f"si_cb_k{k}_w{int(w*100)}",
                dict(si_catboost_w=w, si_topk=k),
            ))

    total = len(configs)
    for ci, (name, kw) in enumerate(configs):
        cfg = best_base_cfg(ext)
        for k, v in kw.items():
            setattr(cfg, k, v)
        for seed in SEEDS:
            clear_fold_cache()
            oof, test = fit_oof(X, Xt, y, cfg, random_state=seed)
            oof, test = blend_extra_head(
                X, Xt, y, oof, test, blocks["fr_only"], 0, BEST_FR_W, seed,
            )
            rows.append(eval_on_base(name, CachedBase(oof, test), oof, ref_scores, seed, y))
        if (ci + 1) % 5 == 0:
            print(f"  config grid: {ci+1}/{total}")

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("name").agg(
        oof_mean=("oof", "mean"),
        delta_mean=("delta", "mean"),
        delta_min=("delta", "min"),
        wins=("delta", lambda s: int((s < -0.05).sum())),
        si_rmse=("si_rmse", "mean"),
        ic50_rmse=("ic50_rmse", "mean"),
        cc50_rmse=("cc50_rmse", "mean"),
    ).reset_index().sort_values("delta_mean")


def main() -> None:
    ensure_data()
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)
    t0 = time.perf_counter()

    print("=== Phase F: SI search on fr42_ic65 base ===")
    print("Caching best base per seed...")
    bases: dict[int, CachedBase] = {}
    ref_scores: dict[int, float] = {}
    for seed in SEEDS:
        bases[seed] = fit_best_base(X, Xt, y, ext, blocks, seed)
        ref_scores[seed] = competition_score(y, bases[seed].oof)
        rmse = per_target_rmse(y, bases[seed].oof)
        print(
            f"  seed {seed}: mean={ref_scores[seed]:.2f} "
            f"SI={rmse[2]:.1f} IC50={rmse[0]:.1f} CC50={rmse[1]:.1f}",
        )

    print(f"\n--- Fast SI heads ({len(SEEDS)} seeds) ---")
    t1 = time.perf_counter()
    fast_df = run_fast_heads(X, Xt, y, blocks, bases, ref_scores)
    fast_sum = summarize(fast_df)
    fast_sum.to_csv("phase_f_si_fast.csv", index=False)
    fast_df.to_csv("phase_f_si_fast.detail.csv", index=False)
    print(fast_sum.head(20).to_string(index=False))
    print(f"Fast section: {time.perf_counter()-t1:.0f}s")

    print(f"\n--- Config grid (full refit, {len(SEEDS)} seeds) ---")
    t2 = time.perf_counter()
    cfg_df = run_config_grid(X, Xt, y, ext, blocks, ref_scores)
    cfg_sum = summarize(cfg_df)
    cfg_sum.to_csv("phase_f_si_config.csv", index=False)
    cfg_df.to_csv("phase_f_si_config.detail.csv", index=False)
    print(cfg_sum.head(20).to_string(index=False))
    print(f"Config section: {time.perf_counter()-t2:.0f}s")

    all_sum = pd.concat([
        fast_sum.assign(section="fast"),
        cfg_sum.assign(section="config"),
    ]).sort_values("delta_mean")
    all_sum.to_csv("phase_f_si_all.csv", index=False)

    print("\n=== TOP 25 overall ===")
    print(all_sum.head(25).to_string(index=False))
    print(f"\nTotal time: {time.perf_counter()-t0:.0f}s")
    print("Saved: phase_f_si_all.csv, phase_f_si_fast.csv, phase_f_si_config.csv")


if __name__ == "__main__":
    main()
