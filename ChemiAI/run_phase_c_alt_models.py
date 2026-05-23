"""Альтернативные базовые модели: target-wise CatBoost/LGBM blend с clustering."""
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from run_local_signal_search import (
    N_SPLITS,
    N_JOBS,
    build_clustering_features,
    clear_fold_cache,
    competition_score,
    enforce_si_invariant,
    ensure_data,
    fit_oof,
    load_data,
    per_target_rmse,
)
from run_phase_b_si_ic50 import SEEDS, frozen_cc50, load_ext_cols

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def fit_alt_blend_oof(
    X_train, X_test, y_train,
    alt_predict_fn,
    blend_weights: tuple[float, float, float],
    si_alpha: float = 0.35,
    si_robust_fn=None,
    random_state: int = 42,
) -> np.ndarray:
    """alt_predict_fn(X_fit, X_valid) -> (valid_pred, test_pred) shape (n,3)."""
    n = len(X_train)
    oof = np.zeros((n, 3))
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)

    for train_idx, valid_idx in kf.split(X_train):
        X_fit = X_train.iloc[train_idx]
        X_valid = X_train.iloc[valid_idx]
        y_fit = y_train[train_idx]

        cl_v = np.zeros((len(valid_idx), 3))
        cl_t = np.zeros((len(X_test), 3))
        X_fit_aug, X_valid_aug = build_clustering_features(X_fit, X_valid, random_state=random_state)
        _, X_test_aug = build_clustering_features(X_fit, X_test, random_state=random_state)
        for t in range(3):
            m = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.03, max_iter=700, random_state=random_state)
            m.fit(X_fit_aug, np.log1p(y_fit[:, t]))
            cl_v[:, t] = np.expm1(np.clip(m.predict(X_valid_aug), 0, 12))
            cl_t[:, t] = np.expm1(np.clip(m.predict(X_test_aug), 0, 12))
        cl_v = np.clip(cl_v, 0, None)
        alt_v, _ = alt_predict_fn(X_fit, X_valid, y_fit, random_state)

        fold = cl_v.copy()
        for t in range(3):
            w = blend_weights[t]
            fold[:, t] = (1 - w) * cl_v[:, t] + w * np.clip(alt_v[:, t], 0, None)

        if si_robust_fn is not None:
            si_alt = si_robust_fn(X_fit, X_valid, y_fit, random_state)
            fold[:, 2] = 0.7 * cl_v[:, 2] + 0.3 * si_alt

        fold = enforce_si_invariant(fold, alpha=si_alpha)
        oof[valid_idx] = fold
    return oof


def catboost_full_predict(X_fit, X_valid, y_fit, random_state):
    valid = np.zeros((len(X_valid), 3))
    for t in range(3):
        m = CatBoostRegressor(
            depth=6, learning_rate=0.03, iterations=500, verbose=False,
            random_seed=random_state, thread_count=-1,
        )
        m.fit(X_fit, np.log1p(y_fit[:, t]), verbose=False)
        valid[:, t] = np.expm1(np.clip(m.predict(X_valid), 0, 12))
    return np.clip(valid, 0, None), None


def lgbm_full_predict(X_fit, X_valid, y_fit, random_state):
    valid = np.zeros((len(X_valid), 3))
    for t in range(3):
        m = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=6,
            random_state=random_state, n_jobs=N_JOBS, verbose=-1,
        )
        m.fit(X_fit, np.log1p(y_fit[:, t]))
        valid[:, t] = np.expm1(np.clip(m.predict(X_valid), 0, 12))
    return np.clip(valid, 0, None), None


def main() -> None:
    ensure_data()
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    ref = frozen_cc50(ext, ic50_cat_w=0.40)

    t0 = time.perf_counter()
    ref_scores = {}
    for seed in SEEDS:
        clear_fold_cache()
        oof, _ = fit_oof(X, Xt, y, ref, random_state=seed)
        ref_scores[seed] = competition_score(y, oof)

    candidates = []
    for w in [0.15, 0.25, 0.35, 0.50]:
        candidates.append((f"catboost_full_w{int(w*100)}", w, "catboost"))
    if HAS_LGB:
        for w in [0.25, 0.35, 0.50]:
            candidates.append((f"lgbm_full_w{int(w*100)}", w, "lgbm"))

    print(f"ref w40: {ref_scores}")
    print(f"{'name':<22} " + " ".join(f"s{s}" for s in SEEDS) + "  mean_d wins")
    for name, w, kind in candidates:
        fn = catboost_full_predict if kind == "catboost" else lgbm_full_predict
        bw = (w, w, w)
        scores = []
        for seed in SEEDS:
            oof = fit_alt_blend_oof(X, Xt, y, fn, bw, random_state=seed)
            scores.append(competition_score(y, oof))
        deltas = [scores[i] - ref_scores[SEEDS[i]] for i in range(len(SEEDS))]
        wins = sum(1 for d in deltas if d < -0.1)
        print(f"{name:<22} " + " ".join(f"{s:6.2f}" for s in scores) + f"  {sum(deltas)/3:+6.2f}  {wins}/3")

    print(f"Время: {time.perf_counter()-t0:.0f}s, LGBM={HAS_LGB}")


if __name__ == "__main__":
    main()
