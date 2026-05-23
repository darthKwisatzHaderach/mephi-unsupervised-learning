"""
Локальный поиск новых сигналов вокруг пайплайна solution_final (public 284.61).
Честный 5-fold OOF, несколько seed для стабильности. Без сабмитов на Kaggle.

Ускорение: n_jobs=-1 (KMeans/KNN), CatBoost thread_count=-1, кэш fold-компонентов.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_SPLITS = 5
N_CLUSTERS = 4
N_JOBS = -1
CATBOOST_THREADS = -1
TARGET_COLS = ["IC50, mM", "CC50, mM", "SI"]
SUBMISSION_COLS = ["IC50", "CC50", "SI"]
ID_COL = "index"
DATA_DIR = Path("data")

SIZE_FEATURE_NAMES = [
    "LabuteASA", "MolMR", "Chi0", "MolWt", "ExactMolWt",
    "Kappa1", "HeavyAtomCount", "Kappa2",
]
EXTENDED_IC50_FEATURES = SIZE_FEATURE_NAMES + [
    "Chi1", "Chi2", "Chi3", "Kappa3", "HallKierAlpha", "BertzCT",
    "NumRotatableBonds", "FractionCSP3", "TPSA", "MolLogP",
]
FR_PREFIX = "fr_"

# Кэш дорогих fold-компонентов между вызовами fit_oof (один процесс).
_FOLD_CACHE: dict[tuple, object] = {}


def clear_fold_cache() -> None:
    _FOLD_CACHE.clear()


def cache_stats() -> int:
    return len(_FOLD_CACHE)


def competition_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmses = [
        np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        for i in range(3)
    ]
    return float(np.mean(rmses))


def per_target_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    return [
        float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        for i in range(3)
    ]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    feature_cols = [c for c in train.columns if c not in [ID_COL, *TARGET_COLS]]
    X_train = train[feature_cols].copy()
    y_train = train[TARGET_COLS].values.astype(float)
    X_test = test[feature_cols].copy()

    const_cols = [c for c in X_train.columns if X_train[c].nunique(dropna=False) <= 1]
    X_train = X_train.drop(columns=const_cols)
    X_test = X_test.drop(columns=const_cols)
    assert X_train.shape[1] == 192, X_train.shape[1]
    return X_train, X_test, y_train, list(X_train.columns)


def build_clustering_features(
    X_fit: pd.DataFrame,
    X_apply: pd.DataFrame,
    n_clusters: int = N_CLUSTERS,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    X_fit_imp = imputer.fit_transform(X_fit)
    X_apply_imp = imputer.transform(X_apply)

    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit_imp)
    X_apply_scaled = scaler.transform(X_apply_imp)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    fit_clusters = kmeans.fit_predict(X_fit_scaled)
    apply_clusters = kmeans.predict(X_apply_scaled)

    X_fit_aug = np.hstack([X_fit_imp, np.eye(n_clusters)[fit_clusters]])
    X_apply_aug = np.hstack([X_apply_imp, np.eye(n_clusters)[apply_clusters]])
    return X_fit_aug, X_apply_aug


def make_transductive_pca_features(
    X_fit_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test_full: pd.DataFrame,
    n_components: int = 20,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit_frame = pd.concat([X_fit_train, X_test_full], axis=0)
    imputer = SimpleImputer(strategy="median")
    fit_imputed = imputer.fit_transform(fit_frame)
    scaler = StandardScaler()
    fit_scaled = scaler.fit_transform(fit_imputed)
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(fit_scaled)

    def transform(frame: pd.DataFrame) -> np.ndarray:
        return pca.transform(scaler.transform(imputer.transform(frame)))

    return transform(X_fit_train), transform(X_valid), transform(X_test_full)


def enforce_si_invariant(
    predictions: np.ndarray,
    alpha: float = 0.35,
    eps: float = 1e-3,
) -> np.ndarray:
    result = predictions.copy()
    ic50_safe = np.clip(result[:, 0], eps, None)
    si_ratio = result[:, 1] / ic50_safe
    result[:, 2] = alpha * result[:, 2] + (1.0 - alpha) * si_ratio
    return np.clip(result, 0, None)


def _catboost(**kwargs) -> CatBoostRegressor:
    defaults = dict(
        depth=6,
        learning_rate=0.03,
        iterations=500,
        verbose=False,
        thread_count=CATBOOST_THREADS,
    )
    defaults.update(kwargs)
    return CatBoostRegressor(**defaults)


def _clustering_fold(
    X_fit: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_fit: np.ndarray,
    n_clusters: int,
    other_k: int | None,
    random_state: int,
    fold_i: int,
) -> tuple[np.ndarray, np.ndarray]:
    key = ("cl", random_state, fold_i, n_clusters, other_k)
    cached = _FOLD_CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    cl_valid = np.zeros((len(X_valid), 3))
    cl_test = np.zeros((len(X_test), 3))
    cluster_ks = [n_clusters]
    if other_k is not None:
        cluster_ks = [n_clusters, other_k]

    for k_cl in cluster_ks:
        w_cl = 1.0 / len(cluster_ks)
        X_fit_aug, X_valid_aug = build_clustering_features(
            X_fit, X_valid, n_clusters=k_cl, random_state=random_state
        )
        _, X_test_aug = build_clustering_features(
            X_fit, X_test, n_clusters=k_cl, random_state=random_state
        )
        for t in range(3):
            model = HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.03,
                max_iter=700,
                random_state=random_state,
            )
            model.fit(X_fit_aug, np.log1p(y_fit[:, t]))
            cl_valid[:, t] += w_cl * np.expm1(np.clip(model.predict(X_valid_aug), 0, 12))
            cl_test[:, t] += w_cl * np.expm1(np.clip(model.predict(X_test_aug), 0, 12))

    cl_valid = np.clip(cl_valid, 0, None)
    cl_test = np.clip(cl_test, 0, None)
    _FOLD_CACHE[key] = (cl_valid, cl_test)
    return cl_valid, cl_test


def _transductive_fold(
    X_fit: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_fit: np.ndarray,
    trans_k: int,
    random_state: int,
    fold_i: int,
) -> tuple[np.ndarray, np.ndarray]:
    key = ("tr", random_state, fold_i, trans_k)
    cached = _FOLD_CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    X_fit_pca, X_valid_pca, X_test_pca = make_transductive_pca_features(
        X_fit, X_valid, X_test, random_state=random_state
    )
    knn = KNeighborsRegressor(
        n_neighbors=trans_k,
        weights="distance",
        n_jobs=N_JOBS,
    )
    knn.fit(X_fit_pca, np.log1p(y_fit))
    tr_valid = np.clip(np.expm1(knn.predict(X_valid_pca)), 0, None)
    tr_test = np.clip(np.expm1(knn.predict(X_test_pca)), 0, None)
    _FOLD_CACHE[key] = (tr_valid, tr_test)
    return tr_valid, tr_test


def _ic50_catboost_fold(
    X_fit: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_fit: np.ndarray,
    size_cols: list[str],
    random_state: int,
    fold_i: int,
) -> tuple[np.ndarray, np.ndarray]:
    cols_key = tuple(size_cols)
    key = ("ic50cb", random_state, fold_i, cols_key)
    cached = _FOLD_CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    imp = SimpleImputer(strategy="median")
    Xf = imp.fit_transform(X_fit[size_cols])
    Xv = imp.transform(X_valid[size_cols])
    Xt = imp.transform(X_test[size_cols])
    m = _catboost(random_seed=random_state)
    m.fit(Xf, y_fit[:, 0], verbose=False)
    valid = np.clip(m.predict(Xv), 0, None)
    test = np.clip(m.predict(Xt), 0, None)
    _FOLD_CACHE[key] = (valid, test)
    return valid, test


def _cc50_catboost_fold(
    X_fit: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_fit: np.ndarray,
    random_state: int,
    fold_i: int,
) -> tuple[np.ndarray, np.ndarray]:
    key = ("cc50cb", random_state, fold_i)
    cached = _FOLD_CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    m = _catboost(random_seed=random_state)
    m.fit(X_fit, np.log1p(y_fit[:, 1]), verbose=False)
    valid = np.clip(np.expm1(np.clip(m.predict(X_valid), 0, 12)), 0, None)
    test = np.clip(np.expm1(np.clip(m.predict(X_test), 0, 12)), 0, None)
    _FOLD_CACHE[key] = (valid, test)
    return valid, test


def _si_robust_fold(
    X_fit: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_fit: np.ndarray,
    n_clusters: int,
    random_state: int,
    fold_i: int,
) -> tuple[np.ndarray, np.ndarray]:
    key = ("sirob", random_state, fold_i, n_clusters)
    cached = _FOLD_CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    X_fit_aug, X_valid_aug = build_clustering_features(
        X_fit, X_valid, n_clusters=n_clusters, random_state=random_state
    )
    _, X_test_aug = build_clustering_features(
        X_fit, X_test, n_clusters=n_clusters, random_state=random_state
    )
    m_si = HistGradientBoostingRegressor(
        max_depth=5,
        learning_rate=0.05,
        max_iter=500,
        loss="absolute_error",
        random_state=random_state,
    )
    m_si.fit(X_fit_aug, np.log1p(y_fit[:, 2]))
    valid = np.clip(np.expm1(np.clip(m_si.predict(X_valid_aug), 0, 12)), 0, None)
    test = np.clip(np.expm1(np.clip(m_si.predict(X_test_aug), 0, 12)), 0, None)
    _FOLD_CACHE[key] = (valid, test)
    return valid, test


def _select_si_topk_cols(
    X_fit: pd.DataFrame,
    y_si: np.ndarray,
    topk: int,
    random_state: int,
) -> list[str]:
    probe = _catboost(random_seed=random_state, iterations=300)
    probe.fit(X_fit, np.log1p(y_si), verbose=False)
    imp = probe.get_feature_importance()
    order = np.argsort(imp)[::-1]
    k = min(topk, len(order))
    return X_fit.columns[order[:k]].tolist()


def _si_catboost_topk_fold(
    X_fit: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_fit: np.ndarray,
    topk: int,
    use_mae: bool,
    random_state: int,
    fold_i: int,
) -> tuple[np.ndarray, np.ndarray]:
    key = ("sicb", random_state, fold_i, topk, use_mae)
    cached = _FOLD_CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    cols = _select_si_topk_cols(X_fit, y_fit[:, 2], topk, random_state)
    cb_kw = dict(random_seed=random_state)
    if use_mae:
        cb_kw["loss_function"] = "MAE"
    m = _catboost(**cb_kw)
    m.fit(X_fit[cols], np.log1p(y_fit[:, 2]), verbose=False)
    valid = np.clip(np.expm1(np.clip(m.predict(X_valid[cols]), 0, 12)), 0, None)
    test = np.clip(np.expm1(np.clip(m.predict(X_test[cols]), 0, 12)), 0, None)
    _FOLD_CACHE[key] = (valid, test)
    return valid, test


@dataclass
class PipelineConfig:
    ic50_cat_w: float = 0.25
    ic50_cat_cols: list[str] | None = None  # None -> SIZE
    ic50_trans_w: float = 0.0
    cc50_blend_w: float = 0.60
    cc50_trans_k: int = 5
    cc50_cat_w: float = 0.0
    si_robust_w: float = 0.30
    si_alpha: float = 0.35
    si_catboost_w: float = 0.0
    si_topk: int = 120
    si_catboost_mae: bool = False
    n_clusters: int = 4
    cluster_blend_other_k: int | None = None


def fit_oof(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    cfg: PipelineConfig,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (oof, test_pred) shape (n_train, 3)."""
    n_train = len(X_train)
    oof = np.zeros((n_train, 3))
    test_pred = np.zeros((len(X_test), 3))
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)

    size_cols = cfg.ic50_cat_cols or [c for c in SIZE_FEATURE_NAMES if c in X_train.columns]

    for fold_i, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
        X_fit = X_train.iloc[train_idx]
        X_valid = X_train.iloc[valid_idx]
        y_fit = y_train[train_idx]

        cl_valid, cl_test = _clustering_fold(
            X_fit, X_valid, X_test, y_fit,
            cfg.n_clusters, cfg.cluster_blend_other_k, random_state, fold_i,
        )
        tr_valid, tr_test = _transductive_fold(
            X_fit, X_valid, X_test, y_fit, cfg.cc50_trans_k, random_state, fold_i,
        )

        ic50_cb_valid = cl_valid[:, 0]
        ic50_cb_test = cl_test[:, 0]
        if cfg.ic50_cat_w > 0 and size_cols:
            ic50_cb_valid, ic50_cb_test = _ic50_catboost_fold(
                X_fit, X_valid, X_test, y_fit, size_cols, random_state, fold_i,
            )

        cc50_cb_valid = cl_valid[:, 1]
        cc50_cb_test = cl_test[:, 1]
        if cfg.cc50_cat_w > 0:
            cc50_cb_valid, cc50_cb_test = _cc50_catboost_fold(
                X_fit, X_valid, X_test, y_fit, random_state, fold_i,
            )

        si_rob_valid = cl_valid[:, 2]
        si_rob_test = cl_test[:, 2]
        if cfg.si_robust_w > 0:
            si_rob_valid, si_rob_test = _si_robust_fold(
                X_fit, X_valid, X_test, y_fit, cfg.n_clusters, random_state, fold_i,
            )

        fold_valid = cl_valid.copy()
        fold_test = cl_test.copy()

        w_ic = cfg.ic50_cat_w
        fold_valid[:, 0] = (1 - w_ic) * cl_valid[:, 0] + w_ic * ic50_cb_valid
        fold_test[:, 0] = (1 - w_ic) * cl_test[:, 0] + w_ic * ic50_cb_test

        w_tr_ic = cfg.ic50_trans_w
        if w_tr_ic > 0:
            fold_valid[:, 0] = (1 - w_tr_ic) * fold_valid[:, 0] + w_tr_ic * tr_valid[:, 0]
            fold_test[:, 0] = (1 - w_tr_ic) * fold_test[:, 0] + w_tr_ic * tr_test[:, 0]

        w_cc = cfg.cc50_blend_w
        w_cc_cb = cfg.cc50_cat_w
        cc50_base_v = (1 - w_cc) * cl_valid[:, 1] + w_cc * tr_valid[:, 1]
        cc50_base_t = (1 - w_cc) * cl_test[:, 1] + w_cc * tr_test[:, 1]
        fold_valid[:, 1] = (1 - w_cc_cb) * cc50_base_v + w_cc_cb * cc50_cb_valid
        fold_test[:, 1] = (1 - w_cc_cb) * cc50_base_t + w_cc_cb * cc50_cb_test

        w_si = cfg.si_robust_w
        fold_valid[:, 2] = (1 - w_si) * cl_valid[:, 2] + w_si * si_rob_valid
        fold_test[:, 2] = (1 - w_si) * cl_test[:, 2] + w_si * si_rob_test

        if cfg.si_catboost_w > 0:
            si_cb_v, si_cb_t = _si_catboost_topk_fold(
                X_fit, X_valid, X_test, y_fit,
                cfg.si_topk, cfg.si_catboost_mae, random_state, fold_i,
            )
            w_cb = cfg.si_catboost_w
            fold_valid[:, 2] = (1 - w_cb) * fold_valid[:, 2] + w_cb * si_cb_v
            fold_test[:, 2] = (1 - w_cb) * fold_test[:, 2] + w_cb * si_cb_t

        fold_valid = enforce_si_invariant(fold_valid, alpha=cfg.si_alpha)
        fold_test = enforce_si_invariant(fold_test, alpha=cfg.si_alpha)

        oof[valid_idx] = fold_valid
        test_pred += fold_test / N_SPLITS

    return oof, test_pred


def ensure_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    required = [DATA_DIR / "train.csv", DATA_DIR / "test.csv"]
    if all(p.exists() for p in required):
        return
    try:
        import gdown
    except ImportError as e:
        raise SystemExit("Нет data/ — установите gdown и скачайте папку с Drive") from e
    gdown.download_folder(
        id="1m1PS44rF9HqIAOUZQeopU5sqwgi6YfO8",
        output=str(DATA_DIR),
    )


def run_experiments(seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train, X_test, y_train, all_cols = load_data()
    fr_cols = [c for c in all_cols if c.startswith(FR_PREFIX)]
    ext_cols = [c for c in EXTENDED_IC50_FEATURES if c in all_cols]

    candidates: list[tuple[str, PipelineConfig]] = [
        ("baseline_final", PipelineConfig()),
        ("ic50_ext_cat_w25", PipelineConfig(ic50_cat_cols=ext_cols)),
        ("ic50_fr_cat_w20", PipelineConfig(ic50_cat_w=0.20, ic50_cat_cols=fr_cols)),
        ("ic50_trans_w15", PipelineConfig(ic50_trans_w=0.15)),
        ("ic50_trans_w20", PipelineConfig(ic50_trans_w=0.20)),
        ("cc50_cat_w15", PipelineConfig(cc50_cat_w=0.15)),
        ("cc50_cat_w20", PipelineConfig(cc50_cat_w=0.20)),
        ("cc50_trans_k3_w60", PipelineConfig(cc50_trans_k=3)),
        ("cc50_trans_k7_w60", PipelineConfig(cc50_trans_k=7)),
        ("cc50_blend_w50", PipelineConfig(cc50_blend_w=0.50)),
        ("cc50_blend_w70", PipelineConfig(cc50_blend_w=0.70)),
        ("si_alpha_30", PipelineConfig(si_alpha=0.30)),
        ("si_alpha_32", PipelineConfig(si_alpha=0.32)),
        ("si_alpha_38", PipelineConfig(si_alpha=0.38)),
        ("si_robust_w25", PipelineConfig(si_robust_w=0.25)),
        ("si_robust_w35", PipelineConfig(si_robust_w=0.35)),
        ("cluster_k3_blend", PipelineConfig(cluster_blend_other_k=3)),
        ("cluster_k5_blend", PipelineConfig(cluster_blend_other_k=5)),
        ("ic50_cat_w20", PipelineConfig(ic50_cat_w=0.20)),
        ("ic50_cat_w30", PipelineConfig(ic50_cat_w=0.30)),
        ("combo_ext_ic50_cc15", PipelineConfig(ic50_cat_cols=ext_cols, cc50_cat_w=0.15)),
    ]

    rows = []
    for seed in seeds:
        clear_fold_cache()
        for name, cfg in candidates:
            oof, _ = fit_oof(X_train, X_test, y_train, cfg, random_state=seed)
            score = competition_score(y_train, oof)
            rmse = per_target_rmse(y_train, oof)
            rows.append({
                "candidate": name,
                "seed": seed,
                "oof": score,
                "rmse_ic50": rmse[0],
                "rmse_cc50": rmse[1],
                "rmse_si": rmse[2],
            })

    df = pd.DataFrame(rows)
    summary = df.groupby("candidate").agg(
        oof_mean=("oof", "mean"),
        oof_std=("oof", "std"),
        oof_min=("oof", "min"),
        oof_max=("oof", "max"),
    ).reset_index()

    wins, deltas = [], []
    for cand in summary["candidate"]:
        sub = df[df["candidate"] == cand].set_index("seed")["oof"]
        base = df[df["candidate"] == "baseline_final"].set_index("seed")["oof"]
        d = sub - base
        deltas.append(d.mean())
        wins.append(int((d < 0).sum()))
    summary["delta_vs_base"] = deltas
    summary["wins_of_seeds"] = wins
    return summary.sort_values("oof_mean"), df


def benchmark_tune_grid(seeds: list[int] | None = None) -> None:
    """Сравнение времени run_tune_ext_k3 с кэшем vs без (на одном seed)."""
    from run_tune_ext_k3 import get_candidates

    seeds = seeds or [42]
    X_train, X_test, y_train, all_cols = load_data()
    ext_cols = [c for c in EXTENDED_IC50_FEATURES if c in all_cols]
    candidates = get_candidates(ext_cols)

    def run_all(clear_every_call: bool) -> float:
        t0 = time.perf_counter()
        for seed in seeds:
            if not clear_every_call:
                clear_fold_cache()
            for _, cfg in candidates:
                if clear_every_call:
                    clear_fold_cache()
                fit_oof(X_train, X_test, y_train, cfg, random_state=seed)
        return time.perf_counter() - t0

    t_nocache = run_all(clear_every_call=True)
    t_cache = run_all(clear_every_call=False)
    print(f"run_tune_ext_k3 ({len(candidates)} cfg x {len(seeds)} seed):")
    print(f"  без кэша (clear каждый fit_oof): {t_nocache:.1f}s")
    print(f"  с кэшем (clear только per seed):  {t_cache:.1f}s")
    print(f"  ускорение: {t_nocache / t_cache:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 7])
    parser.add_argument("--benchmark", action="store_true", help="Бенчмарк кэша (run_tune_ext_k3)")
    args = parser.parse_args()
    ensure_data()

    if args.benchmark:
        benchmark_tune_grid([42])
        return

    print("Загрузка данных OK, запуск OOF-сетки...")
    t0 = time.perf_counter()
    summary, detail = run_experiments(args.seeds)
    elapsed = time.perf_counter() - t0
    out = Path("local_signal_search_results.csv")
    detail.to_csv(out.with_suffix(".detail.csv"), index=False)
    summary.to_csv(out, index=False)
    print(f"\nВремя: {elapsed:.1f}s ({len(args.seeds)} seeds, n_jobs={N_JOBS})")
    print("\n=== Топ кандидатов (ниже OOF = лучше) ===")
    print(summary.head(15).to_string(index=False))
    print(f"\nДетали: {out}, {out.with_suffix('.detail.csv')}")


if __name__ == "__main__":
    main()
