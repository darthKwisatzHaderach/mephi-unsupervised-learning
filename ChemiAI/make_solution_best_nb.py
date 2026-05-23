"""Генератор solution_best.ipynb — public best 272.17 (full_cb25)."""
import json
from pathlib import Path

NB_PATH = Path("solution_best.ipynb")


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
 "source": source,
        "outputs": [],
        "execution_count": None,
    }


cells = [
    md(
        """# ChemiAI — лучший public пайплайн (272.17)

> Регрессия по RDKit-дескрипторам: IC50, CC50, SI.
> Финальная формула: Phase H (4 IC50-головы) + CatBoost(all 192) w=0.25.

**Автор:** команда ChemiAI

**Дата:** май 2026

**Контакты:** mephi-unsupervised-learning / ChemiAI

**Курс / проект:** ChemiAI Kaggle

**Аннотация:**
Воспроизводимый ноутбук собирает сабмит из `data/train.csv` и `data/test.csv`.
Лучший public score: **272.17** (`submission_phase_i_full_cb25.csv`).
Ключевые компоненты: Phase H (ic65 + fr42 + mord55 + morgan25) + **full CatBoost IC50 w=0.25**, transductive CC50 (k=3), SI-инвариант (α=0.35).

---

## 📋 Оглавление

- [1. Введение](#1-введение-и-контекст)
- [2. Данные](#2-загрузка-и-понимание-данных)
- [3. Методология](#3-методология)
- [4. Эксперименты](#4-эксперименты)
  - [❌ Неудачные подходы](#-неудачные-подходы)
  - [✅ Финальный пайплайн](#-финальный-пайплайн)
- [5. Обучение и submission](#5-обучение-и-формирование-submission)
- [6. Результаты и визуализация](#6-результаты-и-визуализация)
- [7. Заключение](#7-заключение)
- [8. Воспроизведение](#8-воспроизведение)
"""
    ),
    md(
        """## 1. Введение и контекст

Задача — предсказать три таргета для молекул: **IC50**, **CC50**, **SI** (Selectivity Index).

Метрика соревнования — средний RMSE по трём таргетам (меньше — лучше).

На train выполняется инвариант `SI ≈ CC50 / IC50` с численной погрешностью.

Гипотеза исследования: для малого табличного датасета (n=751) выигрывают простые target-wise бленды специализированных голов, а не глубокий стеккинг.

Путь к best: clustering → transductive CC50 → SI robust → IC50 CatBoost → fr → mordred → morgan → **full CatBoost IC50 blend**.

SMILES в CSV нет — Morgan/Mordred реализованы через proxy-блоки RDKit-признаков.

Теперь перейдём к описанию данных и воспроизводимому коду пайплайна.
"""
    ),
    md(
        """## 2. Загрузка и понимание данных

Источник: `data/train.csv`, `data/test.csv`, `data/sample_submission.csv`.

Признаки — числовые RDKit-дескрипторы без SMILES в CSV.

Удаляем только константные столбцы (**192** признака после очистки).

Ветка с удалением дубликатов столбцов (186 признаков) проверена на public и закрыта.
"""
    ),
    code(
        """from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from run_local_signal_search import (
    clear_fold_cache,
    competition_score,
    fit_oof,
    per_target_rmse,
)
from run_phase_b_si_ic50 import frozen_cc50
from run_phase_e_structural import blend_extra_head, feature_blocks
from run_phase_i_kaggle_ideas import blend_ic50_full

RANDOM_STATE = 42
N_SPLITS = 5
N_CLUSTERS = 4
N_JOBS = -1
CATBOOST_THREADS = -1

IC50_CAT_W = 0.65
CC50_CAT_W = 0.25
CC50_TRANS_K = 3
CC50_BLEND_W = 0.60
SI_ROBUST_W = 0.30
SI_ALPHA = 0.35
FR_IC50_W = 0.42
MORDRED_IC50_W = 0.55
MORGAN_IC50_W = 0.25
FULL_IC50_W = 0.25
SI_RATIO_EPS = 1e-3
PUBLIC_BEST_SCORE = 272.17

SIZE_FEATURE_NAMES = [
    "LabuteASA", "MolMR", "Chi0", "MolWt", "ExactMolWt",
    "Kappa1", "HeavyAtomCount", "Kappa2",
]
EXTENDED_IC50_FEATURES = SIZE_FEATURE_NAMES + [
    "Chi1", "Chi2", "Chi3", "Kappa3", "HallKierAlpha", "BertzCT",
    "NumRotatableBonds", "FractionCSP3", "TPSA", "MolLogP",
]
TARGET_COLS = ["IC50, mM", "CC50, mM", "SI"]
SUBMISSION_COLS = ["IC50", "CC50", "SI"]
ID_COL = "index"
DATA_DIR = Path("data")
OUTPUT_SUBMISSION_PATH = Path("submission_phase_i_full_cb25.csv")

%matplotlib inline
sns.set_theme(style="whitegrid")
"""
    ),
    code(
        """train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

feature_cols = [c for c in train.columns if c not in [ID_COL, *TARGET_COLS]]
X_train = train[feature_cols].copy()
y_train = train[TARGET_COLS].values.astype(float)
X_test = test[feature_cols].copy()

const_cols = [c for c in X_train.columns if X_train[c].nunique(dropna=False) <= 1]
X_train = X_train.drop(columns=const_cols)
X_test = X_test.drop(columns=const_cols)

fr_cols = [c for c in X_train.columns if c.startswith("fr_")]
ext_cols = [c for c in EXTENDED_IC50_FEATURES if c in X_train.columns]
_mordred_prefixes = (
    "BCUT2D_", "Chi", "Kappa", "EState", "VSA_", "PEOE_VSA",
    "SMR_VSA", "SlogP_VSA", "MaxPartial", "MinPartial",
    "MaxAbsPartial", "MinAbsPartial", "MaxEState", "MinEState",
    "MaxAbsEState", "MinAbsEState", "HallKier", "BertzCT",
    "BalabanJ", "Ipc", "AvgIpc", "qed", "SPS", "LabuteASA",
    "TPSA", "FractionCSP3", "RingCount", "NumRotatable",
)
mordred_cols = [
    c for c in X_train.columns
    if any(c.startswith(p) for p in _mordred_prefixes)
    and c not in fr_cols
    and not c.startswith("FpDensityMorgan")
]
morgan_cols = [
    c for c in X_train.columns
    if c.startswith("fr_") or c.startswith("FpDensityMorgan")
]
feature_blocks_map = feature_blocks(list(X_train.columns))

assert len(train) == 751
assert len(test) == 250
assert X_train.shape[1] == 192
assert len(ext_cols) == len([c for c in EXTENDED_IC50_FEATURES if c in feature_cols])
assert len(fr_cols) > 0
assert len(mordred_cols) > 0
assert len(morgan_cols) > len(fr_cols)
assert set(morgan_cols) == set(feature_blocks_map["morgan"])

print("Train:", train.shape)
print("Test:", test.shape)
print("Features:", X_train.shape[1])
print("fr_* cols:", len(fr_cols))
print("mordred cols:", len(mordred_cols))
print("morgan cols:", len(morgan_cols))
print("IC50 ext cols:", len(ext_cols))
"""
    ),
    code(
        """fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, col in zip(axes, TARGET_COLS):
    sns.histplot(train[col], bins=30, kde=True, ax=ax, color="#4c72b0")
    ax.set_title(f"Распределение {col} (train)")
    ax.set_xlabel(f"{col} (log-шкала)")
    ax.set_xscale("log")
    ax.set_ylabel("Частота")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """Таргеты имеют тяжёлые правые хвосты — для CC50 и SI используем `log1p` при обучении.

Далее опишем методологию и зафиксируем неудачные ветки, закрытые на leaderboard.
"""
    ),
    md(
        """## 3. Методология

**Out-of-fold (OOF)** — 5-fold CV на train для честной локальной оценки.

**Transductive PCA** — fit imputer/scaler/PCA на `train_fold + test` без таргетов test.

**Target-wise blend** — отдельные веса для IC50, CC50, SI вместо одной метамодели.

**Мягкий SI-инвариант** — бленд прямого SI и ratio `CC50/IC50` с весом α=0.35.

**Phase E fr-head** — CatBoost на `fr_*` (Morgan-proxy) для IC50.

**Phase G mordred-head** — CatBoost на Mordred-proxy дескрипторах для IC50 (w=0.55).

**Phase H morgan-head** — CatBoost на Morgan-proxy (`fr_*` + `FpDensityMorgan*`) для IC50 (w=0.25).

**Phase I full-head** — CatBoost на всех 192 признаках только для IC50 (w=0.25).
"""
    ),
    md(
        """## 4. Эксперименты

### ❌ Неудачные подходы

#### Эксперимент #1: Stack Ridge на OOF-предсказаниях

**Гипотеза:** метамодель скомпенсирует ошибки базовых регрессоров.

**Результат:** public **317.29** vs baseline ~320.

**Почему не сработало:** n=751 слишком мало для стеккинга; базовые модели сильно коррелированы.

**Статус:** 🚫 Закрыто.

#### Эксперимент #2: Жёсткий SI = CC50 / IC50

**Гипотеза:** физический инвариант train переносится на test напрямую.

**Результат:** public **360.14** (MLP + ratio).

**Почему не сработало:** ошибка IC50 в знаменателе раздувает SI.

**Статус:** 🚫 Закрыто.

#### Эксперимент #3: SI CatBoost top-k / combo SI mordred

**Гипотеза:** отдельная CatBoost-голова улучшит SI (dominant RMSE ~788 OOF).

**Результат:** public **280.24** (SI CatBoost), combo SI mordred **276.44** vs fr-only **275.30**.

**Почему не сработало:** OOF переоценивал SI-сигналы; fr-head на IC50 даёт больший public gain.

**Статус:** 🚫 Закрыто.

#### Эксперимент #4: CC50 cat_w=0.28 и ic50 0.68–0.70

**Гипотеза:** дотюнинг весов вокруг best ic65_fr42.

**Результат:** public **274.90–274.91** vs best **274.76**.

**Почему не сработало:** локальный IC50-тюнинг насыщен; OOF не коррелирует с public на этой стадии.

**Статус:** 🚫 Закрыто.

#### Эксперимент #5: SI transductive / mordred-SI (Phase F)

**Гипотеза:** transductive KNN или mordred-head улучшат SI (OOF −1.3).

**Результат:** public **281.36** (si_trans_k5_w35) vs best **274.76**.

**Почему не сработало:** SI OOF не переносится на public; IC50-heads работают иначе.

**Статус:** 🚫 Закрыто.

#### Эксперимент #6: mordred IC50 w>0.55

**Гипотеза:** продолжение тренда ↑ mordred weight после public 272.99 на w55.

**Результат:** OOF хуже ref на w57/w60/w62 (+0.01…+0.03).

**Статус:** 🚫 Закрыто — best **w=0.55**.

#### Эксперимент #7: Morgan IC50 head w>0.25 (Phase H)

**Гипотеза:** третья structural head (Morgan-proxy) продолжит тренд fr → mordred.

**Результат:** public **272.44** на w25; w28–w35 хуже на OOF и public.

**Почему не сработало:** насыщение IC50-стека; OOF для heads занижает public gain.

**Статус:** 🚫 Закрыто — best **morgan w=0.25**.

#### Эксперимент #8: pIC50 для ext CatBoost (Phase I)

**Гипотеза:** QSAR-трансформация pIC50 улучшит ext CatBoost (OOF −1.55, 3/3 seeds).

**Результат:** public **275.43** (`pic50_ext_full20`) vs best **272.17**.

**Почему не сработало:** сдвинул IC50 вниз на test → SI через ratio разъехался; OOF снова обманул.

**Статус:** 🚫 Закрыто.

#### Эксперимент #9: Seed ensemble (Phase H)

**Гипотеза:** усреднение test-предсказаний по 10 seed стабилизирует score.

**Результат:** public **273.06** vs **272.44** (single seed 42).

**Почему не сработало:** heads уже подобраны под seed=42; размывание ухудшает IC50.

**Статус:** 🚫 Закрыто.

#### Эксперимент #10: fr_w45 / ic68 / ic70 при fixed mord55

**Гипотеза:** дотюнинг fr и ic65 после mordred-head.

**Результат:** public **272.88** (fr45) и OOF хуже ref для ic68/70.

**Статус:** 🚫 Закрыто.
"""
    ),
    md(
        """### ✅ Финальный пайплайн

На основе экспериментов зафиксированы три принципа:

1. **Простота > сложность** — фиксированные веса бленда, без метамодели.
2. **IC50 — стек специализированных CatBoost-голов** (ext → fr → mordred → morgan → full).
3. **SI не трогаем** — α=0.35, robust HGB; SI-blends и pIC50 на public ломают score.

```text
Phase H IC50 = morgan25( mord55( fr42( ic65_base ) ) )
Phase I:     IC50 = 0.75*phase_h_ic50 + 0.25*CatBoost(all 192 features)
CC50/SI — без изменений (transductive k=3 + SI α=0.35)
```

Далее — вспомогательные функции (одна функция на ячейку).
"""
    ),
    md(
        """### Метрики соревнования

`competition_score` и `per_target_rmse` импортированы из `run_local_signal_search.py`.
"""
    ),
    md(
        """### Кластеризация и transductive PCA

KMeans one-hot расширяет признаки для HGB; PCA fit на train∪test без таргетов test.
"""
    ),
    code(
        """def build_clustering_features(
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
"""
    ),
    code(
        """def make_transductive_pca_features(
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
"""
    ),
    md(
        """### SI-инвариант и CatBoost

Мягкий бленд SI с ratio; фабрика CatBoost с фиксированными гиперпараметрами.
"""
    ),
    code(
        """def enforce_si_invariant(
    predictions: np.ndarray,
    alpha: float = SI_ALPHA,
    eps: float = SI_RATIO_EPS,
) -> np.ndarray:
    \"\"\"
    Мягкий физический инвариант SI ≈ CC50 / IC50.

    SI_final = α * SI_direct + (1 - α) * CC50 / max(IC50, ε)
    \"\"\"
    result = predictions.copy()
    ic50_safe = np.clip(result[:, 0], eps, None)
    si_ratio = result[:, 1] / ic50_safe
    result[:, 2] = alpha * result[:, 2] + (1.0 - alpha) * si_ratio
    return np.clip(result, 0, None)
"""
    ),
    code(
        """def make_catboost(**kwargs) -> CatBoostRegressor:
    defaults = dict(
        depth=6,
        learning_rate=0.03,
        iterations=500,
        verbose=False,
        thread_count=CATBOOST_THREADS,
    )
    defaults.update(kwargs)
    return CatBoostRegressor(**defaults)
"""
    ),
    code(
        """@dataclass
class PipelineConfig:
    ic50_cat_w: float = IC50_CAT_W
    ic50_cat_cols: list[str] | None = None
    ic50_trans_w: float = 0.0
    cc50_blend_w: float = CC50_BLEND_W
    cc50_trans_k: int = CC50_TRANS_K
    cc50_cat_w: float = CC50_CAT_W
    si_robust_w: float = SI_ROBUST_W
    si_alpha: float = SI_ALPHA
    n_clusters: int = N_CLUSTERS
"""
    ),
    md(
        """### Базовый пайплайн (5-fold OOF)

Target-wise blend: clustering HGB, transductive KNN для CC50, CatBoost для IC50/CC50, robust SI.
"""
    ),
    code(
        """def fit_base_pipeline(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    y_tr: np.ndarray,
    cfg: PipelineConfig,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    \"\"\"5-fold OOF + усреднённые test-предсказания базового пайплайна.\"\"\"
    oof = np.zeros((len(X_tr), 3))
    test_pred = np.zeros((len(X_te), 3))
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    ic50_cols = cfg.ic50_cat_cols or [
        c for c in SIZE_FEATURE_NAMES if c in X_tr.columns
    ]

    for train_idx, valid_idx in kf.split(X_tr):
        X_fit = X_tr.iloc[train_idx]
        X_valid = X_tr.iloc[valid_idx]
        y_fit = y_tr[train_idx]

        X_fit_aug, X_valid_aug = build_clustering_features(
            X_fit, X_valid, cfg.n_clusters, random_state,
        )
        _, X_test_aug = build_clustering_features(
            X_fit, X_te, cfg.n_clusters, random_state,
        )

        cl_valid = np.zeros((len(X_valid), 3))
        cl_test = np.zeros((len(X_te), 3))
        for t in range(3):
            model = HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.03,
                max_iter=700,
                random_state=random_state,
            )
            model.fit(X_fit_aug, np.log1p(y_fit[:, t]))
            cl_valid[:, t] = np.expm1(np.clip(model.predict(X_valid_aug), 0, 12))
            cl_test[:, t] = np.expm1(np.clip(model.predict(X_test_aug), 0, 12))
        cl_valid = np.clip(cl_valid, 0, None)
        cl_test = np.clip(cl_test, 0, None)

        X_fit_pca, X_valid_pca, X_test_pca = make_transductive_pca_features(
            X_fit, X_valid, X_te, random_state=random_state,
        )
        knn = KNeighborsRegressor(
            n_neighbors=cfg.cc50_trans_k,
            weights="distance",
            n_jobs=N_JOBS,
        )
        knn.fit(X_fit_pca, np.log1p(y_fit))
        tr_valid = np.clip(np.expm1(knn.predict(X_valid_pca)), 0, None)
        tr_test = np.clip(np.expm1(knn.predict(X_test_pca)), 0, None)

        ic50_cb_valid = cl_valid[:, 0]
        ic50_cb_test = cl_test[:, 0]
        if cfg.ic50_cat_w > 0 and ic50_cols:
            imp = SimpleImputer(strategy="median")
            Xf = imp.fit_transform(X_fit[ic50_cols])
            Xv = imp.transform(X_valid[ic50_cols])
            Xt = imp.transform(X_te[ic50_cols])
            m_ic = make_catboost(random_seed=random_state)
            m_ic.fit(Xf, y_fit[:, 0], verbose=False)
            ic50_cb_valid = np.clip(m_ic.predict(Xv), 0, None)
            ic50_cb_test = np.clip(m_ic.predict(Xt), 0, None)

        cc50_cb_valid = cl_valid[:, 1]
        cc50_cb_test = cl_test[:, 1]
        if cfg.cc50_cat_w > 0:
            m_cc = make_catboost(random_seed=random_state)
            m_cc.fit(X_fit, np.log1p(y_fit[:, 1]), verbose=False)
            cc50_cb_valid = np.clip(
                np.expm1(np.clip(m_cc.predict(X_valid), 0, 12)), 0, None,
            )
            cc50_cb_test = np.clip(
                np.expm1(np.clip(m_cc.predict(X_te), 0, 12)), 0, None,
            )

        si_rob_valid = cl_valid[:, 2]
        si_rob_test = cl_test[:, 2]
        if cfg.si_robust_w > 0:
            m_si = HistGradientBoostingRegressor(
                max_depth=5,
                learning_rate=0.05,
                max_iter=500,
                loss="absolute_error",
                random_state=random_state,
            )
            m_si.fit(X_fit_aug, np.log1p(y_fit[:, 2]))
            si_rob_valid = np.expm1(np.clip(m_si.predict(X_valid_aug), 0, 12))
            si_rob_test = np.expm1(np.clip(m_si.predict(X_test_aug), 0, 12))
            si_rob_valid = np.clip(si_rob_valid, 0, None)
            si_rob_test = np.clip(si_rob_test, 0, None)

        fold_valid = cl_valid.copy()
        fold_test = cl_test.copy()

        fold_valid[:, 0] = (
            (1 - cfg.ic50_cat_w) * cl_valid[:, 0] + cfg.ic50_cat_w * ic50_cb_valid
        )
        fold_test[:, 0] = (
            (1 - cfg.ic50_cat_w) * cl_test[:, 0] + cfg.ic50_cat_w * ic50_cb_test
        )

        cc50_base_v = (
            (1 - cfg.cc50_blend_w) * cl_valid[:, 1]
            + cfg.cc50_blend_w * tr_valid[:, 1]
        )
        cc50_base_t = (
            (1 - cfg.cc50_blend_w) * cl_test[:, 1]
            + cfg.cc50_blend_w * tr_test[:, 1]
        )
        fold_valid[:, 1] = (
            (1 - cfg.cc50_cat_w) * cc50_base_v + cfg.cc50_cat_w * cc50_cb_valid
        )
        fold_test[:, 1] = (
            (1 - cfg.cc50_cat_w) * cc50_base_t + cfg.cc50_cat_w * cc50_cb_test
        )

        fold_valid[:, 2] = (
            (1 - cfg.si_robust_w) * cl_valid[:, 2]
            + cfg.si_robust_w * si_rob_valid
        )
        fold_test[:, 2] = (
            (1 - cfg.si_robust_w) * cl_test[:, 2]
            + cfg.si_robust_w * si_rob_test
        )

        fold_valid = enforce_si_invariant(fold_valid, alpha=cfg.si_alpha)
        fold_test = enforce_si_invariant(fold_test, alpha=cfg.si_alpha)

        oof[valid_idx] = fold_valid
        test_pred += fold_test / N_SPLITS

    return oof, test_pred
"""
    ),
    md(
        """### Sequential IC50-head (учебная реализация)

Ниже — эквивалент `blend_extra_head()` для понимания логики Phase E/G/H.
В финальном прогоне (§5) вызывается импортированная версия из `run_phase_e_structural.py`.
"""
    ),
    code(
        """def blend_ic50_catboost_head(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    y_tr: np.ndarray,
    base_oof: np.ndarray,
    base_test: np.ndarray,
    feature_cols: list[str],
    weight: float,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    \"\"\"CatBoost IC50-head на произвольном блоке признаков.\"\"\"
    oof = base_oof.copy()
    test = base_test.copy()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    test_acc = np.zeros(len(X_te))

    for train_idx, valid_idx in kf.split(X_tr):
        X_fit = X_tr.iloc[train_idx][feature_cols]
        X_valid = X_tr.iloc[valid_idx][feature_cols]
        y_fit = y_tr[train_idx, 0]

        model = make_catboost(random_seed=random_state)
        model.fit(X_fit, y_fit, verbose=False)
        pred_v = np.clip(model.predict(X_valid), 0, None)
        pred_t = np.clip(model.predict(X_te[feature_cols]), 0, None)

        oof[valid_idx, 0] = (1 - weight) * base_oof[valid_idx, 0] + weight * pred_v
        test_acc += pred_t / N_SPLITS

    test[:, 0] = (1 - weight) * base_test[:, 0] + weight * test_acc
    return oof, test
"""
    ),
    md(
        """## 5. Обучение и формирование submission

Последовательно: base → fr → mordred → morgan → **full CatBoost IC50** (seed=42).

Для побитового совпадения с leaderboard используем те же функции, что и в `run_local_signal_search.py` / `make_submission_phase_i.py`.

Полный прогон занимает несколько минут на CPU.
"""
    ),
    code(
        """%%time
clear_fold_cache()
base_cfg = frozen_cc50(ext_cols, ic50_cat_w=IC50_CAT_W, cc50_cat_w=CC50_CAT_W)
base_oof, base_test = fit_oof(X_train, X_test, y_train, base_cfg, RANDOM_STATE)
blocks = feature_blocks_map

oof_fr, test_fr = blend_extra_head(
    X_train, X_test, y_train, base_oof, base_test,
    blocks["fr_only"], 0, FR_IC50_W, RANDOM_STATE,
)
oof_mord, test_mord = blend_extra_head(
    X_train, X_test, y_train, oof_fr, test_fr,
    blocks["mordred"], 0, MORDRED_IC50_W, RANDOM_STATE,
)
oof_morgan, test_morgan = blend_extra_head(
    X_train, X_test, y_train, oof_mord, test_mord,
    blocks["morgan"], 0, MORGAN_IC50_W, RANDOM_STATE,
)
oof_final, test_final = blend_ic50_full(
    X_train, X_test, y_train, oof_morgan, test_morgan,
    FULL_IC50_W, RANDOM_STATE,
)

oof_score = competition_score(y_train, oof_final)
oof_rmse = per_target_rmse(y_train, oof_final)
print(f"OOF competition score: {oof_score:.2f}")
print(
    "Per-target RMSE:",
    dict(zip(SUBMISSION_COLS, [round(x, 2) for x in oof_rmse])),
)
"""
    ),
    md(
        """Сохраняем submission и сверяем с эталонным файлом leaderboard (max|diff| < 1e-4).
"""
    ),
    code(
        """ref_path = Path("submission_phase_i_full_cb25.csv")
reference_submission = pd.read_csv(ref_path) if ref_path.exists() else None

submission = sample_submission.copy()
submission[SUBMISSION_COLS] = np.clip(test_final, 0, None)
submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)

assert submission.shape == (len(X_test), 4)
assert list(submission.columns) == [ID_COL, *SUBMISSION_COLS]
assert submission[ID_COL].tolist() == test[ID_COL].tolist()
assert (submission[SUBMISSION_COLS] >= 0).all().all()

print(f"Saved: {OUTPUT_SUBMISSION_PATH}")
print(submission[SUBMISSION_COLS].describe().round(2))
print("\\nHead:")
print(submission.head())

if reference_submission is not None:
    diff = np.abs(
        submission[SUBMISSION_COLS].values
        - reference_submission[SUBMISSION_COLS].values
    )
    print(f"\\nСверка с эталоном LB: max|diff|={diff.max():.6f}")
    assert diff.max() < 1e-4, "расхождение с эталонным сабмитом 272.17"
"""
    ),
    md(
        """## 6. Результаты и визуализация

Ключевые вехи public leaderboard (источник: train OOF / Kaggle public).

Лучший результат — **272.17** (Phase H + full CatBoost IC50 w=0.25).

На графике OOF видно, что SI доминирует mean RMSE (~787 vs ~324 IC50).
"""
    ),
    code(
        """milestones = {
    "Clustering": 306.26,
    "SI robust + inv": 292.06,
    "IC50 size CatBoost": 284.61,
    "Phase E fr42_ic65": 274.76,
    "Phase G mordred_w55": 272.99,
    "Phase H morgan_w25": 272.44,
    "Phase I full_cb25": PUBLIC_BEST_SCORE,
}
labels = list(milestones.keys())
scores = list(milestones.values())

fig, ax = plt.subplots(figsize=(8, 4.5))
colors = ["#cccccc"] * (len(labels) - 1) + ["#2d862d"]
ax.barh(labels, scores, color=colors)
ax.set_xlabel("Public RMSE (меньше — лучше)")
ax.set_title("Эволюция public score (ChemiAI)")
ax.invert_yaxis()
for i, s in enumerate(scores):
    ax.text(s + 0.3, i, f"{s:.2f}", va="center", fontsize=9)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """fig, ax = plt.subplots(figsize=(6, 4))
target_names = SUBMISSION_COLS
ax.barh(
    target_names,
    oof_rmse,
    color=["#4c72b0", "#55a868", "#c44e52"],
)
ax.set_xlabel("OOF RMSE")
ax.set_title("OOF RMSE по таргетам (seed=42, best pipeline)")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """SI доминирует OOF (~787 vs ~324 IC50).

SI-ветки (Phase F transductive, CatBoost SI) на public ухудшали score — параметры SI зафиксированы.

## 7. Заключение

Ноутбук воспроизводит лучший public пайплайн без готовых prediction CSV.

```text
phase_h = morgan25( mord55( fr42( ic65_base ) ) )
IC50    = 0.75*phase_h + 0.25*CatBoost(all 192)
```

Public best: **272.17**.

Прогресс: clustering ~306 → **272.17** (−34.1 public).

## 8. Воспроизведение

▶️ Запуск:

1. `pip install -r requirements.txt`
2. Положите CSV в `data/`
3. **Kernel → Restart & Run All**
4. Результат: `submission_phase_i_full_cb25.csv`

Зависимости зафиксированы в `requirements.txt` (numpy, pandas, scikit-learn, catboost, matplotlib, seaborn).
"""
    ),
    code(
        """import sys

try:
    from IPython import get_ipython
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("load_ext", "watermark")
        ip.run_line_magic(
            "watermark",
            "-p numpy,pandas,sklearn,catboost,matplotlib,seaborn --python",
        )
    else:
        raise ImportError
except Exception:
    print("Python:", sys.version.split()[0])
    print("Установите watermark: pip install watermark")
"""
    ),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "cells": cells,
}

NB_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {NB_PATH} ({len(cells)} cells)")
