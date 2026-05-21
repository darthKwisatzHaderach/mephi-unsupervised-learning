# ChemiAI — отчет по экспериментам

## TL;DR для команды

Текущий лучший публичный score: **274.76167**.

Лучший файл: `submission_phase_e_fr42_ic65.csv`.

Предыдущий best: **275.30064** (`submission_phase_e_fr_ic50_w35.csv`).

Предыдущий best: **280.13129** (`submission_phase_b_ic50_w35.csv`).

Формула лучшего сабмита:

```text
IC50 = 75% clustering (KMeans k=4 + HGB) + 25% CatBoost(size descriptors)
CC50 = 40% clustering + 60% transductive PCA20 KNN5
SI_direct = 70% clustering + 30% SI-модель (HGB loss=absolute_error)
SI_final  = 0.35 * SI_direct + 0.65 * (CC50 / IC50)   # eps=1e-3
```

Предыдущие вехи:
- **`submission_combo_exp3_inv_a35.csv` → 292.05889** (предыдущий best)
- `submission_exp3_si_absolute_robust.csv` → **293.09177**
- `submission_transductive_neighbor_target_blend.csv` → **298.48843**

Главный вывод: второй крупный скачок дал **CatBoost на size-дескрипторах для IC50** (−7.45 public при −1…−3 OOF). Ранее SI robust head (−6.4) при неизменном IC50.

Что не отправлять:

- `submission_si_tail_aware.csv` — public score `353.06336`, ветка закрыта.
- `submission_si_topology_blend50.csv` — public score `302.12458`, ветка закрыта.
- `submission_cc50_specialist_target_blend.csv` — public score `299.22851`.
- `submission_cc50_specialist_blend.csv` — public score `299.10772` (чуть лучше старого specialist, хуже best на `0.62`).
- `submission_stack_ridge.csv` — public score `317.29305`, ветка закрыта.
- `submission_exp2_nonlinear_geometry_cc50.csv` — `308.01234`.
- `submission_exp5_multitask_mlp_ratio_si.csv` — `360.14158`, жёсткий `SI=CC50/IC50` + MLP, ветка закрыта.
- `submission_e34_ic50cc4_si3.csv` — `293.71572`, exp4 IC50/CC50 + exp3 SI хуже чистого exp3.
- Все `submission_clustering_public_k64_blend_10/15/20.csv` — `k64`-направление ухудшило public уже на `5%`.

## Данные и метрика

- Train: 751 объект.
- Test: 250 объектов.
- Признаки: числовые RDKit-дескрипторы.
- Таргеты в train: `IC50, mM`, `CC50, mM`, `SI`.
- Таргеты в submission: `IC50`, `CC50`, `SI`.
- Метрика: средний RMSE по трем таргетам.

В данных есть важный инвариант:

```text
SI = CC50 / IC50
```

На train это равенство выполняется практически точно. Но жестко заменять `SI` на `pred_CC50 / pred_IC50` нельзя: ошибка в маленьком `IC50` стоит в знаменателе и раздувает ratio.

## Динамика leaderboard

- ~`320.13` — первые стабильные версии.
- `317.78` — seed ensemble.
- `316.15` — дополнительный CatBoost tuning.
- `313.51` — top-k признаки.
- `312.21848` — target-wise CatBoost + top-k.
- `306.26468` — кластерный трек `submission_clustering.csv`.
- `305.44107` — первый target-wise blend с neighbor.
- `304.69450` — усиление neighbor.
- `303.87592` — дальнейшее усиление neighbor.
- `302.93165` — `CC50=24%`, `SI=12%` neighbor.
- `300.82888` — target-вариант raw neighbor.
- **`274.76167` — `submission_phase_e_fr42_ic65.csv`, текущий best.**
- `274.80622` — `submission_phase_e_fr_ic50_w42.csv`.
- `275.30064` — `submission_phase_e_fr_ic50_w35.csv`.
- **`279.91622` — `submission_phase_b_ic50_w40.csv`.**
- `280.13129` — `submission_phase_b_ic50_w35.csv`.
- **`284.60804` — `submission_ic50_size_catboost_w25.csv`, предыдущий best.**
- `292.05889` — `submission_combo_exp3_inv_a35.csv` (предыдущий best).
- `292.11970` — `submission_combo_exp3_inv_a45.csv`.
- `292.16381` — `submission_combo_exp3_inv_a50.csv`.
- `292.27921` — `submission_combo_exp3_inv_a60.csv`.
- `293.09177` — `submission_exp3_si_absolute_robust.csv`.
- `296.70802` — `submission_exp4_cluster_cv_blend.csv`.
- `296.77745` — `submission_exp1_si_invariant_alpha70.csv`.
- `298.48843` — `submission_transductive_neighbor_target_blend.csv` (предыдущий best).
- `299.10772` — `submission_cc50_specialist_blend.csv`.
- `308.01234` — `submission_exp2_nonlinear_geometry_cc50.csv`.
- `317.29305` — `submission_stack_ridge.csv`.
- `360.14158` — `submission_exp5_multitask_mlp_ratio_si.csv`.
- `293.71572` — `submission_e34_ic50cc4_si3.csv` (exp4 IC50/CC50 + exp3 SI, хуже exp3).

## Лучший текущий подход

### 1. Базовый clustering baseline

Файл: `submission_clustering.csv`.

Score: `306.26468`.

Идея: добавить `KMeans` one-hot к исходным дескрипторам и обучить регрессоры на расширенном пространстве.

Почему сработало лучше `solution_v2.ipynb`:

- `solution_v2` оптимизирует индивидуальные признаки через CatBoost/top-k.
- KMeans использует признаки как координаты молекулярной геометрии.
- Признак может быть слабым для CatBoost, но полезным для разделения молекул на локальные режимы.
- Поэтому top-k отбор может выкидывать признаки, важные для кластеризации.

Проверки:

- `k=64 + one-hot + HGB`: OOF `537.41511`.
- `one-hot + distances` не улучшил OOF.
- Вывод: полезен дискретный сегмент молекулы, а не расстояние до центроида.

### 2. Raw neighbor signal

Лучший локальный neighbor:

```text
PCA(20) + KNN(k=8, weights="distance")
OOF: 537.61000
per-target RMSE: [342.01921, 492.19600, 778.61479]
```

Профиль относительно clustering:

- `IC50` хуже.
- `CC50` лучше.
- `SI` почти нейтрален.

Поэтому общий blend не использовался. Мы делали target-wise blend: не трогали `IC50`, добавляли neighbor только в `CC50/SI`.

Лучший raw-neighbor target blend:

```text
submission_clustering_neighbor_target_blend_target.csv
IC50 = 100% clustering
CC50 = 40% clustering + 60% raw neighbor
SI   = 70% clustering + 30% raw neighbor
```

Public score: `300.82888`.

### 3. Transductive neighbor signal

Это лучший на данный момент сигнал.

Идея: обучать `imputer/scaler/PCA` на `train + test` без использования таргетов. Это не использует ответы test, но лучше подгоняет геометрию признаков под реальные test-молекулы.

Лучший transductive neighbor:

```text
PCA(20) + KNN(k=5, weights="distance")
OOF: 538.26189
per-target RMSE: [349.33818, 487.01243, 778.43505]
```

Он хуже общего raw-neighbor OOF, но лучше по `CC50`, а leaderboard уже показал, что `CC50`-направление самое полезное.

Финальный успешный файл:

```text
submission_transductive_neighbor_target_blend.csv
IC50 = clustering baseline
CC50 = 40% clustering baseline + 60% transductive neighbor
SI   = 70% clustering baseline + 30% transductive neighbor
```

Public score: **298.48843**.

OOF grid для `KMeans/HGB IC50 + transductive-neighbor CC50/SI`:

- `CC50=0%, SI=0%`: `537.41511`
- `CC50=20%, SI=10%`: `532.39312`
- `CC50=40%, SI=20%`: `529.53429`
- `CC50=60%, SI=30%`: **`528.95624`**
- `CC50=80%, SI=40%`: `530.68331`
- `CC50=100%, SI=50%`: `534.64154`

Вывод: текущие веса `60/30` близки к локальному optimum. Дальше просто увеличивать weight рискованно.

## Что пробовали

### CatBoost baseline и tuning

- Target-wise CatBoost.
- Обучение в raw и `log1p` шкале.
- `log1p` оказался лучше.
- Seed ensemble дал небольшой прирост.
- Target-wise tuning улучшил ранний baseline.

Лучший CatBoost/top-k результат: **312.21848**.

### Feature importance, SHAP, top-k

Проверяли top-k признаки по aggregated CatBoost importance и SHAP.

CV:

- `top-60`: `477.31870 ± 163.23383`
- `top-90`: `474.32910 ± 165.71511`
- `top-120`: **`472.42535 ± 166.34214`**
- `top-150`: `474.03069 ± 164.19053`
- `top-180`: `472.63745 ± 166.69251`
- `top-210`: `474.23903 ± 165.61606`

Лучший `k=120`, но этот трек уступил clustering/neighbor.

### `k64` clustering refinement

Проверяли `k=64 + one-hot + HGB`.

Положительный blend с public-best ухудшил:

```text
submission_clustering_public_k64_blend_05.csv -> 306.27268
baseline submission_clustering.csv             -> 306.26468
```

Вывод: `k64`-направление не отправлять дальше.

### Физико-химические feature engineering признаки

Проверены:

- нормировки на `MolWt`, `HeavyAtomCount`, `LabuteASA`, `TPSA`, `MolMR`;
- `MolLogP/TPSA`, `HBA/HBD`, rotatable/ring balances;
- VSA-суммы и weighted-bin признаки;
- BCUT/charge/EState spreads;
- группы `fr_*` по химическим семействам.

Для neighbor-трека это не улучшило OOF:

```text
raw PCA(20)+KNN(k=8): 537.61000
лучший domain-feature: 540.50085
```

Вывод: простое физико-химическое агрегирование не дало прироста.

### CC50 specialist

Проверяли отдельные target-specific модели для `CC50`:

- transductive PCA + KNN;
- разные scaler: `StandardScaler`, `PowerTransformer`, `QuantileTransformer`;
- метрики `euclidean`, `manhattan`, `cosine`;
- `PLSRegression`;
- `HistGradientBoosting`, `ExtraTrees`, `RandomForest`.

Лучший локальный `CC50`:

```text
PowerTransformer + PCA(16) + KNN(k=3, manhattan)
CC50 RMSE = 482.39203
```

Public (май 2026, повторная проверка с весами `CC50=60%`, `SI=20%`):

```text
submission_cc50_specialist_blend.csv        -> 299.10772  (новый)
submission_cc50_specialist_target_blend.csv -> 299.22851  (старый)
best (transductive neighbor blend)            -> 298.48843
```

Локально specialist давал −3 OOF и стабильный выигрыш по `CC50` на 3 seed, но public снова хуже best на `~0.62`. Новый файл чуть лучше старого specialist (`−0.12`), но до transductive-neighbor далеко.

Вывод: **ветку закрыть окончательно**. Не тратить оставшиеся сабмиты на дальнейший тюнинг PowerTransformer/KNN для `CC50`.

### SI topology specialist

Нашелся локально сильный SI-сигнал:

```text
topology + StandardScaler + PCA(8) + KNN(k=3)
SI RMSE = 770.30848
```

Но public ухудшился:

```text
submission_si_topology_blend50.csv -> 302.12458
best                               -> 298.48843
```

Вывод: topology-SI не переносится на public. Ветку закрыть.

### Tail-aware SI

Проблема `SI`: редкие экстремальные хвосты.

- `SI >= 1000`: 9 объектов из 751.
- `SI >= 2000`: 5 объектов.
- `SI >= 10000`: 2 объекта.
- Один самый большой объект дает около `51%` squared error при обычной константной модели.

Проверяли:

- классификаторы `SI >= 50/100/300/500/1000`;
- weighted-регрессоры по `log1p(SI)`;
- gated uplift для high-tail объектов.

OOF выглядел очень сильно:

```text
SI RMSE: 778.50 -> 725.50
```

Но public провалился:

```text
submission_si_tail_aware.csv -> 353.06336
```

Вывод: public test не содержит таких high-SI объектов или классификатор поднимает неверные молекулы. Tail-aware uplift не отправлять.

## Закрытые ветки

- `solution_deep_learning.ipynb` / `submission_deep_learning.csv`: `340.45095`.
- `solution_stacking.ipynb` / `submission_stacking.csv`: `342.55968`.
- `PCA + Ridge` stacking: OOF blend с CatBoost нулевой/отрицательный.
- `LightGBM/XGBoost` первый прогон:
  - CatBoost: `536.24101`
  - LightGBM: `542.15088`
  - XGBoost: `543.29186`
- `SI = pred_CC50 / pred_IC50`: ухудшает OOF из-за ошибки в знаменателе.
- `SI` robust/Huber/RepeatedKFold: без заметного public-прироста.
- `k64` blend: ухудшил public.
- `CC50 specialist`: ухудшил public.
- `SI topology`: ухудшил public.
- `SI tail-aware`: сильно ухудшил public.
- `submission_stack_ridge.csv` (`317.29305`): Ridge meta на OOF `[clustering, transductive KNN, ridge_trans]`. Локально лучший рискованный OOF (`~518`, −15 на 3 seed), public катастрофа (+18.8 vs best). Причина: переобучение meta на train-OOF + раздувание SI на test (mean SI ~99 vs ~26 у best).

## Пакет из 5 экспериментальных сабмитов (май 2026)

Скрипт: `make_submission_experiments_batch.py` (нужен `umap-learn`).

| Файл | Гипотеза |
|------|----------|
| `submission_exp1_si_invariant_alpha70.csv` | best + `SI = 0.7*SI + 0.3*CC50/IC50` |
| `submission_exp2_nonlinear_geometry_cc50.csv` | CC50: ансамбль UMAP(16)+Isomap(12)+graph KNN; SI/IC50 как best |
| `submission_exp3_si_absolute_robust.csv` | IC50/CC50=best; SI: HGB `loss=absolute_error` |
| `submission_exp4_cluster_cv_blend.csv` | blend, обучение на 5 cluster-stratified подвыборках |
| `submission_exp5_multitask_mlp_ratio_si.csv` | MLP(128,64) на IC50+CC50; `SI = CC50/IC50` |

### Public scores (май 2026)

| Файл | Score | Δ vs 298.49 |
|------|-------|-------------|
| **`submission_combo_exp3_inv_a35.csv`** | **292.05889** | **−6.43** |
| `submission_combo_exp3_inv_a45.csv` | 292.11970 | −6.37 |
| `submission_combo_exp3_inv_a50.csv` | 292.16381 | −6.32 |
| `submission_combo_exp3_inv_a60.csv` | 292.27921 | −6.21 |
| `submission_exp3_si_absolute_robust.csv` | 293.09177 | −5.40 |
| `submission_exp4_cluster_cv_blend.csv` | 296.70802 | −1.78 |
| `submission_exp1_si_invariant_alpha70.csv` | 296.77745 | −1.71 |
| `submission_transductive_neighbor_target_blend.csv` | 298.48843 | — |
| `submission_exp2_nonlinear_geometry_cc50.csv` | 308.01234 | +9.52 |
| `submission_exp5_multitask_mlp_ratio_si.csv` | 360.14158 | +61.65 |

**Выводы по пакету:**
- **exp3 (SI robust)** — прорыв до 293.09: отдельный SI-head `absolute_error`.
- **exp3 + inv α=0.5** — новый best **292.16**: мягкий физический `CC50/IC50` поверх robust SI (сильнее чем α=0.7 на старом best).
- **exp4 / e34** — не смешивать IC50/CC50 с exp3.
- **exp2 (UMAP/Isomap)** — хуже; нелинейная геометрия для CC50 на public не зашла.
- **exp5 (жёсткий ratio + MLP)** — провал, как и раньше при агрессивном ratio.

## Что делать дальше

1. **База для сабмитов:** `submission_combo_exp3_inv_a35.csv`.

2. **Тюнинг α — остановить.** Прирост `a50→a45→a35` ≈ `0.04–0.06` за сабмит; дальше `a30/a32` — лотерея, не стоит оставшихся попыток (май 2026).
   Также: `exp3_siw25/35` на базе exp3 без invariant.

3. **Не повторять:** e34 (exp4 IC50/CC50), exp2, exp5, stack_ridge, specialist.

4. **Тюнинг exp3:** вес SI blend (20–40%), `epsilon`/глубина HGB, `absolute_error` vs `quantile` если появится LightGBM.

5. Для воспроизведения exp3:

```bash
.venv/bin/python make_submission_experiments_batch.py  # exp3 = submission_exp3_si_absolute_robust.csv
```

## Пакет 12 combo-сабмитов (вокруг exp3)

Скрипт: `make_submission_12_combos.py`.

| Файл | Public | Δ vs best |
|------|--------|-----------|
| **`submission_combo_exp3_inv_a35.csv`** | **292.05889** | **best** |
| `submission_combo_exp3_inv_a45.csv` | 292.11970 | +0.06 |
| `submission_combo_exp3_inv_a50.csv` | 292.16381 | +0.10 |
| `submission_combo_exp3_inv_a60.csv` | 292.27921 | +0.22 |

| Приоритет | Файл | Идея |
|-----------|------|------|
| 1 | `submission_combo_exp3_inv_a50.csv` | exp3 + `SI = 0.5·SI + 0.5·CC50/IC50` |
| 2 | `submission_combo_exp3_inv_a60.csv` | α=0.6 |
| 3 | `submission_combo_exp3_inv_a65.csv` | α=0.65 |
| 4 | `submission_combo_exp3_inv_a75.csv` | α=0.75 |
| 5 | `submission_combo_exp3_inv_a80.csv` | α=0.8 |
| 6 | `submission_combo_exp3_siw25.csv` | SI robust blend 25% |
| 7 | `submission_combo_exp3_siw35.csv` | SI robust blend 35% |
| 8 | `submission_combo_exp3_siw40.csv` | SI robust blend 40% |
| 9 | `submission_combo_exp3_siw20.csv` | SI robust blend 20% |
| 10 | `submission_combo_exp4cc_exp3si.csv` | exp4 IC50/CC50 + exp3 SI |
| 11 | `submission_combo_exp3_si_deep.csv` | SI HGB depth=6, iter=700 |
| 12 | `submission_combo_ens50_exp3_exp4.csv` | 50% exp3 + 50% exp4 |

## Пакет exp3 + exp4 (target-wise, не мешают)

Скрипт: `make_submission_exp3_exp4_combos.py`.

Логика: **exp3** = лучший SI (robust HGB); **exp4** = другой train (cluster-CV) → другие IC50/CC50. Смешиваем по таргетам, не усредняем всё подряд.

| Приоритет | Файл | Схема |
|-----------|------|--------|
| 1 | `submission_e34_ic50cc4_si3.csv` | IC50+CC50 из exp4, SI из exp3 |
| 2 | `submission_e34_ic504_cc503_si3.csv` | IC50/CC50 50–50, SI=exp3 |
| 3 | `submission_e34_ic504_cc503_si3_inv60.csv` | то же + inv SI α=0.6 |
| 4–6 | `ic70cc70`, `ic30cc30`, `ic50cc40` | разные веса exp4 в IC50/CC50 |
| 7–9 | `ic504_cc3`, `ic4_cc503`, `ic4_cc3` | один таргет от exp4 |
| 10–12 | `si3w35`, `inv65`, `ic40 inv55` | SI-тюнинг поверх e4 IC50/CC50 |

### Public (exp3+exp4)

| Файл | Score | Δ vs exp3 (293.09) |
|------|-------|---------------------|
| **`submission_exp3_si_absolute_robust.csv`** | **293.09177** | **best** |
| `submission_e34_ic50cc4_si3.csv` | 293.71572 | +0.62 (хуже) |

**Вывод:** exp4 IC50/CC50 + exp3 SI не улучшает public. Ветку e34 IC50/CC50 закрыть; остаются комбо только вокруг SI (invariant, si_w) на базе exp3.

## Предобработка solution.ipynb → solution_final (OOF, май 2026)

Скрипт: `run_preprocessing_ablation.py`. Пайплайн = полный best (cluster + transductive + robust SI + inv α=0.35).

| Режим | Признаков | Train | OOF score | Δ vs const_only |
|-------|-----------|-------|-----------|-----------------|
| **const_only** (было в final) | 192 | 751 | 530.85 | 0 |
| **const_dup** (как solution.ipynb) | 186 | 751 | 530.67 | −0.17 |
| const_dup + global impute | 186 | 751 | 530.05 | −0.80 |
| agg_train (медиана по dup X) | 186 | **630** | 443.92 | артефакт OOF |

**Вывод (локально):** dup-drop давал −0.17 OOF — шум.

### Public (тот же файл, после dup-drop в ноутбуке)

| Файл | Score | Δ vs best 292.06 |
|------|-------|------------------|
| `submission_combo_exp3_inv_a35.csv` (186 feat, rerun) | **293.20683** | **+1.15** (хуже) |

**Ветка закрыта.** Best остаётся **292.05889** на **192** признаках (только const drop). В `solution_final.ipynb` откат к 192 feat.

## Новый сигнал — IC50 size-CatBoost (май 2026) — **ПОДТВЕРЖДЁН PUBLIC**

Скрипты: `run_new_signal_exploration.py`, `make_submission_ic50_size_catboost.py`, `solution_final.ipynb`.

```text
IC50 = 75% clustering + 25% CatBoost(size)
CC50, SI — без изменений (exp3 + inv α=0.35)
```

| Метрика | Значение |
|---------|----------|
| **Public** | **284.60804** |
| Δ vs 292.06 | **−7.45** |
| OOF seed=42 | 529.73 (−1.12) |
| OOF stability | 3/3 wins (−1.1, −2.0, −2.9) |

Локальный OOF занижал эффект (~−1), public дал **−7** — редкий случай, когда OOF консервативен, а LB перевёл IC50-сигнал в SI через ratio.

**Не сработало в той же сессии:** dup-column drop (293.21 public), CC50 robust, cascade, exp4 blend.

### Тюнинг веса CatBoost (public)

| Файл | Public | Δ vs w25 best |
|------|--------|---------------|
| **`submission_ic50_size_catboost_w25.csv`** | **284.60804** | **best** |
| `submission_ic50_size_catboost_w30.csv` | 285.25786 | +0.65 |
| `submission_ic50_size_catboost_w35.csv` | 285.34732 | +0.74 |

OOF favorил больший `w`, public — **оптимум w=0.25**. Ветка тюнинга `w` **закрыта** (w30, w35 хуже).

## Локальный поиск сигнала — CC50 k=3 + CatBoost 15% (май 2026)

Скрипты: `run_local_signal_search.py`, `run_stability_top_signals.py`, `make_submission_ext_k3_cat15.py`.

База = `solution_final` (IC50 size CatBoost w=0.25, CC50 trans k=5 w=0.60, SI robust + inv α=0.35).

| Кандидат | OOF s42 | s2024 | s7 | mean Δ vs base | wins |
|----------|---------|-------|-----|----------------|------|
| baseline_final | 530.20 | 538.74 | 541.74 | 0 | 0/3 |
| cc50_trans_k3_w60 | 528.79 | 537.66 | 538.57 | **−1.89** | 3/3 |
| cc50_k3_cat15 | 527.66 | 536.00 | 537.09 | **−3.31** | 3/3 |
| **ext_k3_cat15** | 527.65 | 535.36 | 536.38 | **−3.76** | 3/3 |

Формула **ext_k3_cat15** (поверх текущего best):

```text
IC50  = 0.75·clustering + 0.25·CatBoost(MolWt, Chi0-3, Kappa, TPSA, …)  # 18 size/shape фич
CC50  = 0.4·clustering + 0.6·transductive_PCA20_KNN(k=3)
      затем blend: 85%·CC50_base + 15%·CatBoost(full, log1p)
SI    — без изменений (robust HGB + inv α=0.35)
```

**Отличие от Фазы A:** раньше `cc50_catboost` один давал OOF −7 на seed 42 и **901** на seed 2024. Здесь CatBoost только **15%** в CC50 и вместе с **k=3** transductive — стабильно лучше baseline на **всех 3 seed**.

Файлы для сабмита (если останется попытка):

1. `submission_ext_k3_cat15.csv` — приоритет (mean OOF −3.76)
2. `submission_cc50_k3_cat15.csv` — проще, только CC50 (−3.31)

### Public (подтверждено)

| Файл | Public | Δ vs 284.61 |
|------|--------|-------------|
| **`submission_ext_k3_cat15.csv`** | **280.75531** | **best** |
| `submission_ext_k3_cat20.csv` | 280.78404 | +0.03 (хуже, ветка закрыта) |
| `submission_ic50_size_catboost_w25.csv` | 284.60804 | — |

OOF mean Δ ≈ −3.76 на 3 seed — **впервые OOF и LB согласованы** (в отличие от IC50 size, где OOF занижал эффект).

**Новая база для сабмитов:** `ext_k3_cat15` (IC50 ext CatBoost + CC50 k=3 + CatBoost 15%).

### Тюнинг вокруг ext_k3 (локально, после public 280.76)

Скрипт: `run_tune_ext_k3.py`. Сравнение с ref `ext_k3_cat15` на seeds 42/2024/7.

| Кандидат | s42 | s2024 | s7 | mean Δ vs cat15 | wins |
|----------|-----|-------|-----|-----------------|------|
| **ext_k3_cat20** | 527.47 | 535.02 | 536.11 | **−0.26** | 3/3 |
| ext_k3_cat18 | 527.53 | 535.14 | 536.20 | −0.17 | 3/3 |
| ext_k3_cat15 (ref) | 527.65 | 535.36 | 536.38 | 0 | — |
| ext_k3_w55 / w65 | — | — | — | +0.06…+0.08 | 0–1/3 |
| k3_cat15_size (без ext IC50) | 527.66 | 536.00 | 537.09 | +0.46 | 0/3 |

**Следующий сабмит (если есть попытка):** ~~`submission_ext_k3_cat20.csv`~~ — public **280.78404**, хуже cat15 на **+0.03**. **Ветка `cc50_cat_w` закрыта**, оптимум **0.15**.

**Текущий best для сабмитов:** `submission_ext_k3_cat15.csv` (**280.75531**).

### Ускорение локальных прогонов

`run_local_signal_search.py`: кэш fold-компонентов между кандидатами (внутри одного seed), `KNN n_jobs=-1`, `CatBoost thread_count=-1`.

```bash
python run_local_signal_search.py --benchmark   # 14 cfg: 476s -> 36s (~13x)
python run_tune_ext_k3.py                       # ~2-3 мин вместо ~25 мин на 3 seed
```

### Фаза B — SI / IC50 при frozen CC50 cat15 (локально, public ref 280.76)

Скрипты: `run_phase_b_si_ic50.py`, `run_phase_b_top_combos.py`, `make_submission_phase_b.py`.

CC50 заморожен: `k=3`, blend 60%, CatBoost 15%. Ref OOF (3 seed): 527.65 / 535.36 / 536.38.

| Кандидат | mean Δ OOF | wins | Комментарий |
|----------|------------|------|-------------|
| **combo_w35_a45** | **−1.66** | 3/3 | IC50 CatBoost w=0.35 + SI α=0.45 |
| combo_w35_a42 | −1.53 | 3/3 | запасной combo |
| **ic50_cat_w35** | **−1.23** | 3/3 | только IC50, проще |
| si_a45 | −0.41 | 3/3 | только SI α=0.45 |
| ic50_trans_w* | −0.02…−0.09 | 1/3 | нестабильно |
| ic50_size_cols | +0.46 | 0/3 | хуже ext |

**SI robust w** (0.15–0.25) и **α<0.40** — слабее ref; лучший SI-сдвиг **α=0.45** (больше direct SI, меньше ratio).

**Рекомендация сабмита** (1 попытка): ~~`submission_phase_b_combo_w35_a45.csv`~~  
Консервативно: ~~`submission_phase_b_ic50_w35.csv`~~

### Public (фаза B, май 2026)

| Файл | Public | Δ vs 280.76 | Δ vs prev best |
|------|--------|-------------|----------------|
| **`submission_phase_e_fr42_ic65.csv`** | **274.76167** | **best** |
| `submission_phase_e_fr_ic50_w42.csv` | 274.80622 | +0.04 |
| `submission_phase_e_fr42_ic65_cc28.csv` | 274.89742 | +0.14 — cc28 **хуже**, закрыть |
| `submission_phase_e_fr_ic50_w35.csv` | 275.30064 | prev |
| `submission_phase_b_ic50_w50.csv` | ? | OOF −0.86, не сабмитили |

**Вывод:** w55_cc25 подтверждён. OOF −1.61 → public −0.11 (CC50 cat снова занижен OOF). **База = w55_cc25.**

**Выводы (актуально):**
- IC50 w + CC50 cat — рабочий трек, но **~−0.1 public/сабмит**, потолок ≈ 278–279.
- SI α>0.35 — закрыть.
- **Новая база:** `w55_cc25` (IC50 cat 55%, CC50 cat 25%, k3/trans60).

### Тюнинг ic50_cat_w (локально, public ref 280.13)

Скрипт: `run_ic50_w_grid.py`. Ref = w0.35, CC50/SI frozen.

| w | s42 | s2024 | s7 | mean Δ vs w35 | wins |
|---|-----|-------|-----|-----------------|------|
| 0.30 | 527.26 | 534.64 | 535.56 | +0.59 | 0/3 |
| 0.32 | 527.12 | 534.37 | 535.24 | +0.35 | 0/3 |
| 0.33 | 527.05 | 534.24 | 535.09 | +0.23 | 0/3 |
| **0.35** | 526.93 | 533.98 | 534.79 | 0 (public **280.13**) | — |
| **0.37** | 526.81 | 533.74 | 534.50 | **−0.22** | 3/3 |
| **0.38** | 526.75 | 533.62 | 534.36 | **−0.32** | 3/3 |
| **0.40** | 526.65 | 533.39 | 534.08 | **−0.53** | 3/3 |

OOF монотонно улучшается w35→w40 (как раньше IC50 size). Public на w35 подтвердил тренд.

**Следующие сабмиты (фаза C, OOF vs w40):**

| Кандидат | mean Δ OOF | wins | Public |
|----------|------------|------|--------|
| ic50_w40 | 0 | — | **279.91622** |
| **w55_cc25** | **−1.61** | 3/3 | **279.80541** (−0.11) |
| ic50_w50 | −0.86 | 3/3 | ? |
| ic50_w55 | −1.20 | 3/3 | ? |

- **Новая база:** `w55_cc25` — IC50 CatBoost ext **55%**, CC50 k3/trans60/**cat25%**, SI α=0.35.

```bash
python make_submission_phase_b.py w55_cc25
```

**OOF −1.61 → public −0.11** — снова сильное занижение эффекта CC50 cat, но направление верное.

**До LB 264–270 (~10–15 public):** текущий трек IC50/CC50 весов ≈ **−0.1 public/сабмит** → нужен **SI-прорыв**, не ещё w60/w65.

### Фаза D — SI CatBoost top-k (public ref w55_cc25 279.81)

Скрипты: `run_phase_d_si_catboost.py`, `run_phase_d2_si_replace.py`.

| Подход | mean Δ OOF | wins | SI RMSE |
|--------|------------|------|---------|
| blend SI CatBoost **поверх** robust | +0.03…+0.09 | 0/3 | ~787 |
| **replace robust:** 65% cl_SI + 35% CatBoost top-90 | **−0.26** | 3/3 | 787 |
| replace robust w45_k120 | −0.19 | 3/3 | 787 |
| w≥0.70 / MAE | +0.02…+0.30 | 0/3 | 788+ |

**Вывод:** SI CatBoost top-k **не blend**, а **замена** robust HGB. OOF −0.26 → public **+0.43** (хуже w55_cc25). **Ветка закрыта.**

~~Опциональный сабмит: `submission_phase_b_si_cb_w35_k90.csv`~~

### Фаза E — Morgan / Mordred proxy (public ref w55_cc25 279.81)

**SMILES в CSV нет** — настоящие Morgan/Mordred недоступны. Прокси-блоки:
- **Morgan** = `fr_*` (70) + `FpDensityMorgan*` (73 cols)
- **Mordred** = BCUT/Chi/Kappa/VSA/charge/topo (98 cols)

Скрипты: `run_phase_e_structural.py`, `make_submission_phase_e.py`.

| Head | mean Δ OOF | wins | Public |
|------|------------|------|--------|
### Дотюнинг fr_ic50_w35 (public 275.30)

Скрипты: `run_tune_fr_best.py`, `run_tune_fr_top.py`.

Ref = fr_w35 + w55_cc25. Все Δ vs ref.

| Кандидат | mean Δ OOF | wins | Комментарий |
|----------|------------|------|-------------|
| fr_w42 | −0.05 | 2/3 | малый шаг fr 35→42 |
| fr_w40 | −0.05 | 2/3 | |
| **fr42_ic65** | **−0.32** | 2/3 | ext IC50 55→65 |
| fr42_ic65_cc28 | −0.37 | 2/3 | + cc50 cat 28% |
| fr42_ic65_cc30 | −0.38 | 2/3 | seed42 +0.2 |

**SI/combo не трогать** (combo public 276.44).

**Public дотюнинг (май 2026):**

| Файл | Public | Δ vs fr35 |
|------|--------|-----------|
| **fr42_ic65** | **274.76167** | **−0.54** |
| fr42 only | 274.80622 | −0.49 |
| fr42_ic65_cc28 | 274.89742 | −0.40 (cc28 хуже ic65 alone) |

**Формула best:**
```text
w55_cc25 base, но ic50_ext_cat_w=0.65 (было 0.55)
+ fr_* CatBoost IC50 blend w=0.42 (было 0.35)
CC50 cat=0.25, SI не трогать
```

**Закрыто:** cc50 cat 28% поверх ic65, combo SI, fr35 без ic65.

**Следующий локальный тюнинг (если сабмиты есть):** ic50 ∈ {0.62, 0.68, 0.70}, fr ∈ {0.40, 0.45} при cc50=0.25 фикс.

### Почему LB 264–270 — другой порядок задачи

Ref w40 OOF per-target (seed 42): **IC50≈327, CC50≈466, SI≈788**, mean≈527.

- **SI ≈ 50% mean RMSE** — главный потолок. Пока SI OOF ~788, mean OOF не опустится к ~500 (грубая цель public ~265) без **прорыва по SI**, а не ещё +0.5% веса CatBoost.
- Тюнинг IC50 w40→w65 даёт OOF −1.7 (~public −0.3…−0.7 по истории). CC50 cat 25% + w55 ещё −0.4 OOF. Потолок этого трека ≈ **278–279 public**.
- Full CatBoost/LGBM blend — OOF **+10…+13** (хуже). Stack_ridge — провал.

**Ветки для прыжка к 264–270:**
1. ~~Morgan/Mordred~~ — **прокси без SMILES** (фаза E): IC50 −1.84 OOF, combo −2.65. Public TBD.
2. Настоящие Morgan 2048-bit — нужен SMILES (нет в датасете).
3. SI-proxy mordred w35 — −0.81 OOF, слабее IC50.

**Закрыто:** SI α>0.35, full-model blend, CC50 k>4, SI CatBoost top-k (+0.43 public).

## Фаза A — сетка IC50/CC50 (локально, май 2026)

Скрипт: `run_phase_a_ic50_cc50_grid.py`. SI заморожен: robust HGB + invariant α=0.35 (как `solution_final`).

| Кандидат | OOF seed=42 | Δ vs baseline | Stability (seeds 42/2024/7) |
|----------|-------------|---------------|------------------------------|
| `cc50_catboost` | **523.55** | **−7.06** | **НЕТ** — seed 2024 OOF **901** (катастрофа) |
| `ic50_trans_20` | 530.33 | −0.29 | 3/3 wins, но −14 только на seed 2024 |
| `cc50_trans_k3` | 529.70 | −0.91 | 2/3 wins, mean Δ ≈ −1.1 |
| `baseline_final` | 530.61 | 0 | — |
| `ic50_catboost` | 2692 | — | SI взрывается через ratio (мелкий IC50) |
| `cc50_catboost` + log1p | 535.69 | +5.08 | хуже на всех seeds |

**Вывод Фазы A:** жёсткий порог «mean OOF −3 и 2/3 seeds» **не выполнен**. `cc50_catboost` — ложный локальный лидер (overfit seed 42). Единственный опциональный сабмит Фазы B: `submission_phase_a_ic50_trans20.csv` (`make_submission_phase_a_ic50_trans20.py`) — слабый сигнал на seed 42, рискованный из‑за разброса по seeds.

## Файлы, которые важны

- `solution_v2.ipynb` — основной CatBoost/top-k путь, полезен как объяснимый baseline.
- `solution_clustering.ipynb` — clustering/neighbor логика и объяснение, почему она лучше классического top-k.
- `solution_best.ipynb` — предыдущий best (transductive blend, 298.49).
- `solution_final.ipynb` — **текущий best** (IC50 CatBoost size + exp3 SI, 284.61).
- `solution_local_lab.ipynb` — локальная лаборатория для OOF-сравнений, target-wise blend grid и проверки стабильности гипотез без лишних сабмитов.
- `run_preprocessing_ablation.py` — сравнение предобработки solution.ipynb vs final.
- `run_phase_a_ic50_cc50_grid.py` — Фаза A: сетка IC50/CC50 при замороженном SI.
- `make_submission_phase_a_ic50_trans20.py` — опциональный сабмит IC50 20% transductive.
- `EXPERIMENTS_SUMMARY.md` — этот отчет.
- `submission_ic50_size_catboost_w25.csv` — **текущий лучший сабмит**.
- `submission_combo_exp3_inv_a35.csv` — предыдущий best (292.06).
- `submission_exp3_si_absolute_robust.csv` — база SI-head.
- `submission_transductive_neighbor_target_blend.csv` — предыдущий best.
- `make_submission_experiments_batch.py` — генерация exp1–exp5.

