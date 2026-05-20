# ChemiAI — отчет по экспериментам

## TL;DR для команды

Текущий лучший публичный score: **298.48843**.

Лучший файл: `submission_transductive_neighbor_target_blend.csv`.

Формула лучшего сабмита:

```text
IC50 = clustering baseline
CC50 = 40% clustering baseline + 60% transductive neighbor
SI   = 70% clustering baseline + 30% transductive neighbor
```

Главный вывод: лучший прирост дали не CatBoost-тюнинг и не feature importance, а **локальная геометрия молекул** через KNN/PCA. Особенно полезным оказался сигнал для `CC50`.

Что не отправлять:

- `submission_si_tail_aware.csv` — public score `353.06336`, ветка закрыта.
- `submission_si_topology_blend50.csv` — public score `302.12458`, ветка закрыта.
- `submission_cc50_specialist_target_blend.csv` — public score `299.22851`, хуже текущего best.
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
- **`298.48843` — transductive neighbor target blend, текущий best.**

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

Но public ухудшился:

```text
submission_cc50_specialist_target_blend.csv -> 299.22851
best                                      -> 298.48843
```

Вывод: ветку закрыть. Локальный OOF лучше, но public test предпочитает старый transductive-neighbor.

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

## Что делать дальше

1. Не трогать текущий best без нового независимого сигнала.

2. Если делать следующий эксперимент, искать новый сигнал для `CC50`, но не через закрытый `PowerTransformer + PCA16 + KNN3 manhattan`.

3. Проверить другой transductive подход:
   - не PCA, а UMAP/Isomap/SpectralEmbedding, если зависимости доступны;
   - несколько transductive PCA-пространств и усреднение только по `CC50`;
   - локальные модели внутри кластеров для `CC50`.

4. `SI` держать консервативно. Public явно не любит агрессивные high-tail коррекции.

5. Для финального сабмита сейчас использовать:

```text
submission_transductive_neighbor_target_blend.csv
```

## Файлы, которые важны

- `solution_v2.ipynb` — основной CatBoost/top-k путь, полезен как объяснимый baseline.
- `solution_clustering.ipynb` — clustering/neighbor логика и объяснение, почему она лучше классического top-k.
- `EXPERIMENTS_SUMMARY.md` — этот отчет.
- `submission_transductive_neighbor_target_blend.csv` — текущий лучший сабмит.

