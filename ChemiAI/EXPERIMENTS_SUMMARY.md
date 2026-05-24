# ChemiAI — отчёт по экспериментам (актуально: май 2026)

## TL;DR

| | |
|---|---|
| **Public best** | **268.16320** |
| **Файл** | `submission_phase_u_si_a32.csv` |
| **Скрипт** | `python make_submission_phase_u.py si_a32` |
| **Ноутбук** | `solution_best.ipynb` |

Прогресс: clustering ~**306** → **268.16** (−**37.8** public).

---

## Текущая формула (best)

```text
IC50 base = ic65(ext CatBoost) + clustering + transductive(192+4 ratio fe, k=3, w=0.70)
Phase H   = morgan25( mord55( fr42( IC50_base ) ) )
Phase I     = 0.75·phase_h + 0.25·CatBoost(all 192)
Phase K     = 0.45·phase_i + 0.55·LightGBM(192 + 4 ratio fe)

CC50 = 0.72·(0.30·cluster + 0.70·transductive(192+4fe))
     + 0.28·CatBoost(192+4fe, log1p)

SI = 0.32·SI_ml + 0.68·(CC50/IC50)     # α=0.32 (Phase U)
```

**4 ratio fe (IC50/LGB + CC50):** `fe_logp_tpsa`, `fe_tpsa_molwt`, `fe_bertz_tpsa`, `fe_fcsp3_tpsa`

**Константы:** ic65, fr42, mord55, morgan25, full_cb25, lgb55, cc50_blend_w=**0.70**, cc50_cat_w=**0.28**, SI α=**0.32**

---

## OOF vs Public (seed=42, best pipeline)

| Таргет | OOF RMSE | Доля в mean | Train std |
|--------|----------|-------------|-----------|
| SI | ~788 | **50%** | 789 |
| CC50 | ~458 | 29% | 642 |
| IC50 | ~324 | 21% | 370 |
| **Mean** | **~523** | | |

**Паттерн:** OOF для IC50/CC50-heads систематически **занижает** public gain; для SI **learned blend** даёт OOF −8.6 и public **+44**.

---

## Эволюция public score (ключевые вехи)

| Public | Фаза | Файл | Δ vs prev |
|--------|------|------|-----------|
| 306.26 | clustering | `submission_clustering.csv` | — |
| 292.06 | SI robust + inv | `submission_combo_exp3_inv_a35.csv` | |
| 284.61 | IC50 size CatBoost | `submission_ic50_size_catboost_w25.csv` | |
| 274.76 | fr42 + ic65 | `submission_phase_e_fr42_ic65.csv` | |
| 272.44 | morgan w25 | `submission_phase_h_morgan_w25.csv` | |
| 272.17 | full_cb25 | `submission_phase_i_full_cb25.csv` | |
| 269.43 | LGB IC50 head | `submission_phase_j_lgb_ic42.csv` | −2.7 |
| 269.09 | ratio fe + LGB w55 | `submission_phase_k_ratio_lgb55.csv` | −0.34 |
| 268.82 | CC50 transductive fe | `submission_phase_p_cc50_fe_lgb55.csv` | −0.27 |
| 268.60 | cc50_blend_w70 | `submission_phase_q_cc50_blend_w70.csv` | −0.21 |
| 268.23 | CC50 cb on fe | `submission_phase_s_cc50_cb_fe.csv` | −0.37 |
| 268.19 | cc50_cat_w28 | `submission_phase_t_cc50_cat_w28.csv` | −0.04 |
| **268.16** | **SI α=0.32** | **`submission_phase_u_si_a32.csv`** | **−0.03** |

---

## Phase J–V (детально)

### Phase J — LGB IC50 head
- **Best:** lgb_ic42 → **269.43** (−2.74 vs Phase I)
- OOF хуже ref; public gain подтверждён

### Phase K — ratio fe + LGB w-tune
- **Best:** ratio_lgb55 → **269.09**
- w≥0.58 насыщение; log1p+Ro5 combo **+0.33** — закрыто

### Phase P — PDF fe (fe_v2, ro5, mord, cc50_fe)
| Сабмит | Public | Итог |
|--------|--------|------|
| cc50_fe_lgb55 | 268.82 | ✅ CC50 trans fe работает |
| fe_v2_lgb55 | 269.16 | ✗ |
| ro5_head | 269.74 | ✗ |
| combo_mord_ro5 | 269.81 | ✗ |

### Phase Q — CC50 blend_w tune
| w | Public |
|---|--------|
| 0.70 | **268.60** ✅ |
| 0.65 | 268.66 |
| 0.72–0.78 | хуже |

### Phase R — blend_w micro (на w70)
- w72/w75/w78 все хуже w70 — **закрыто**

### Phase S — CC50 fe в других ветках
| Сабмит | Public |
|--------|--------|
| cc50_cb_fe | **268.23** ✅ |
| cc50_all_fe | 268.26 |
| lgb_w52 | 268.62 |

### Phase T — cc50_cat_w tune (cb_fe)
| cat_w | Public |
|-------|--------|
| 0.28 | **268.19** ✅ |
| 0.30 | 268.20 |
| 0.35 | 268.33 (OOF ловушка) |

### Phase U — SI α + CC50 fe variants
| Сабмит | Public |
|--------|--------|
| **si_a32** | **268.163** ✅ |
| si_a38 | 268.248 |
| cc50_fe_cc50only | 268.663 |

### Phase V — новый сигнал (KNN / physchem / SI meta)
| Сабмит | Public | OOF Δ |
|--------|--------|-------|
| si_meta_ridge | **312.70** 💀 | −8.6 |
| si_meta_huber | 269.47 | −1.3 |
| cc50_knn_k4 | 270.21 | −0.11 |
| physchem* | — | +5…+16 |

**Вывод Phase V:** SI Ridge meta — **закрыт навсегда** (класс isotonic/stack). CC50-only KNN и physchem-subset — закрыты.

---

## Закрытые ветки (не повторять)

### SI
- Жёсткий SI = CC50/IC50 → public **360**
- SI CatBoost top-k → **+0.43** public
- SI transductive, tail-aware, topology → регресс
- **SI Ridge/Huber meta** (Phase V) → **+44** public при OOF −8.6
- α=0.38 хуже α=0.32 на LB

### IC50
- Stack Ridge → **317**
- pIC50, isotonic → регресс
- fe_v2, ro5, mord fe heads (Phase P)
- LGB w>0.55 насыщение
- Pair interactions (+0.70), LGB morgan (+0.36)

### CC50
- cc50_trans_k4 на LB → **270.69** (Phase Q)
- blend_w > 0.70
- cc50_fe_only, pca_n≠20, clust_fe solo
- cc50_fe_cc50only, cb_lgb
- physchem transductive (Phase V)

### Прочее
- GNN / ChemBERTa — нет SMILES
- Seed full ensemble, minimal CB multitarget
- full_cb_w ≠ 0.25 (Phase M)

---

## Что ещё можно (микро, ~0.01–0.03)

- **SI α=0.30, 0.31** — единственный безопасный SI-слот
- **Ensemble** avg(si_a32 + cat_w28) — иногда −0.02 на LB
- **CatBoost cb hyper** (depth/iters) на frozen rest

**−2…−3 до 265** без SMILES после Phase V — **нереалистично**. Потолок ~**268.0–268.16**.

---

## Ключевые файлы

| Файл | Назначение |
|------|------------|
| `solution_best.ipynb` | Воспроизводимый best-пайплайн |
| `make_solution_best_nb.py` | Генератор ноутбука |
| `phase_k_fe.py` | ratio fe, physchem frames |
| `run_local_signal_search.py` | fit_oof, transductive, SI meta |
| `run_phase_u.py` / `make_submission_phase_u.py` | текущий best |
| `run_phase_v.py` / `make_submission_phase_v.py` | Phase V (закрыта) |
| `pipeline_core.py` | legacy Phase F core |

---

## Воспроизведение best

```bash
pip install -r requirements.txt
# data/train.csv, data/test.csv в data/

python make_submission_phase_u.py si_a32
# → submission_phase_u_si_a32.csv

# или Kernel → Run All в solution_best.ipynb
```

Зависимости: numpy, pandas, scikit-learn, catboost, **lightgbm**, matplotlib, seaborn.
