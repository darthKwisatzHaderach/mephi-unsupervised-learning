"""
Phase P — PDF: fe v2, mordred/ro5 heads, CC50 transductive fe.

Кандидаты:
  ref_lgb55           — baseline ratio_lgb55
  fe_v2_lgb55         — 9 ratio fe (4 старых + 5 PDF) в LGB head
  fe_mord_head_lgb55  — mordred + 4 ratio, w=0.15 → full_cb + lgb55
  ro5_head_lgb55      — Ro5 CatBoost-head w=0.12 → full_cb + lgb55
  cc50_fe_lgb55       — 4 ratio в transductive KNN для CC50
  combo_mord_ro5      — mord_fe + ro5 вместе

Запуск:
  python run_phase_p.py
  python run_phase_p.py --quick
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import pandas as pd

from phase_k_fe import engineer_features, fe_column_names
from run_local_signal_search import clear_fold_cache, competition_score, ensure_data, load_data, per_target_rmse
from run_phase_b_si_ic50 import SEEDS, load_ext_cols
from run_phase_e_structural import blend_extra_head, feature_blocks
from run_phase_i_kaggle_ideas import blend_ic50_full, fit_phase_h
from run_phase_j import FULL_CB_W, blend_ic50_lgb

LGB_W = 0.55
MORD_FE_W = 0.15
RO5_W = 0.12


@dataclass
class PhasePCfg:
    name: str
    ratios: bool = True
    ratios_v2: bool = False
    mord_fe_w: float = 0.0
    ro5_w: float = 0.0
    cc50_fe: bool = False
    lgb_w: float = LGB_W


CANDIDATES: list[PhasePCfg] = [
    PhasePCfg("ref_lgb55"),
    PhasePCfg("fe_v2_lgb55", ratios_v2=True),
    PhasePCfg("fe_mord_head_lgb55", mord_fe_w=MORD_FE_W),
    PhasePCfg("fe_mord_w12", mord_fe_w=0.12),
    PhasePCfg("fe_mord_w18", mord_fe_w=0.18),
    PhasePCfg("ro5_head_lgb55", ro5_w=RO5_W),
    PhasePCfg("ro5_w10", ro5_w=0.10),
    PhasePCfg("ro5_w15", ro5_w=0.15),
    PhasePCfg("cc50_fe_lgb55", cc50_fe=True),
    PhasePCfg("combo_mord_ro5", mord_fe_w=MORD_FE_W, ro5_w=RO5_W),
]


def fit_phase_p(
    X: pd.DataFrame,
    Xt: pd.DataFrame,
    y,
    ext,
    blocks,
    seed: int,
    cfg: PhasePCfg,
) -> tuple:
    Xh = engineer_features(X, ratios=cfg.ratios, ratios_v2=cfg.ratios_v2)
    Xth = engineer_features(Xt, ratios=cfg.ratios, ratios_v2=cfg.ratios_v2)

    X_trans = Xh if cfg.cc50_fe else None
    Xt_trans = Xth if cfg.cc50_fe else None

    oof, test = fit_phase_h(
        X, Xt, y, ext, blocks, seed,
        X_trans=X_trans, Xt_trans=Xt_trans,
    )

    if cfg.mord_fe_w > 0:
        fe_cols = fe_column_names(ratios=True)
        mord_cols = blocks["mordred"] + fe_cols
        oof, test = blend_extra_head(
            Xh, Xth, y, oof, test, mord_cols, 0, cfg.mord_fe_w, seed,
        )

    if cfg.ro5_w > 0:
        Xr = engineer_features(X, ratios=False, ro5_rules=True)
        Xtr = engineer_features(Xt, ratios=False, ro5_rules=True)
        ro5_cols = fe_column_names(ratios=False, ro5_rules=True)
        oof, test = blend_extra_head(
            Xr, Xtr, y, oof, test, ro5_cols, 0, cfg.ro5_w, seed,
        )

    oof, test = blend_ic50_full(X, Xt, y, oof, test, FULL_CB_W, seed)
    oof, test = blend_ic50_lgb(Xh, Xth, y, oof, test, cfg.lgb_w, seed)
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

    ref_scores: dict[int, float] = {}
    for seed in seeds:
        clear_fold_cache()
        oof, _ = fit_phase_p(X, Xt, y, ext, blocks, seed, PhasePCfg("ref_lgb55"))
        ref_scores[seed] = competition_score(y, oof)
    print("REF lgb55: " + ", ".join(f"s{s}={ref_scores[s]:.2f}" for s in seeds))

    rows: list[dict] = []
    for cfg in CANDIDATES:
        print(f"  {cfg.name}...", flush=True, end=" ")
        for seed in seeds:
            clear_fold_cache()
            oof, _ = fit_phase_p(X, Xt, y, ext, blocks, seed, cfg)
            sc = competition_score(y, oof)
            rmse = per_target_rmse(y, oof)
            d = sc - ref_scores[seed]
            rows.append({
                "name": cfg.name,
                "seed": seed,
                "oof": sc,
                "delta": d,
                "ic50": rmse[0],
                "cc50": rmse[1],
                "si": rmse[2],
            })
            print(f"s{seed}={sc:.2f}({d:+.2f})", end=" ")
        print()

    df = pd.DataFrame(rows)
    sm = (
        df.groupby("name")
        .agg(
            oof_mean=("oof", "mean"),
            delta_mean=("delta", "mean"),
            min_delta=("delta", "min"),
            ic50=("ic50", "mean"),
            cc50=("cc50", "mean"),
            si=("si", "mean"),
        )
        .reset_index()
        .sort_values("delta_mean")
    )
    sm.to_csv("phase_p.csv", index=False)
    df.to_csv("phase_p.detail.csv", index=False)

    print(f"\n=== Phase P ({time.perf_counter()-t0:.0f}s) ===")
    print(sm.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print("\nPublic ref: ratio_lgb55 = 269.09")
    print("  python make_submission_phase_p.py fe_v2_lgb55")


if __name__ == "__main__":
    main()
