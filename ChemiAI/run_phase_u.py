"""
Phase U — SI soft + CC50-specific fe + LGB cb-head (public 268.19).

Кандидаты:
  ref_cat_w28         — baseline (cb_fe, cat_w=0.28)
  si_a32 / si_a38     — мягкий SI alpha
  cc50_fe_cc50only    — CC50 fe в trans+cb, IC50 fe без изменений
  cc50_cb_lgb         — LightGBM вместо CatBoost в CC50-head

Запуск:
  python run_phase_u.py --quick
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from phase_k_fe import engineer_features, engineer_features_cc50
from run_local_signal_search import clear_fold_cache, competition_score, ensure_data, load_data, per_target_rmse
from run_phase_b_si_ic50 import SEEDS, load_ext_cols
from run_phase_e_structural import feature_blocks
from run_phase_i_kaggle_ideas import blend_ic50_full, fit_phase_h
from run_phase_j import FULL_CB_W, blend_ic50_lgb

LGB_W = 0.55
BLEND_W = 0.70
CAT_W = 0.28


@dataclass
class PhaseUCfg:
    name: str
    cc50_cat_w: float = CAT_W
    cc50_blend_w: float = BLEND_W
    si_alpha: float | None = None
    cc50_fe_mode: Literal["ic50", "cc50"] = "ic50"
    cb_lgb: bool = False
    lgb_w: float = LGB_W
    full_cb_w: float = FULL_CB_W


def fit_phase_u(
    X: pd.DataFrame,
    Xt: pd.DataFrame,
    y,
    ext,
    blocks,
    seed: int,
    cfg: PhaseUCfg,
) -> tuple:
    Xh = engineer_features(X, ratios=True)
    Xth = engineer_features(Xt, ratios=True)

    if cfg.cc50_fe_mode == "cc50":
        Xc = engineer_features_cc50(X)
        Xtc = engineer_features_cc50(Xt)
    else:
        Xc, Xtc = Xh, Xth

    oof, test = fit_phase_h(
        X, Xt, y, ext, blocks, seed,
        X_trans=Xc, Xt_trans=Xtc,
        cc50_blend_w=cfg.cc50_blend_w,
        cc50_cat_w=cfg.cc50_cat_w,
        si_alpha=cfg.si_alpha,
        cc50_cb_lgb=cfg.cb_lgb,
        X_cc50_cb=Xc, Xt_cc50_cb=Xtc,
    )
    oof, test = blend_ic50_full(X, Xt, y, oof, test, cfg.full_cb_w, seed)
    oof, test = blend_ic50_lgb(Xh, Xth, y, oof, test, cfg.lgb_w, seed)
    return oof, test


CANDIDATES: list[PhaseUCfg] = [
    PhaseUCfg("ref_cat_w28"),
    PhaseUCfg("si_a32", si_alpha=0.32),
    PhaseUCfg("si_a38", si_alpha=0.38),
    PhaseUCfg("cc50_fe_cc50only", cc50_fe_mode="cc50"),
    PhaseUCfg("cc50_cb_lgb", cb_lgb=True),
    PhaseUCfg("cc50_fe_lgb", cc50_fe_mode="cc50", cb_lgb=True),
]


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

    ref_cfg = PhaseUCfg("ref_cat_w28")
    ref_scores: dict[int, float] = {}
    for seed in seeds:
        clear_fold_cache()
        oof, _ = fit_phase_u(X, Xt, y, ext, blocks, seed, ref_cfg)
        ref_scores[seed] = competition_score(y, oof)
    print("REF cat_w28: " + ", ".join(f"s{s}={ref_scores[s]:.2f}" for s in seeds))

    rows: list[dict] = []
    for cfg in CANDIDATES:
        print(f"  {cfg.name}...", flush=True, end=" ")
        for seed in seeds:
            clear_fold_cache()
            oof, _ = fit_phase_u(X, Xt, y, ext, blocks, seed, cfg)
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
    sm.to_csv("phase_u.csv", index=False)
    df.to_csv("phase_u.detail.csv", index=False)

    print(f"\n=== Phase U ({time.perf_counter()-t0:.0f}s) ===")
    print(sm.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print("\nPublic ref: cc50_cat_w28 = 268.19")
    print("  python make_submission_phase_u.py cc50_fe_cc50only")


if __name__ == "__main__":
    main()
