"""
Phase R — cc50_blend_w / cc50_cat_w поверх cc50_blend_w70 (public 268.60).

Кандидаты:
  ref_blend_w70       — baseline
  cc50_blend_w72/75/78/80
  cc50_cat_w0/10/20/25  — при blend_w=0.70
  cc50_w75_cat_w20    — combo

Запуск:
  python run_phase_r.py --quick
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import pandas as pd

from phase_k_fe import engineer_features
from run_local_signal_search import clear_fold_cache, competition_score, ensure_data, load_data, per_target_rmse
from run_phase_b_si_ic50 import SEEDS, load_ext_cols
from run_phase_e_structural import feature_blocks
from run_phase_i_kaggle_ideas import blend_ic50_full, fit_phase_h
from run_phase_j import FULL_CB_W, blend_ic50_lgb

LGB_W = 0.55
BLEND_W = 0.70


@dataclass
class PhaseRCfg:
    name: str
    cc50_blend_w: float = BLEND_W
    cc50_cat_w: float | None = None
    lgb_w: float = LGB_W


def fit_phase_r(
    X: pd.DataFrame,
    Xt: pd.DataFrame,
    y,
    ext,
    blocks,
    seed: int,
    cfg: PhaseRCfg,
) -> tuple:
    Xh = engineer_features(X, ratios=True)
    Xth = engineer_features(Xt, ratios=True)

    oof, test = fit_phase_h(
        X, Xt, y, ext, blocks, seed,
        X_trans=Xh, Xt_trans=Xth,
        cc50_blend_w=cfg.cc50_blend_w,
        cc50_cat_w=cfg.cc50_cat_w,
    )
    oof, test = blend_ic50_full(X, Xt, y, oof, test, FULL_CB_W, seed)
    oof, test = blend_ic50_lgb(Xh, Xth, y, oof, test, cfg.lgb_w, seed)
    return oof, test


CANDIDATES: list[PhaseRCfg] = [
    PhaseRCfg("ref_blend_w70"),
    PhaseRCfg("cc50_blend_w72", cc50_blend_w=0.72),
    PhaseRCfg("cc50_blend_w75", cc50_blend_w=0.75),
    PhaseRCfg("cc50_blend_w78", cc50_blend_w=0.78),
    PhaseRCfg("cc50_blend_w80", cc50_blend_w=0.80),
    PhaseRCfg("cc50_cat_w0", cc50_cat_w=0.0),
    PhaseRCfg("cc50_cat_w10", cc50_cat_w=0.10),
    PhaseRCfg("cc50_cat_w20", cc50_cat_w=0.20),
    PhaseRCfg("cc50_cat_w25", cc50_cat_w=0.25),
    PhaseRCfg("cc50_w75_cat_w20", cc50_blend_w=0.75, cc50_cat_w=0.20),
    PhaseRCfg("cc50_w75_cat_w0", cc50_blend_w=0.75, cc50_cat_w=0.0),
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

    ref_cfg = PhaseRCfg("ref_blend_w70")
    ref_scores: dict[int, float] = {}
    for seed in seeds:
        clear_fold_cache()
        oof, _ = fit_phase_r(X, Xt, y, ext, blocks, seed, ref_cfg)
        ref_scores[seed] = competition_score(y, oof)
    print("REF blend_w70: " + ", ".join(f"s{s}={ref_scores[s]:.2f}" for s in seeds))

    rows: list[dict] = []
    for cfg in CANDIDATES:
        print(f"  {cfg.name}...", flush=True, end=" ")
        for seed in seeds:
            clear_fold_cache()
            oof, _ = fit_phase_r(X, Xt, y, ext, blocks, seed, cfg)
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
    sm.to_csv("phase_r.csv", index=False)
    df.to_csv("phase_r.detail.csv", index=False)

    print(f"\n=== Phase R ({time.perf_counter()-t0:.0f}s) ===")
    print(sm.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print("\nPublic ref: cc50_blend_w70 = 268.60")
    print("  python make_submission_phase_r.py cc50_blend_w75")


if __name__ == "__main__":
    main()
