"""
Phase T — cc50_cat_w tune поверх cc50_cb_fe (public 268.23).

Кандидаты:
  ref_cb_fe           — cat_w=0.25 (baseline)
  cc50_cat_w15/20/30/35
  cc50_cat_w18/22/28  — микро-сетка
  cb_fe_blend_w68/69  — combo с blend_w

Запуск:
  python run_phase_t.py --quick
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
CAT_W = 0.25


@dataclass
class PhaseTCfg:
    name: str
    cc50_cat_w: float = CAT_W
    cc50_blend_w: float = BLEND_W
    cb_fe: bool = True
    lgb_w: float = LGB_W
    full_cb_w: float = FULL_CB_W


def fit_phase_t(
    X: pd.DataFrame,
    Xt: pd.DataFrame,
    y,
    ext,
    blocks,
    seed: int,
    cfg: PhaseTCfg,
) -> tuple:
    Xh = engineer_features(X, ratios=True)
    Xth = engineer_features(Xt, ratios=True)

    oof, test = fit_phase_h(
        X, Xt, y, ext, blocks, seed,
        X_trans=Xh, Xt_trans=Xth,
        cc50_blend_w=cfg.cc50_blend_w,
        cc50_cat_w=cfg.cc50_cat_w,
        X_cc50_cb=Xh if cfg.cb_fe else None,
        Xt_cc50_cb=Xth if cfg.cb_fe else None,
    )
    oof, test = blend_ic50_full(X, Xt, y, oof, test, cfg.full_cb_w, seed)
    oof, test = blend_ic50_lgb(Xh, Xth, y, oof, test, cfg.lgb_w, seed)
    return oof, test


CANDIDATES: list[PhaseTCfg] = [
    PhaseTCfg("ref_cb_fe"),
    PhaseTCfg("cc50_cat_w15", cc50_cat_w=0.15),
    PhaseTCfg("cc50_cat_w18", cc50_cat_w=0.18),
    PhaseTCfg("cc50_cat_w20", cc50_cat_w=0.20),
    PhaseTCfg("cc50_cat_w22", cc50_cat_w=0.22),
    PhaseTCfg("cc50_cat_w28", cc50_cat_w=0.28),
    PhaseTCfg("cc50_cat_w30", cc50_cat_w=0.30),
    PhaseTCfg("cc50_cat_w35", cc50_cat_w=0.35),
    PhaseTCfg("cb_fe_blend_w68", cc50_blend_w=0.68),
    PhaseTCfg("cb_fe_blend_w69", cc50_blend_w=0.69),
    PhaseTCfg("cb_fe_cat_w20_w68", cc50_cat_w=0.20, cc50_blend_w=0.68),
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

    ref_cfg = PhaseTCfg("ref_cb_fe")
    ref_scores: dict[int, float] = {}
    for seed in seeds:
        clear_fold_cache()
        oof, _ = fit_phase_t(X, Xt, y, ext, blocks, seed, ref_cfg)
        ref_scores[seed] = competition_score(y, oof)
    print("REF cb_fe: " + ", ".join(f"s{s}={ref_scores[s]:.2f}" for s in seeds))

    rows: list[dict] = []
    for cfg in CANDIDATES:
        print(f"  {cfg.name}...", flush=True, end=" ")
        for seed in seeds:
            clear_fold_cache()
            oof, _ = fit_phase_t(X, Xt, y, ext, blocks, seed, cfg)
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
    sm.to_csv("phase_t.csv", index=False)
    df.to_csv("phase_t.detail.csv", index=False)

    print(f"\n=== Phase T ({time.perf_counter()-t0:.0f}s) ===")
    print(sm.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print("\nPublic ref: cc50_cb_fe = 268.23")
    print("  python make_submission_phase_t.py cc50_cat_w20")


if __name__ == "__main__":
    main()
