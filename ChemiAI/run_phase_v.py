"""
Phase V — 3 новых направления (public 268.16 si_a32):

  A) CC50-only KNN (transductive только y[:,1])
  B) SI learned blend (Ridge/Huber meta)
  C) Physchem-subset transductive (10 + ratio fe)

+ комбинации и расширенная сетка.

Запуск:
  python run_phase_v.py --quick
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from phase_k_fe import build_physchem_trans_frame, engineer_features
from run_local_signal_search import (
    apply_si_meta_blend,
    clear_fold_cache,
    competition_score,
    ensure_data,
    load_data,
    per_target_rmse,
)
from run_phase_b_si_ic50 import SEEDS, load_ext_cols
from run_phase_e_structural import feature_blocks
from run_phase_i_kaggle_ideas import blend_ic50_full, fit_phase_h
from run_phase_j import FULL_CB_W, blend_ic50_lgb

LGB_W = 0.55
BLEND_W = 0.70
CAT_W = 0.28
SI_A = 0.32

TransMode = Literal["full_fe", "physchem", "physchem_cc50_fe", "physchem_only"]


@dataclass
class PhaseVCfg:
    name: str
    trans_mode: TransMode = "full_fe"
    cc50_trans_only: bool = False
    si_meta: bool = False
    si_meta_huber: bool = False
    si_alpha: float = SI_A
    cc50_trans_k: int | None = None
    cc50_cat_w: float = CAT_W
    cc50_blend_w: float = BLEND_W
    lgb_w: float = LGB_W
    full_cb_w: float = FULL_CB_W


def _build_trans(X: pd.DataFrame, mode: TransMode) -> pd.DataFrame:
    if mode == "full_fe":
        return engineer_features(X, ratios=True)
    if mode == "physchem":
        return build_physchem_trans_frame(X, with_ic50_ratios=True)
    if mode == "physchem_cc50_fe":
        return build_physchem_trans_frame(
            X, with_ic50_ratios=False, with_cc50_ratios=True,
        )
    return build_physchem_trans_frame(
        X, with_ic50_ratios=False, with_cc50_ratios=False,
    )


def fit_phase_v(
    X: pd.DataFrame,
    Xt: pd.DataFrame,
    y,
    ext,
    blocks,
    seed: int,
    cfg: PhaseVCfg,
) -> tuple:
    Xh = engineer_features(X, ratios=True)
    Xth = engineer_features(Xt, ratios=True)
    Xc = _build_trans(X, cfg.trans_mode)
    Xtc = _build_trans(Xt, cfg.trans_mode)

    oof, test = fit_phase_h(
        X, Xt, y, ext, blocks, seed,
        X_trans=Xc, Xt_trans=Xtc,
        cc50_blend_w=cfg.cc50_blend_w,
        cc50_cat_w=cfg.cc50_cat_w,
        cc50_trans_k=cfg.cc50_trans_k,
        cc50_trans_target_only=cfg.cc50_trans_only,
        si_meta_blend=cfg.si_meta,
        si_alpha=None if cfg.si_meta else cfg.si_alpha,
        X_cc50_cb=Xh, Xt_cc50_cb=Xth,
    )
    oof, test = blend_ic50_full(X, Xt, y, oof, test, cfg.full_cb_w, seed)
    oof, test = blend_ic50_lgb(Xh, Xth, y, oof, test, cfg.lgb_w, seed)
    if cfg.si_meta:
        oof, test = apply_si_meta_blend(
            oof, test, y, use_huber=cfg.si_meta_huber,
        )
    return oof, test


CANDIDATES: list[PhaseVCfg] = [
    PhaseVCfg("ref_si_a32"),
    # A — CC50-only KNN
    PhaseVCfg("cc50_knn_only", cc50_trans_only=True),
    PhaseVCfg("cc50_knn_k2", cc50_trans_only=True, cc50_trans_k=2),
    PhaseVCfg("cc50_knn_k4", cc50_trans_only=True, cc50_trans_k=4),
    PhaseVCfg("cc50_knn_k5", cc50_trans_only=True, cc50_trans_k=5),
    PhaseVCfg("cc50_knn_cat_w26", cc50_trans_only=True, cc50_cat_w=0.26),
    PhaseVCfg("cc50_knn_cat_w30", cc50_trans_only=True, cc50_cat_w=0.30),
    # C — physchem transductive
    PhaseVCfg("cc50_trans_physchem", trans_mode="physchem"),
    PhaseVCfg("cc50_trans_physchem_knn", trans_mode="physchem", cc50_trans_only=True),
    PhaseVCfg("cc50_trans_physchem_only", trans_mode="physchem_only"),
    PhaseVCfg("cc50_trans_physchem_cc50fe", trans_mode="physchem_cc50_fe"),
    PhaseVCfg("cc50_trans_physchem_cc50fe_knn", trans_mode="physchem_cc50_fe", cc50_trans_only=True),
    PhaseVCfg("cc50_trans_physchem_k2", trans_mode="physchem", cc50_trans_k=2),
    PhaseVCfg("cc50_trans_physchem_k4", trans_mode="physchem", cc50_trans_k=4),
    # B — SI meta blend
    PhaseVCfg("si_meta_ridge", si_meta=True),
    PhaseVCfg("si_meta_huber", si_meta=True, si_meta_huber=True),
    PhaseVCfg("si_meta_ridge_knn", si_meta=True, cc50_trans_only=True),
    PhaseVCfg("si_meta_ridge_physchem", si_meta=True, trans_mode="physchem"),
    PhaseVCfg("si_meta_huber_physchem", si_meta=True, si_meta_huber=True, trans_mode="physchem"),
    # комбо A+B+C
    PhaseVCfg("combo_knn_physchem", cc50_trans_only=True, trans_mode="physchem"),
    PhaseVCfg("combo_knn_physchem_si_meta", cc50_trans_only=True, trans_mode="physchem", si_meta=True),
    PhaseVCfg("combo_knn_si_meta", si_meta=True, cc50_trans_only=True),
    PhaseVCfg("combo_physchem_si_meta_knn", si_meta=True, trans_mode="physchem", cc50_trans_only=True),
    # SI micro + новый trans
    PhaseVCfg("si_a30_physchem", si_alpha=0.30, trans_mode="physchem"),
    PhaseVCfg("si_a31_knn", si_alpha=0.31, cc50_trans_only=True),
    PhaseVCfg("si_a30_knn_physchem", si_alpha=0.30, cc50_trans_only=True, trans_mode="physchem"),
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

    ref_cfg = PhaseVCfg("ref_si_a32")
    ref_scores: dict[int, float] = {}
    for seed in seeds:
        clear_fold_cache()
        oof, _ = fit_phase_v(X, Xt, y, ext, blocks, seed, ref_cfg)
        ref_scores[seed] = competition_score(y, oof)
    print("REF si_a32: " + ", ".join(f"s{s}={ref_scores[s]:.2f}" for s in seeds))

    rows: list[dict] = []
    for cfg in CANDIDATES:
        print(f"  {cfg.name}...", flush=True, end=" ")
        for seed in seeds:
            clear_fold_cache()
            oof, _ = fit_phase_v(X, Xt, y, ext, blocks, seed, cfg)
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
    sm.to_csv("phase_v.csv", index=False)
    df.to_csv("phase_v.detail.csv", index=False)

    top = sm.head(8)
    print(f"\n=== Phase V ({time.perf_counter()-t0:.0f}s) ===")
    print(sm.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print("\nTop-8 для LB:")
    print(top[["name", "delta_mean", "cc50", "si"]].to_string(index=False))
    print("\nPublic ref: si_a32 = 268.16")
    print("  python make_submission_phase_v.py cc50_knn_only")


if __name__ == "__main__":
    main()
