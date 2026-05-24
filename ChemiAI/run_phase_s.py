"""
Phase S — новые оси поверх cc50_blend_w70 (public 268.60).

Кандидаты:
  ref_w70              — baseline
  cc50_clust_fe        — ratio fe в clustering (30% CC50)
  cc50_cb_fe           — ratio fe в CC50 CatBoost-head
  cc50_all_fe          — clustering + cb + trans на fe
  cc50_pca_n10/n30     — PCA components в transductive
  cc50_blend_w68/w69   — микро blend_w
  lgb_w52 / full_cb_w22 — IC50 микро-tune
  si_a33               — SI alpha (риск)
  ic50_trans_w05       — transductive fe в IC50 (5%)

Запуск:
  python run_phase_s.py --quick
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
class PhaseSCfg:
    name: str
    cc50_blend_w: float = BLEND_W
    clust_fe: bool = False
    cb_fe: bool = False
    cc50_pca_n: int | None = None
    si_alpha: float | None = None
    ic50_trans_w: float | None = None
    lgb_w: float = LGB_W
    full_cb_w: float = FULL_CB_W


def fit_phase_s(
    X: pd.DataFrame,
    Xt: pd.DataFrame,
    y,
    ext,
    blocks,
    seed: int,
    cfg: PhaseSCfg,
) -> tuple:
    Xh = engineer_features(X, ratios=True)
    Xth = engineer_features(Xt, ratios=True)

    X_cl = Xh if cfg.clust_fe else None
    Xt_cl = Xth if cfg.clust_fe else None
    X_cb = Xh if cfg.cb_fe else None
    Xt_cb = Xth if cfg.cb_fe else None

    oof, test = fit_phase_h(
        X, Xt, y, ext, blocks, seed,
        X_trans=Xh, Xt_trans=Xth,
        cc50_blend_w=cfg.cc50_blend_w,
        cc50_pca_n=cfg.cc50_pca_n,
        si_alpha=cfg.si_alpha,
        ic50_trans_w=cfg.ic50_trans_w,
        X_clust=X_cl, Xt_clust=Xt_cl,
        X_cc50_cb=X_cb, Xt_cc50_cb=Xt_cb,
    )
    oof, test = blend_ic50_full(X, Xt, y, oof, test, cfg.full_cb_w, seed)
    oof, test = blend_ic50_lgb(Xh, Xth, y, oof, test, cfg.lgb_w, seed)
    return oof, test


CANDIDATES: list[PhaseSCfg] = [
    PhaseSCfg("ref_w70"),
    PhaseSCfg("cc50_clust_fe", clust_fe=True),
    PhaseSCfg("cc50_cb_fe", cb_fe=True),
    PhaseSCfg("cc50_all_fe", clust_fe=True, cb_fe=True),
    PhaseSCfg("cc50_pca_n10", cc50_pca_n=10),
    PhaseSCfg("cc50_pca_n30", cc50_pca_n=30),
    PhaseSCfg("cc50_blend_w68", cc50_blend_w=0.68),
    PhaseSCfg("cc50_blend_w69", cc50_blend_w=0.69),
    PhaseSCfg("lgb_w52", lgb_w=0.52),
    PhaseSCfg("full_cb_w22", full_cb_w=0.22),
    PhaseSCfg("si_a33", si_alpha=0.33),
    PhaseSCfg("ic50_trans_w05", ic50_trans_w=0.05),
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

    ref_cfg = PhaseSCfg("ref_w70")
    ref_scores: dict[int, float] = {}
    for seed in seeds:
        clear_fold_cache()
        oof, _ = fit_phase_s(X, Xt, y, ext, blocks, seed, ref_cfg)
        ref_scores[seed] = competition_score(y, oof)
    print("REF w70: " + ", ".join(f"s{s}={ref_scores[s]:.2f}" for s in seeds))

    rows: list[dict] = []
    for cfg in CANDIDATES:
        print(f"  {cfg.name}...", flush=True, end=" ")
        for seed in seeds:
            clear_fold_cache()
            oof, _ = fit_phase_s(X, Xt, y, ext, blocks, seed, cfg)
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
    sm.to_csv("phase_s.csv", index=False)
    df.to_csv("phase_s.detail.csv", index=False)

    print(f"\n=== Phase S ({time.perf_counter()-t0:.0f}s) ===")
    print(sm.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print("\nPublic ref: cc50_blend_w70 = 268.60")
    print("  python make_submission_phase_s.py cc50_clust_fe")


if __name__ == "__main__":
    main()
