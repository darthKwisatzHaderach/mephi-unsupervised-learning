"""
Phase Q — CC50 transductive fe (продолжение cc50_fe_lgb55, public 268.82).

Кандидаты:
  ref_cc50_fe        — baseline (4 ratio в transductive, k=3, blend_w=0.60)
  cc50_fe_v2         — 5 PDF ratio только в transductive
  cc50_fe_v1v2       — 9 ratio в transductive
  cc50_fe_only       — transductive только на 4 fe (без 192 base)
  cc50_fe_log1p      — ratio + log1p в transductive
  cc50_trans_k2/4/5/7
  cc50_blend_w50/55/65/70

Запуск:
  python run_phase_q.py --quick
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import pandas as pd

from phase_k_fe import engineer_features, fe_column_names
from run_local_signal_search import clear_fold_cache, competition_score, ensure_data, load_data, per_target_rmse
from run_phase_b_si_ic50 import SEEDS, load_ext_cols
from run_phase_e_structural import feature_blocks
from run_phase_i_kaggle_ideas import blend_ic50_full, fit_phase_h
from run_phase_j import FULL_CB_W, blend_ic50_lgb

LGB_W = 0.55


@dataclass
class PhaseQCfg:
    name: str
    trans_ratios: bool = True
    trans_ratios_v2: bool = False
    trans_log1p: bool = False
    trans_fe_only: bool = False
    cc50_trans_k: int | None = None
    cc50_blend_w: float | None = None
    lgb_w: float = LGB_W


def _build_trans(
    X: pd.DataFrame,
    *,
    trans_ratios: bool,
    trans_ratios_v2: bool,
    trans_log1p: bool,
    trans_fe_only: bool,
) -> pd.DataFrame:
    Xfe = engineer_features(
        X,
        ratios=trans_ratios,
        ratios_v2=trans_ratios_v2,
        log1p_cols=trans_log1p,
    )
    if not trans_fe_only:
        return Xfe
    fe_cols = fe_column_names(
        ratios=trans_ratios,
        ratios_v2=trans_ratios_v2,
        log1p_cols=trans_log1p,
    )
    return Xfe[fe_cols]


def fit_phase_q(
    X: pd.DataFrame,
    Xt: pd.DataFrame,
    y,
    ext,
    blocks,
    seed: int,
    cfg: PhaseQCfg,
) -> tuple:
    Xh = engineer_features(X, ratios=True)
    Xth = engineer_features(Xt, ratios=True)
    X_trans = _build_trans(
        X,
        trans_ratios=cfg.trans_ratios,
        trans_ratios_v2=cfg.trans_ratios_v2,
        trans_log1p=cfg.trans_log1p,
        trans_fe_only=cfg.trans_fe_only,
    )
    Xt_trans = _build_trans(
        Xt,
        trans_ratios=cfg.trans_ratios,
        trans_ratios_v2=cfg.trans_ratios_v2,
        trans_log1p=cfg.trans_log1p,
        trans_fe_only=cfg.trans_fe_only,
    )

    oof, test = fit_phase_h(
        X, Xt, y, ext, blocks, seed,
        X_trans=X_trans, Xt_trans=Xt_trans,
        cc50_trans_k=cfg.cc50_trans_k,
        cc50_blend_w=cfg.cc50_blend_w,
    )
    oof, test = blend_ic50_full(X, Xt, y, oof, test, FULL_CB_W, seed)
    oof, test = blend_ic50_lgb(Xh, Xth, y, oof, test, cfg.lgb_w, seed)
    return oof, test


CANDIDATES: list[PhaseQCfg] = [
    PhaseQCfg("ref_cc50_fe"),
    PhaseQCfg("cc50_fe_v2", trans_ratios=False, trans_ratios_v2=True),
    PhaseQCfg("cc50_fe_v1v2", trans_ratios=True, trans_ratios_v2=True),
    PhaseQCfg("cc50_fe_only", trans_fe_only=True),
    PhaseQCfg("cc50_fe_log1p", trans_log1p=True),
    PhaseQCfg("cc50_trans_k2", cc50_trans_k=2),
    PhaseQCfg("cc50_trans_k4", cc50_trans_k=4),
    PhaseQCfg("cc50_trans_k5", cc50_trans_k=5),
    PhaseQCfg("cc50_trans_k7", cc50_trans_k=7),
    PhaseQCfg("cc50_blend_w50", cc50_blend_w=0.50),
    PhaseQCfg("cc50_blend_w55", cc50_blend_w=0.55),
    PhaseQCfg("cc50_blend_w65", cc50_blend_w=0.65),
    PhaseQCfg("cc50_blend_w70", cc50_blend_w=0.70),
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

    ref_cfg = PhaseQCfg("ref_cc50_fe")
    ref_scores: dict[int, float] = {}
    for seed in seeds:
        clear_fold_cache()
        oof, _ = fit_phase_q(X, Xt, y, ext, blocks, seed, ref_cfg)
        ref_scores[seed] = competition_score(y, oof)
    print("REF cc50_fe: " + ", ".join(f"s{s}={ref_scores[s]:.2f}" for s in seeds))

    rows: list[dict] = []
    for cfg in CANDIDATES:
        print(f"  {cfg.name}...", flush=True, end=" ")
        for seed in seeds:
            clear_fold_cache()
            oof, _ = fit_phase_q(X, Xt, y, ext, blocks, seed, cfg)
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
    sm.to_csv("phase_q.csv", index=False)
    df.to_csv("phase_q.detail.csv", index=False)

    print(f"\n=== Phase Q ({time.perf_counter()-t0:.0f}s) ===")
    print(sm.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print("\nPublic ref: cc50_fe_lgb55 = 268.82")
    print("  python make_submission_phase_q.py cc50_fe_v1v2")


if __name__ == "__main__":
    main()
