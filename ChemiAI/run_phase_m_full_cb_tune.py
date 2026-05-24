"""
Phase M — микро-tune full_cb_w при fixed ratio_lgb55 (public 269.09).

Запуск:
  python run_phase_m_full_cb_tune.py
  python run_phase_m_full_cb_tune.py --quick
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from phase_k_fe import engineer_features
from run_local_signal_search import clear_fold_cache, competition_score, ensure_data, load_data, per_target_rmse
from run_phase_b_si_ic50 import SEEDS, load_ext_cols
from run_phase_e_structural import feature_blocks
from run_phase_i_kaggle_ideas import blend_ic50_full, fit_phase_h
from run_phase_j import blend_ic50_lgb

LGB_W = 0.55
FULL_CB_WEIGHTS = [0.18, 0.20, 0.22, 0.25, 0.28, 0.30]
REF_FULL_CB_W = 0.25


def fit_ratio_lgb(
    X,
    Xt,
    y,
    ext,
    blocks,
    seed: int,
    *,
    full_cb_w: float,
    lgb_w: float = LGB_W,
):
    Xh = engineer_features(X, ratios=True)
    Xth = engineer_features(Xt, ratios=True)
    oof, test = fit_phase_h(X, Xt, y, ext, blocks, seed)
    oof, test = blend_ic50_full(X, Xt, y, oof, test, full_cb_w, seed)
    oof, test = blend_ic50_lgb(Xh, Xth, y, oof, test, lgb_w, seed)
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
        oof, _ = fit_ratio_lgb(
            X, Xt, y, ext, blocks, seed, full_cb_w=REF_FULL_CB_W,
        )
        ref_scores[seed] = competition_score(y, oof)
    print(
        f"REF full_cb{int(REF_FULL_CB_W*100)} + lgb{LGB_W}: "
        + ", ".join(f"s{s}={ref_scores[s]:.2f}" for s in seeds)
    )

    rows: list[dict] = []
    for full_w in FULL_CB_WEIGHTS:
        name = f"fcb{int(round(full_w * 100))}_lgb55"
        print(f"  {name}...", flush=True, end=" ")
        for seed in seeds:
            clear_fold_cache()
            oof, _ = fit_ratio_lgb(
                X, Xt, y, ext, blocks, seed, full_cb_w=full_w,
            )
            sc = competition_score(y, oof)
            rmse = per_target_rmse(y, oof)
            d = sc - ref_scores[seed]
            rows.append({
                "name": name,
                "full_cb_w": full_w,
                "seed": seed,
                "oof": sc,
                "delta": d,
                "ic50": rmse[0],
            })
            print(f"s{seed}={sc:.2f}({d:+.2f})", end=" ")
        print()

    df = pd.DataFrame(rows)
    sm = (
        df.groupby(["name", "full_cb_w"])
        .agg(oof_mean=("oof", "mean"), delta_mean=("delta", "mean"), ic50=("ic50", "mean"))
        .reset_index()
        .sort_values("delta_mean")
    )
    sm.to_csv("phase_m_full_cb_tune.csv", index=False)
    df.to_csv("phase_m_full_cb_tune.detail.csv", index=False)

    print(f"\n=== Phase M ({time.perf_counter()-t0:.0f}s) ===")
    print(sm.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print("\nPublic ref: ratio_lgb55 fcb25 = 269.09")
    print("  python make_submission_phase_m.py fcb22_lgb55")


if __name__ == "__main__":
    main()
