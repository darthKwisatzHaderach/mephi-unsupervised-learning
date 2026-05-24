"""
Phase K — feature engineering поверх Phase J (public best lgb_ic42 = 269.43).

Кандидаты:
  - ref_lgb42          — baseline без fe
  - ratio_lgb42        — ratios только в LGB head
  - ratio_both_lgb42   — ratios в full_cb + LGB
  - ratio_bin_lgb42    — ratios + Ro5 binary, LGB only
  - ratio_all_lgb42    — ratios + log1p + binary, LGB only
  - ratio_lgb38/45     — перекалибровка w с ratios (LGB only)

Запуск:
  python run_phase_k.py
  python run_phase_k.py --quick
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

LGB_W = 0.42


@dataclass
class FeCfg:
    ratios: bool = False
    log1p_cols: bool = False
    binary_rules: bool = False
    augment_full: bool = False  # full_cb на fe_* или только base 192


def fit_phase_j_custom(
    X: pd.DataFrame,
    Xt: pd.DataFrame,
    y,
    ext,
    blocks,
    seed: int,
    fe: FeCfg,
    *,
    full_w: float = FULL_CB_W,
    lgb_w: float = LGB_W,
) -> tuple:
    Xh = engineer_features(
        X,
        ratios=fe.ratios,
        log1p_cols=fe.log1p_cols,
        binary_rules=fe.binary_rules,
    )
    Xth = engineer_features(
        Xt,
        ratios=fe.ratios,
        log1p_cols=fe.log1p_cols,
        binary_rules=fe.binary_rules,
    )

    oof, test = fit_phase_h(X, Xt, y, ext, blocks, seed)

    if fe.augment_full and (fe.ratios or fe.log1p_cols or fe.binary_rules):
        oof, test = blend_ic50_full(Xh, Xth, y, oof, test, full_w, seed)
    else:
        oof, test = blend_ic50_full(X, Xt, y, oof, test, full_w, seed)

    use_head = fe.ratios or fe.log1p_cols or fe.binary_rules
    if use_head:
        oof, test = blend_ic50_lgb(Xh, Xth, y, oof, test, lgb_w, seed)
    else:
        oof, test = blend_ic50_lgb(X, Xt, y, oof, test, lgb_w, seed)

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

    candidates: list[tuple[str, FeCfg, float]] = [
        ("ref_lgb42", FeCfg(), LGB_W),
        ("ratio_lgb42", FeCfg(ratios=True), LGB_W),
        ("ratio_both_lgb42", FeCfg(ratios=True, augment_full=True), LGB_W),
        ("ratio_bin_lgb42", FeCfg(ratios=True, binary_rules=True), LGB_W),
        ("ratio_all_lgb42", FeCfg(ratios=True, log1p_cols=True, binary_rules=True), LGB_W),
        ("ratio_lgb38", FeCfg(ratios=True), 0.38),
        ("ratio_lgb45", FeCfg(ratios=True), 0.45),
    ]

    ref_scores: dict[int, float] = {}
    for seed in seeds:
        clear_fold_cache()
        oof, _ = fit_phase_j_custom(X, Xt, y, ext, blocks, seed, FeCfg(), lgb_w=LGB_W)
        ref_scores[seed] = competition_score(y, oof)
    print("REF lgb42: " + ", ".join(f"s{s}={ref_scores[s]:.2f}" for s in seeds))

    rows: list[dict] = []
    for name, fe, lgb_w in candidates:
        print(f"  {name}...", flush=True, end=" ")
        for seed in seeds:
            clear_fold_cache()
            oof, _ = fit_phase_j_custom(X, Xt, y, ext, blocks, seed, fe, lgb_w=lgb_w)
            sc = competition_score(y, oof)
            rmse = per_target_rmse(y, oof)
            d = sc - ref_scores[seed]
            rows.append({
                "name": name,
                "seed": seed,
                "oof": sc,
                "delta": d,
                "ic50": rmse[0],
                "lgb_w": lgb_w,
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
            lgb_w=("lgb_w", "first"),
        )
        .reset_index()
        .sort_values("delta_mean")
    )
    sm.to_csv("phase_k.csv", index=False)
    df.to_csv("phase_k.detail.csv", index=False)

    print(f"\n=== Phase K ({time.perf_counter()-t0:.0f}s) ===")
    print(sm.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print("\nPublic ref: lgb_ic42 = 269.43")
    print("  python make_submission_phase_k.py ratio_lgb42")


if __name__ == "__main__":
    main()
