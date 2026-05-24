"""
Phase K сабмиты — feature engineering поверх Phase J.

Варианты: ratio_lgb42, ratio_both_lgb42, ratio_bin_lgb42, ratio_all_lgb42,
          ratio_lgb38, ratio_lgb45

Запуск:
  python make_submission_phase_k.py ratio_lgb42
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from phase_k_fe import engineer_features
from run_local_signal_search import DATA_DIR, SUBMISSION_COLS, clear_fold_cache, load_data
from run_phase_b_si_ic50 import load_ext_cols
from run_phase_e_structural import feature_blocks
from run_phase_i_kaggle_ideas import blend_ic50_full, fit_phase_h
from run_phase_j import FULL_CB_W, blend_ic50_lgb

PRESETS = {
    "ratio_lgb42": {
        "ratios": True,
        "log1p_cols": False,
        "binary_rules": False,
        "augment_full": False,
        "lgb_w": 0.42,
    },
    "ratio_both_lgb42": {
        "ratios": True,
        "log1p_cols": False,
        "binary_rules": False,
        "augment_full": True,
        "lgb_w": 0.42,
    },
    "ratio_bin_lgb42": {
        "ratios": True,
        "log1p_cols": False,
        "binary_rules": True,
        "augment_full": False,
        "lgb_w": 0.42,
    },
    "ratio_all_lgb42": {
        "ratios": True,
        "log1p_cols": True,
        "binary_rules": True,
        "augment_full": False,
        "lgb_w": 0.42,
    },
    "ratio_lgb38": {
        "ratios": True,
        "log1p_cols": False,
        "binary_rules": False,
        "augment_full": False,
        "lgb_w": 0.38,
    },
    "ratio_lgb45": {
        "ratios": True,
        "log1p_cols": False,
        "binary_rules": False,
        "augment_full": False,
        "lgb_w": 0.45,
    },
    "ratio_lgb48": {
        "ratios": True,
        "log1p_cols": False,
        "binary_rules": False,
        "augment_full": False,
        "lgb_w": 0.48,
    },
    "ratio_lgb50": {
        "ratios": True,
        "log1p_cols": False,
        "binary_rules": False,
        "augment_full": False,
        "lgb_w": 0.50,
    },
    "ratio_lgb52": {
        "ratios": True,
        "log1p_cols": False,
        "binary_rules": False,
        "augment_full": False,
        "lgb_w": 0.52,
    },
    "ratio_lgb55": {
        "ratios": True,
        "log1p_cols": False,
        "binary_rules": False,
        "augment_full": False,
        "lgb_w": 0.55,
    },
    "ratio_lgb58": {
        "ratios": True,
        "log1p_cols": False,
        "binary_rules": False,
        "augment_full": False,
        "lgb_w": 0.58,
    },
}


def build_test(variant: str, seed: int = 42) -> np.ndarray:
    cfg = PRESETS[variant]
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)
    clear_fold_cache()

    Xh = engineer_features(
        X,
        ratios=cfg["ratios"],
        log1p_cols=cfg["log1p_cols"],
        binary_rules=cfg["binary_rules"],
    )
    Xth = engineer_features(
        Xt,
        ratios=cfg["ratios"],
        log1p_cols=cfg["log1p_cols"],
        binary_rules=cfg["binary_rules"],
    )

    oof, test = fit_phase_h(X, Xt, y, ext, blocks, seed)

    if cfg["augment_full"]:
        oof, test = blend_ic50_full(Xh, Xth, y, oof, test, FULL_CB_W, seed)
    else:
        oof, test = blend_ic50_full(X, Xt, y, oof, test, FULL_CB_W, seed)

    oof, test = blend_ic50_lgb(Xh, Xth, y, oof, test, cfg["lgb_w"], seed)
    return test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=list(PRESETS))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    test = build_test(args.variant, args.seed)
    out = f"submission_phase_k_{args.variant}.csv"
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv").copy()
    sub[SUBMISSION_COLS] = np.clip(test, 0, None)
    sub.to_csv(out, index=False)
    print(f"Saved: {out}")
    print("Means:", sub[SUBMISSION_COLS].mean().round(2).to_dict())
    print("Features: +", len(PRESETS[args.variant]), "preset,", end=" ")
    n_fe = 4
    cfg = PRESETS[args.variant]
    if cfg["binary_rules"]:
        n_fe += 3
    if cfg["log1p_cols"]:
        n_fe += 3
    print(f"~{192 + n_fe} cols in LGB head")


if __name__ == "__main__":
    main()
