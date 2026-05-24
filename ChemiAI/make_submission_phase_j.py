"""
Phase J сабмиты — vsa / LGBM поверх Phase I (full_cb25).

Варианты: i_vsa_w15, i_vsa_w20, h_vsa_w15, lgb_ic15, lgb_ic20, lgb_ic25

Запуск:
  python make_submission_phase_j.py i_vsa_w15
  python make_submission_phase_j.py lgb_ic20
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from run_local_signal_search import DATA_DIR, SUBMISSION_COLS, clear_fold_cache, load_data
from run_phase_b_si_ic50 import load_ext_cols
from run_phase_e_structural import blend_extra_head, feature_blocks
from run_phase_i_kaggle_ideas import fit_phase_h, vsa_block
from run_phase_j import blend_ic50_lgb, fit_phase_i

PRESETS = {
    "i_vsa_w15": {"vsa_w": 0.15, "on": "phase_i"},
    "i_vsa_w20": {"vsa_w": 0.20, "on": "phase_i"},
    "h_vsa_w15": {"vsa_w": 0.15, "on": "phase_h"},
    "lgb_ic10": {"lgb_w": 0.10},
    "lgb_ic12": {"lgb_w": 0.12},
    "lgb_ic15": {"lgb_w": 0.15},
    "lgb_ic18": {"lgb_w": 0.18},
    "lgb_ic20": {"lgb_w": 0.20},
    "lgb_ic22": {"lgb_w": 0.22},
    "lgb_ic25": {"lgb_w": 0.25},
    "lgb_ic28": {"lgb_w": 0.28},
    "lgb_ic30": {"lgb_w": 0.30},
    "lgb_ic32": {"lgb_w": 0.32},
    "lgb_ic35": {"lgb_w": 0.35},
    "lgb_ic38": {"lgb_w": 0.38},
    "lgb_ic40": {"lgb_w": 0.40},
    "lgb_ic42": {"lgb_w": 0.42},
}


def build_test(variant: str, seed: int = 42) -> np.ndarray:
    cfg = PRESETS[variant]
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)
    clear_fold_cache()

    if "lgb_w" in cfg:
        oof, test = fit_phase_i(X, Xt, y, ext, blocks, seed)
        _, test = blend_ic50_lgb(X, Xt, y, oof, test, cfg["lgb_w"], seed)
        return test

    vsa_cols = vsa_block(cols)
    if cfg["on"] == "phase_i":
        oof, test = fit_phase_i(X, Xt, y, ext, blocks, seed)
    else:
        oof, test = fit_phase_h(X, Xt, y, ext, blocks, seed)

    _, test = blend_extra_head(X, Xt, y, oof, test, vsa_cols, 0, cfg["vsa_w"], seed)
    return test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=list(PRESETS))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    test = build_test(args.variant, args.seed)
    out = f"submission_phase_j_{args.variant}.csv"
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv").copy()
    sub[SUBMISSION_COLS] = np.clip(test, 0, None)
    sub.to_csv(out, index=False)
    print(f"Saved: {out}")
    print("Means:", sub[SUBMISSION_COLS].mean().round(2).to_dict())


if __name__ == "__main__":
    main()
