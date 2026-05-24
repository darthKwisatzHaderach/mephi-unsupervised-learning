"""
Phase M сабмиты — tune full_cb_w при ratio_lgb55.

Запуск:
  python make_submission_phase_m.py fcb22_lgb55
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
from run_phase_j import blend_ic50_lgb

LGB_W = 0.55

PRESETS = {
    "fcb18_lgb55": 0.18,
    "fcb20_lgb55": 0.20,
    "fcb22_lgb55": 0.22,
    "fcb25_lgb55": 0.25,
    "fcb28_lgb55": 0.28,
    "fcb30_lgb55": 0.30,
}


def build_test(full_cb_w: float, seed: int = 42) -> np.ndarray:
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)
    clear_fold_cache()

    Xh = engineer_features(X, ratios=True)
    Xth = engineer_features(Xt, ratios=True)

    oof, test = fit_phase_h(X, Xt, y, ext, blocks, seed)
    oof, test = blend_ic50_full(X, Xt, y, oof, test, full_cb_w, seed)
    _, test = blend_ic50_lgb(Xh, Xth, y, oof, test, LGB_W, seed)
    return test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=list(PRESETS))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    test = build_test(PRESETS[args.variant], args.seed)
    out = f"submission_phase_m_{args.variant}.csv"
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv").copy()
    sub[SUBMISSION_COLS] = np.clip(test, 0, None)
    sub.to_csv(out, index=False)
    print(f"Saved: {out}")
    print("Means:", sub[SUBMISSION_COLS].mean().round(2).to_dict())
    print(f"full_cb_w={PRESETS[args.variant]}, lgb_w={LGB_W}")


if __name__ == "__main__":
    main()
