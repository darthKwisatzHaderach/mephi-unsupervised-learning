"""
Phase L2 сабмиты — interactions (2.2) и LGB morgan-head (2.1).

Запуск:
  python make_submission_phase_l2.py inter_lgb48
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from run_local_signal_search import DATA_DIR, SUBMISSION_COLS, clear_fold_cache, load_data
from run_phase_b_si_ic50 import load_ext_cols
from run_phase_e_structural import feature_blocks
from run_phase_l2 import FeCfg, fit_ratio_lgb_pipeline

PRESETS = {
    "inter_lgb45": {"fe": FeCfg(interactions=True), "lgb_w": 0.45, "lgb_morgan": False},
    "inter_lgb48": {"fe": FeCfg(interactions=True), "lgb_w": 0.48, "lgb_morgan": False},
    "inter_lgb50": {"fe": FeCfg(interactions=True), "lgb_w": 0.50, "lgb_morgan": False},
    "lgb_morgan_lgb45": {"fe": FeCfg(), "lgb_w": 0.45, "lgb_morgan": True},
    "inter_lgb_morgan45": {
        "fe": FeCfg(interactions=True),
        "lgb_w": 0.45,
        "lgb_morgan": True,
    },
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=list(PRESETS))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = PRESETS[args.variant]
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)
    clear_fold_cache()

    _, test = fit_ratio_lgb_pipeline(
        X, Xt, y, ext, blocks, args.seed,
        cfg["fe"],
        lgb_w=cfg["lgb_w"],
        lgb_morgan=cfg["lgb_morgan"],
    )

    out = f"submission_phase_l2_{args.variant}.csv"
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv").copy()
    sub[SUBMISSION_COLS] = np.clip(test, 0, None)
    sub.to_csv(out, index=False)
    print(f"Saved: {out}")
    print("Means:", sub[SUBMISSION_COLS].mean().round(2).to_dict())


if __name__ == "__main__":
    main()
