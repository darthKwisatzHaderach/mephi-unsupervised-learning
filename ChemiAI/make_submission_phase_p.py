"""
Phase P сабмиты — PDF feature engineering.

Варианты: fe_v2_lgb55, fe_mord_head_lgb55, ro5_head_lgb55, cc50_fe_lgb55, combo_mord_ro5

Запуск:
  python make_submission_phase_p.py fe_v2_lgb55
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from run_local_signal_search import DATA_DIR, SUBMISSION_COLS, clear_fold_cache, load_data
from run_phase_p import CANDIDATES, fit_phase_p
from run_phase_b_si_ic50 import load_ext_cols
from run_phase_e_structural import feature_blocks

PRESETS = {c.name: c for c in CANDIDATES if c.name != "ref_lgb55"}
PRESETS["ref_lgb55"] = next(c for c in CANDIDATES if c.name == "ref_lgb55")


def build_test(variant: str, seed: int = 42) -> np.ndarray:
    cfg = PRESETS[variant]
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)
    clear_fold_cache()
    _, test = fit_phase_p(X, Xt, y, ext, blocks, seed, cfg)
    return test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=list(PRESETS))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    test = build_test(args.variant, args.seed)
    out = f"submission_phase_p_{args.variant}.csv"
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv").copy()
    sub[SUBMISSION_COLS] = np.clip(test, 0, None)
    sub.to_csv(out, index=False)
    print(f"Saved: {out}")
    print("Means:", sub[SUBMISSION_COLS].mean().round(2).to_dict())


if __name__ == "__main__":
    main()
