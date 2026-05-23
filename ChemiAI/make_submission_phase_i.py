"""
Phase I сабмиты — Kaggle-QSAR идеи поверх Phase H.

Варианты: full_cb25, pic50_ext, ext_multiseed, pic50_fr, vsa_w20,
          pic50_ext_full20, full25_pic50

Запуск:
  .venv/bin/python make_submission_phase_i.py full_cb25
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from run_local_signal_search import DATA_DIR, SUBMISSION_COLS, clear_fold_cache, fit_oof, load_data
from run_phase_b_si_ic50 import frozen_cc50, load_ext_cols
from run_phase_e_structural import blend_extra_head, feature_blocks
from run_phase_i_kaggle_ideas import (
    CAT_SEEDS,
    CC50_W,
    FR_W,
    IC50_W,
    MORD_W,
    MORGAN_W,
    blend_ic50_full,
    blend_ic50_pic50_fr,
    fit_phase_h,
    vsa_block,
)

PRESETS = {
    "phase_h_ref": {},
    "full_cb20": {"full_w": 0.20},
    "full_cb25": {"full_w": 0.25},
    "pic50_ext": {"ic50_pic50": True},
    "ext_multiseed": {"ic50_cat_seeds": CAT_SEEDS},
    "pic50_fr": {"pic50_fr": True},
    "vsa_w20": {"vsa_w": 0.20},
    "pic50_ext_full20": {"ic50_pic50": True, "full_w": 0.20},
    "full25_pic50": {"full_w": 0.25, "full_pic50": True},
}


def build_test(
    variant: str,
    seed: int = 42,
) -> np.ndarray:
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)
    cfg = PRESETS[variant]

    clear_fold_cache()

    if cfg.get("pic50_fr"):
        base = frozen_cc50(ext, ic50_cat_w=IC50_W, cc50_cat_w=CC50_W)
        oof, test = fit_oof(X, Xt, y, base, seed)
        oof, test = blend_ic50_pic50_fr(X, Xt, y, oof, test, blocks["fr_only"], FR_W, seed)
        oof, test = blend_extra_head(X, Xt, y, oof, test, blocks["mordred"], 0, MORD_W, seed)
        oof, test = blend_extra_head(X, Xt, y, oof, test, blocks["morgan"], 0, MORGAN_W, seed)
        return test

    oof, test = fit_phase_h(
        X, Xt, y, ext, blocks, seed,
        ic50_pic50=cfg.get("ic50_pic50", False),
        ic50_cat_seeds=cfg.get("ic50_cat_seeds"),
    )

    if "full_w" in cfg:
        oof, test = blend_ic50_full(
            X, Xt, y, oof, test, cfg["full_w"], seed,
            pic50=cfg.get("full_pic50", False),
        )

    if "vsa_w" in cfg:
        vsa_cols = vsa_block(cols)
        oof, test = blend_extra_head(X, Xt, y, oof, test, vsa_cols, 0, cfg["vsa_w"], seed)

    return test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=list(PRESETS))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    test = build_test(args.variant, args.seed)
    out = f"submission_phase_i_{args.variant}.csv"
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv").copy()
    sub[SUBMISSION_COLS] = np.clip(test, 0, None)
    sub.to_csv(out, index=False)
    print(f"Saved: {out}")
    print("Means:", sub[SUBMISSION_COLS].mean().round(2).to_dict())


if __name__ == "__main__":
    main()
