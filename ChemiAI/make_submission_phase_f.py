"""Сабмиты Phase F — SI-эксперименты поверх fr42_ic65."""
import argparse

import numpy as np
import pandas as pd

from run_local_signal_search import DATA_DIR, SUBMISSION_COLS, clear_fold_cache, fit_oof, load_data
from run_phase_b_si_ic50 import frozen_cc50, load_ext_cols
from run_phase_e_structural import blend_extra_head, feature_blocks
from run_phase_f_si_search import (
    BEST_CC50_W,
    BEST_FR_W,
    BEST_IC50_W,
    blend_si_alt_robust_head,
    blend_si_logratio_head,
    blend_si_transductive_head,
    best_base_cfg,
)


def build_base_fr(seed: int = 42):
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)
    clear_fold_cache()
    oof, test = fit_oof(X, Xt, y, best_base_cfg(ext), random_state=seed)
    oof, test = blend_extra_head(
        X, Xt, y, oof, test, blocks["fr_only"], 0, BEST_FR_W, seed,
    )
    return X, Xt, y, blocks, oof, test


def apply_si_head(variant: str, seed: int = 42) -> np.ndarray:
    X, Xt, y, blocks, oof, test = build_base_fr(seed)

    if variant.startswith("si_fr_only_w"):
        w = int(variant.split("_w")[1]) / 100
        _, pred = blend_extra_head(X, Xt, y, oof, test, blocks["fr_only"], 2, w, seed)
    elif variant.startswith("si_mordred_w"):
        w = int(variant.split("_w")[1]) / 100
        _, pred = blend_extra_head(X, Xt, y, oof, test, blocks["mordred"], 2, w, seed)
    elif variant.startswith("si_morgan_w"):
        w = int(variant.split("_w")[1]) / 100
        _, pred = blend_extra_head(X, Xt, y, oof, test, blocks["morgan"], 2, w, seed)
    elif variant.startswith("si_trans_k"):
        k = int(variant.split("_k")[1].split("_")[0])
        w = int(variant.split("_w")[1]) / 100
        _, pred = blend_si_transductive_head(X, Xt, y, oof, test, k, w, seed)
    elif variant.startswith("si_ratio_"):
        rest = variant[len("si_ratio_"):]
        block, wpart = rest.rsplit("_w", 1)
        w = int(wpart) / 100
        _, pred = blend_si_logratio_head(X, Xt, y, oof, test, blocks[block], w, seed)
    elif variant.startswith("si_huber_w") or variant.startswith("si_quantile_w"):
        kind = "huber" if "huber" in variant else "quantile"
        w = int(variant.split("_w")[1]) / 100
        _, pred = blend_si_alt_robust_head(X, Xt, y, oof, test, w, seed, kind)
    elif variant.startswith("si_a") and "_rw" not in variant:
        alpha = int(variant.split("_a")[1]) / 100
        ext = load_ext_cols(list(X.columns))
        clear_fold_cache()
        cfg = best_base_cfg(ext)
        cfg.si_alpha = alpha
        oof2, test2 = fit_oof(X, Xt, y, cfg, seed)
        _, pred = blend_extra_head(
            X, Xt, y, oof2, test2, blocks["fr_only"], 0, BEST_FR_W, seed,
        )
    elif variant.startswith("si_rw"):
        rw = int(variant.split("_rw")[1]) / 100
        ext = load_ext_cols(list(X.columns))
        clear_fold_cache()
        cfg = best_base_cfg(ext)
        cfg.si_robust_w = rw
        oof2, test2 = fit_oof(X, Xt, y, cfg, seed)
        _, pred = blend_extra_head(
            X, Xt, y, oof2, test2, blocks["fr_only"], 0, BEST_FR_W, seed,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return np.clip(pred, 0, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("variant", help="e.g. si_mordred_w25, si_trans_k3_w20")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    pred = apply_si_head(args.variant, args.seed)
    out = args.out or f"submission_phase_f_{args.variant}.csv"
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv").copy()
    sub[SUBMISSION_COLS] = pred
    sub.to_csv(out, index=False)
    print(f"Saved {out}")
    print(f"  base: ic50={BEST_IC50_W} cc50={BEST_CC50_W} fr={BEST_FR_W}")
    print(f"  si_variant={args.variant}")
    print(sub[SUBMISSION_COLS].describe().round(2))


if __name__ == "__main__":
    main()
