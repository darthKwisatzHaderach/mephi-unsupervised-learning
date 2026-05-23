"""Сабмиты фазы E + дотюнинг fr best (public 275.30)."""
import argparse

import numpy as np
import pandas as pd

from run_local_signal_search import DATA_DIR, SUBMISSION_COLS, clear_fold_cache, fit_oof, load_data
from run_phase_b_si_ic50 import frozen_cc50, load_ext_cols
from run_phase_e_structural import blend_extra_head, feature_blocks


def build_submission(
    ic50_cat_w: float = 0.55,
    cc50_cat_w: float = 0.25,
    fr_w: float = 0.35,
    out_path: str = "submission.csv",
    seed: int = 42,
) -> None:
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)
    base = frozen_cc50(ext, ic50_cat_w=ic50_cat_w, cc50_cat_w=cc50_cat_w)
    clear_fold_cache()
    oof, test = fit_oof(X, Xt, y, base, random_state=seed)
    _, pred = blend_extra_head(
        X, Xt, y, oof, test, blocks["fr_only"], 0, fr_w, seed,
    )
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv").copy()
    sub[SUBMISSION_COLS] = np.clip(pred, 0, None)
    sub.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(f"  ic50_cat={ic50_cat_w} cc50_cat={cc50_cat_w} fr_w={fr_w}")
    print(sub[SUBMISSION_COLS].describe().round(2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "variant",
        choices=[
            "fr_ic50_w35", "fr_ic50_w40", "fr_ic50_w42",
            "fr42_ic65", "fr42_ic65_cc28", "fr42_ic65_cc30",
            "fr40_ic68", "fr40_ic70", "fr42_ic68", "fr42_ic70", "fr45_ic70",
            "mordred_ic50_w35", "combo_ic50_si",
        ],
    )
    args = p.parse_args()
    presets = {
        "fr_ic50_w35": (0.55, 0.25, 0.35, "submission_phase_e_fr_ic50_w35.csv"),
        "fr_ic50_w40": (0.55, 0.25, 0.40, "submission_phase_e_fr_ic50_w40.csv"),
        "fr_ic50_w42": (0.55, 0.25, 0.42, "submission_phase_e_fr_ic50_w42.csv"),
        "fr42_ic65": (0.65, 0.25, 0.42, "submission_phase_e_fr42_ic65.csv"),
        "fr42_ic65_cc28": (0.65, 0.28, 0.42, "submission_phase_e_fr42_ic65_cc28.csv"),
        "fr42_ic65_cc30": (0.65, 0.30, 0.42, "submission_phase_e_fr42_ic65_cc30.csv"),
        "fr40_ic68": (0.68, 0.25, 0.40, "submission_phase_e_fr40_ic68.csv"),
        "fr40_ic70": (0.70, 0.25, 0.40, "submission_phase_e_fr40_ic70.csv"),
        "fr42_ic68": (0.68, 0.25, 0.42, "submission_phase_e_fr42_ic68.csv"),
        "fr42_ic70": (0.70, 0.25, 0.42, "submission_phase_e_fr42_ic70.csv"),
        "fr45_ic70": (0.70, 0.25, 0.45, "submission_phase_e_fr45_ic70.csv"),
        "mordred_ic50_w35": None,
        "combo_ic50_si": None,
    }
    if args.variant in ("mordred_ic50_w35", "combo_ic50_si"):
        # legacy multi-head
        X, Xt, y, cols = load_data()
        ext = load_ext_cols(cols)
        blocks = feature_blocks(cols)
        base = frozen_cc50(ext, ic50_cat_w=0.55, cc50_cat_w=0.25)
        clear_fold_cache()
        oof, test = fit_oof(X, Xt, y, base, 42)
        pred = test.copy()
        heads = [{"block": "fr_only", "target": 0, "weight": 0.35}]
        if args.variant == "mordred_ic50_w35":
            heads = [{"block": "mordred", "target": 0, "weight": 0.35}]
            out = "submission_phase_e_mordred_ic50_w35.csv"
        else:
            heads = [
                {"block": "fr_only", "target": 0, "weight": 0.35},
                {"block": "mordred", "target": 2, "weight": 0.35},
            ]
            out = "submission_phase_e_combo_ic50_si.csv"
        for h in heads:
            _, pred = blend_extra_head(
                X, Xt, y, oof, pred, blocks[h["block"]], h["target"], h["weight"], 42,
            )
        sub = pd.read_csv(DATA_DIR / "sample_submission.csv").copy()
        sub[SUBMISSION_COLS] = np.clip(pred, 0, None)
        sub.to_csv(out, index=False)
        print(f"Saved {out}")
        return

    ic50_w, cc50_w, fr_w, out = presets[args.variant]
    build_submission(ic50_w, cc50_w, fr_w, out)


if __name__ == "__main__":
    main()
