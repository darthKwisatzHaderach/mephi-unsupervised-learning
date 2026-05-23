"""Сабмиты фазы B: CC50 frozen ext_k3_cat15, меняем IC50/SI."""
import argparse
from pathlib import Path

import pandas as pd

from run_local_signal_search import (
    DATA_DIR,
    SUBMISSION_COLS,
    fit_oof,
    load_data,
)
from run_phase_b_si_ic50 import frozen_cc50, load_ext_cols


def build_submission(cfg, out: Path, seed: int = 42) -> None:
    X_train, X_test, y_train, all_cols = load_data()
    ext_cols = load_ext_cols(all_cols)
    _, test_pred = fit_oof(X_train, X_test, y_train, cfg, random_state=seed)
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv").copy()
    sub[SUBMISSION_COLS] = test_pred.clip(min=0)
    sub.to_csv(out, index=False)
    print(f"Saved {out}")
    print(sub[SUBMISSION_COLS].describe().round(2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "variant",
        choices=[
            "ic50_w35", "ic50_w37", "ic50_w38", "ic50_w40",
            "ic50_w45", "ic50_w50", "ic50_w55", "ic50_w60",
            "w50_cc25", "w55_cc25",
            "si_cb_w35_k90", "si_cb_w45_k120",
            "si_a45", "combo_w35_a45", "combo_w35_a42",
        ],
    )
    args = parser.parse_args()
    _, _, _, all_cols = load_data()
    ext = load_ext_cols(all_cols)

    variants = {
        "ic50_w35": (frozen_cc50(ext, ic50_cat_w=0.35), "submission_phase_b_ic50_w35.csv"),
        "ic50_w37": (frozen_cc50(ext, ic50_cat_w=0.37), "submission_phase_b_ic50_w37.csv"),
        "ic50_w38": (frozen_cc50(ext, ic50_cat_w=0.38), "submission_phase_b_ic50_w38.csv"),
        "ic50_w40": (frozen_cc50(ext, ic50_cat_w=0.40), "submission_phase_b_ic50_w40.csv"),
        "ic50_w45": (frozen_cc50(ext, ic50_cat_w=0.45), "submission_phase_b_ic50_w45.csv"),
        "ic50_w50": (frozen_cc50(ext, ic50_cat_w=0.50), "submission_phase_b_ic50_w50.csv"),
        "ic50_w55": (frozen_cc50(ext, ic50_cat_w=0.55), "submission_phase_b_ic50_w55.csv"),
        "ic50_w60": (frozen_cc50(ext, ic50_cat_w=0.60), "submission_phase_b_ic50_w60.csv"),
        "w50_cc25": (frozen_cc50(ext, ic50_cat_w=0.50, cc50_cat_w=0.25), "submission_phase_b_w50_cc25.csv"),
        "w55_cc25": (frozen_cc50(ext, ic50_cat_w=0.55, cc50_cat_w=0.25), "submission_phase_b_w55_cc25.csv"),
        "si_cb_w35_k90": (
            frozen_cc50(ext, ic50_cat_w=0.55, cc50_cat_w=0.25, si_robust_w=0.0, si_catboost_w=0.35, si_topk=90),
            "submission_phase_b_si_cb_w35_k90.csv",
        ),
        "si_cb_w45_k120": (
            frozen_cc50(ext, ic50_cat_w=0.55, cc50_cat_w=0.25, si_robust_w=0.0, si_catboost_w=0.45, si_topk=120),
            "submission_phase_b_si_cb_w45_k120.csv",
        ),
        "si_a45": (frozen_cc50(ext, si_alpha=0.45), "submission_phase_b_si_a45.csv"),
        "combo_w35_a45": (
            frozen_cc50(ext, ic50_cat_w=0.35, si_alpha=0.45),
            "submission_phase_b_combo_w35_a45.csv",
        ),
        "combo_w35_a42": (
            frozen_cc50(ext, ic50_cat_w=0.35, si_alpha=0.42),
            "submission_phase_b_combo_w35_a42.csv",
        ),
    }
    cfg, path = variants[args.variant]
    build_submission(cfg, Path(path))


if __name__ == "__main__":
    main()
