"""Сабмит: ext_k3_cat20 — CC50 CatBoost blend 20% (OOF -0.26 vs cat15)."""
from pathlib import Path

import pandas as pd

from run_local_signal_search import (
    DATA_DIR,
    EXTENDED_IC50_FEATURES,
    SUBMISSION_COLS,
    PipelineConfig,
    fit_oof,
    load_data,
)

OUTPUT = Path("submission_ext_k3_cat20.csv")


def main() -> None:
    X_train, X_test, y_train, all_cols = load_data()
    ext_cols = [c for c in EXTENDED_IC50_FEATURES if c in all_cols]
    cfg = PipelineConfig(
        ic50_cat_cols=ext_cols,
        cc50_trans_k=3,
        cc50_blend_w=0.60,
        cc50_cat_w=0.20,
    )
    _, test_pred = fit_oof(X_train, X_test, y_train, cfg, random_state=42)

    sub = pd.read_csv(DATA_DIR / "sample_submission.csv").copy()
    sub[SUBMISSION_COLS] = test_pred
    sub[SUBMISSION_COLS] = sub[SUBMISSION_COLS].clip(lower=0)
    sub.to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT}")
    print(sub[SUBMISSION_COLS].describe().round(2))


if __name__ == "__main__":
    main()
