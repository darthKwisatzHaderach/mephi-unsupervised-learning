"""
Сабмит: best pipeline + CC50 transductive k=3 + CatBoost blend 15%.
Локально: mean OOF -3.31 vs baseline на seeds 42/2024/7.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from run_local_signal_search import (
    DATA_DIR,
    PipelineConfig,
    SUBMISSION_COLS,
    TARGET_COLS,
    enforce_si_invariant,
    fit_oof,
    load_data,
)

OUTPUT = Path("submission_cc50_k3_cat15.csv")


def main() -> None:
    X_train, X_test, y_train, _ = load_data()
    cfg = PipelineConfig(cc50_trans_k=3, cc50_blend_w=0.60, cc50_cat_w=0.15)
    _, test_pred = fit_oof(X_train, X_test, y_train, cfg, random_state=42)

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sub = sample.copy()
    sub[SUBMISSION_COLS] = test_pred
    sub[SUBMISSION_COLS] = sub[SUBMISSION_COLS].clip(lower=0)
    sub.to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT}")
    print(sub[SUBMISSION_COLS].describe().round(2))
    assert sub["index"].tolist() == test["index"].tolist()


if __name__ == "__main__":
    main()
