"""Узкая сетка morgan IC50 w ∈ {0.25, 0.28, 0.30, 0.32, 0.35} vs ref w25 (public 272.44)."""
import time

import pandas as pd

from run_local_signal_search import clear_fold_cache, competition_score, ensure_data, fit_oof, load_data, per_target_rmse
from run_phase_b_si_ic50 import SEEDS, frozen_cc50, load_ext_cols
from run_phase_e_structural import blend_extra_head, feature_blocks

IC50_W = 0.65
CC50_W = 0.25
FR_W = 0.42
MORD_W = 0.55
REF_MORGAN_W = 0.25


def fit_pipeline(X, Xt, y, ext, blocks, morgan_w, seed):
    clear_fold_cache()
    base = frozen_cc50(ext, ic50_cat_w=IC50_W, cc50_cat_w=CC50_W)
    oof, test = fit_oof(X, Xt, y, base, random_state=seed)
    oof, test = blend_extra_head(X, Xt, y, oof, test, blocks["fr_only"], 0, FR_W, seed)
    oof, test = blend_extra_head(X, Xt, y, oof, test, blocks["mordred"], 0, MORD_W, seed)
    oof, test = blend_extra_head(X, Xt, y, oof, test, blocks["morgan"], 0, morgan_w, seed)
    return oof, test


def main() -> None:
    ensure_data()
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)

    ref_scores = {}
    for seed in SEEDS:
        oof, _ = fit_pipeline(X, Xt, y, ext, blocks, REF_MORGAN_W, seed)
        ref_scores[seed] = competition_score(y, oof)

    print("ref morgan_w25:", {s: round(ref_scores[s], 2) for s in SEEDS})
    t0 = time.perf_counter()
    rows = []

    for mw in [0.25, 0.28, 0.30, 0.32, 0.35]:
        for seed in SEEDS:
            oof, _ = fit_pipeline(X, Xt, y, ext, blocks, mw, seed)
            rmse = per_target_rmse(y, oof)
            sc = competition_score(y, oof)
            rows.append({
                "name": f"morgan_w{int(round(mw*100))}",
                "morgan_w": mw,
                "seed": seed,
                "oof": sc,
                "delta": sc - ref_scores[seed],
                "ic50": rmse[0],
            })

    sm = (
        pd.DataFrame(rows)
        .groupby("name")
        .agg(
            oof_mean=("oof", "mean"),
            delta_mean=("delta", "mean"),
            wins=("delta", lambda s: int((s < -0.05).sum())),
            ic50=("ic50", "mean"),
        )
        .reset_index()
        .sort_values("delta_mean")
    )
    print("\n=== morgan weight grid ===")
    print(sm.to_string(index=False))
    sm.to_csv("phase_h_morgan_tune.csv", index=False)
    print(f"\nВремя: {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
