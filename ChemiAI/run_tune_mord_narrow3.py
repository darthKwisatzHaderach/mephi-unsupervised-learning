"""Узкая сетка mordred_w ∈ {0.48, 0.50, 0.52, 0.55} vs ref w48 (public 273.11)."""
import time

import pandas as pd

from run_local_signal_search import clear_fold_cache, competition_score, ensure_data, fit_oof, load_data, per_target_rmse
from run_phase_b_si_ic50 import SEEDS, frozen_cc50, load_ext_cols
from run_phase_e_structural import blend_extra_head, feature_blocks

IC50_W = 0.65
CC50_W = 0.25
FR_W = 0.42
REF_MORD_W = 0.48


def fit_pipeline(X, Xt, y, ext, blocks, mord_w, seed):
    clear_fold_cache()
    base = frozen_cc50(ext, ic50_cat_w=IC50_W, cc50_cat_w=CC50_W)
    oof, test = fit_oof(X, Xt, y, base, random_state=seed)
    oof, test = blend_extra_head(
        X, Xt, y, oof, test, blocks["fr_only"], 0, FR_W, seed,
    )
    oof, test = blend_extra_head(
        X, Xt, y, oof, test, blocks["mordred"], 0, mord_w, seed,
    )
    return oof, test


def main() -> None:
    ensure_data()
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)

    ref_scores = {}
    for seed in SEEDS:
        oof, _ = fit_pipeline(X, Xt, y, ext, blocks, REF_MORD_W, seed)
        ref_scores[seed] = competition_score(y, oof)

    print("ref mord_w48:", {s: round(ref_scores[s], 2) for s in SEEDS})
    t0 = time.perf_counter()
    rows = []

    for mw in [0.48, 0.50, 0.52, 0.55]:
        for seed in SEEDS:
            oof, _ = fit_pipeline(X, Xt, y, ext, blocks, mw, seed)
            rmse = per_target_rmse(y, oof)
            sc = competition_score(y, oof)
            rows.append({
                "name": f"mord_w{int(mw*100)}",
                "mord_w": mw,
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
    print("\n=== mordred w48+ grid ===")
    print(sm.to_string(index=False))
    sm.to_csv("phase_g_mord_tune3.csv", index=False)
    print(f"\nВремя: {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
