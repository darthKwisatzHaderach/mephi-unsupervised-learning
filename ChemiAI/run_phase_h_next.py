"""
Phase H: следующие варианты поверх mordred_w55 (public 272.99).
1) morgan IC50 head
2) fr_w / ic65 retune (mord55 fixed)
3) seed ensemble submission
"""
import time

import numpy as np
import pandas as pd

from run_local_signal_search import (
    DATA_DIR,
    SUBMISSION_COLS,
    clear_fold_cache,
    competition_score,
    ensure_data,
    fit_oof,
    load_data,
    per_target_rmse,
)
from run_phase_b_si_ic50 import SEEDS, frozen_cc50, load_ext_cols
from run_phase_e_structural import blend_extra_head, feature_blocks

MORD_W = 0.55
FR_W = 0.42
IC50_W = 0.65
CC50_W = 0.25


def fit_mord55(
    X, Xt, y, ext, blocks, seed, ic50_w=IC50_W, fr_w=FR_W, mord_w=MORD_W,
):
    clear_fold_cache()
    base = frozen_cc50(ext, ic50_cat_w=ic50_w, cc50_cat_w=CC50_W)
    oof, test = fit_oof(X, Xt, y, base, random_state=seed)
    oof, test = blend_extra_head(X, Xt, y, oof, test, blocks["fr_only"], 0, fr_w, seed)
    oof, test = blend_extra_head(X, Xt, y, oof, test, blocks["mordred"], 0, mord_w, seed)
    return oof, test


def eval_rows(name, fn, X, Xt, y, ext, blocks, ref_scores):
    rows = []
    for seed in SEEDS:
        oof, _ = fn(X, Xt, y, ext, blocks, seed)
        sc = competition_score(y, oof)
        rmse = per_target_rmse(y, oof)
        rows.append({
            "name": name,
            "seed": seed,
            "oof": sc,
            "delta": sc - ref_scores[seed],
            "ic50": rmse[0],
            "cc50": rmse[1],
            "si": rmse[2],
        })
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("name").agg(
        oof_mean=("oof", "mean"),
        delta_mean=("delta", "mean"),
        wins=("delta", lambda s: int((s < -0.05).sum())),
        ic50=("ic50", "mean"),
        cc50=("cc50", "mean"),
        si=("si", "mean"),
    ).reset_index().sort_values("delta_mean")


def main() -> None:
    ensure_data()
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)
    t0 = time.perf_counter()

    ref_scores = {}
    ref_tests = {}
    for seed in SEEDS:
        oof, test = fit_mord55(X, Xt, y, ext, blocks, seed)
        ref_scores[seed] = competition_score(y, oof)
        ref_tests[seed] = test
    print("ref mord55:", {s: round(ref_scores[s], 2) for s in SEEDS})

    rows = []

    # H1: morgan IC50 head
    for w in [0.15, 0.20, 0.25, 0.30]:
        def fn(X, Xt, y, ext, blocks, seed, w=w):
            oof, test = fit_mord55(X, Xt, y, ext, blocks, seed)
            return blend_extra_head(X, Xt, y, oof, test, blocks["morgan"], 0, w, seed)

        rows.extend(eval_rows(f"morgan_w{int(w*100)}", fn, X, Xt, y, ext, blocks, ref_scores))

    # H2: fr_w retune
    for fw in [0.38, 0.40, 0.44, 0.45]:
        def fn(X, Xt, y, ext, blocks, seed, fw=fw):
            return fit_mord55(X, Xt, y, ext, blocks, seed, fr_w=fw)

        rows.extend(eval_rows(f"fr_w{int(fw*100)}", fn, X, Xt, y, ext, blocks, ref_scores))

    # H3: ic65 retune
    for iw in [0.62, 0.68, 0.70]:
        def fn(X, Xt, y, ext, blocks, seed, iw=iw):
            return fit_mord55(X, Xt, y, ext, blocks, seed, ic50_w=iw)

        rows.extend(eval_rows(f"ic65_{int(iw*100)}", fn, X, Xt, y, ext, blocks, ref_scores))

    df = pd.DataFrame(rows)
    sm = summarize(df)
    sm.to_csv("phase_h_next.csv", index=False)
    df.to_csv("phase_h_next.detail.csv", index=False)
    print("\n=== TOP 15 Phase H ===")
    print(sm.head(15).to_string(index=False))

    # H4: seed ensemble
    ens_seeds = [42, 2024, 7, 0, 1, 13, 77, 123, 456, 999]
    tests = []
    for seed in ens_seeds:
        clear_fold_cache()
        _, test = fit_mord55(X, Xt, y, ext, blocks, seed)
        tests.append(test)
    ens = np.mean(tests, axis=0)
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv").copy()
    sub[SUBMISSION_COLS] = np.clip(ens, 0, None)
    sub.to_csv("submission_phase_h_seed_ensemble.csv", index=False)
    print(
        f"\nSeed ensemble saved: mean SI={ens[:,2].mean():.2f}, "
        f"seed42 SI={ref_tests[42][:,2].mean():.2f}"
    )
    print(f"Total time: {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
