"""
Узкая сетка вокруг public best fr42_ic65 (274.76).
ic50_cat_w ∈ {0.62, 0.65, 0.68, 0.70}, fr_w ∈ {0.40, 0.42, 0.45}, cc50_cat_w=0.25.
"""
import itertools
import time

import pandas as pd

from run_local_signal_search import clear_fold_cache, competition_score, ensure_data, fit_oof, load_data, per_target_rmse
from run_phase_b_si_ic50 import SEEDS, frozen_cc50, load_ext_cols
from run_phase_e_structural import ExtraHead, blend_extra_head, feature_blocks

REF_IC50 = 0.65
REF_FR = 0.42
CC50_W = 0.25


def eval_cfg(name, ic50_w, fr_w, X, Xt, y, ext, blocks, ref_scores) -> dict:
    base = frozen_cc50(ext, ic50_cat_w=ic50_w, cc50_cat_w=CC50_W)
    head = ExtraHead("fr", blocks["fr_only"], 0, fr_w)
    scores, deltas = [], []
    for seed in SEEDS:
        clear_fold_cache()
        bo, bt = fit_oof(X, Xt, y, base, random_state=seed)
        oof, _ = blend_extra_head(X, Xt, y, bo, bt, head.cols, head.target, head.weight, seed)
        s = competition_score(y, oof)
        scores.append(s)
        deltas.append(s - ref_scores[seed])
    clear_fold_cache()
    bo, bt = fit_oof(X, Xt, y, base, random_state=42)
    oof42, _ = blend_extra_head(X, Xt, y, bo, bt, head.cols, head.target, head.weight, 42)
    rmse = per_target_rmse(y, oof42)
    return {
        "name": name,
        "ic50_w": ic50_w,
        "fr_w": fr_w,
        "s42": scores[0],
        "s2024": scores[1],
        "s7": scores[2],
        "mean": sum(scores) / len(scores),
        "mean_d": sum(deltas) / len(deltas),
        "wins": sum(1 for d in deltas if d < -0.05),
        "ic50": rmse[0],
        "cc50": rmse[1],
        "si": rmse[2],
    }


def main() -> None:
    ensure_data()
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)

    ref_base = frozen_cc50(ext, ic50_cat_w=REF_IC50, cc50_cat_w=CC50_W)
    ref_head = ExtraHead("fr", blocks["fr_only"], 0, REF_FR)
    ref_scores = {}
    for seed in SEEDS:
        clear_fold_cache()
        bo, bt = fit_oof(X, Xt, y, ref_base, seed)
        oof, _ = blend_extra_head(X, Xt, y, bo, bt, ref_head.cols, 0, REF_FR, seed)
        ref_scores[seed] = competition_score(y, oof)

    print(f"ref ic50={REF_IC50} fr={REF_FR} cc50={CC50_W}:", {s: round(ref_scores[s], 2) for s in SEEDS})
    t0 = time.perf_counter()
    rows = []

    ic50_vals = [0.62, 0.65, 0.68, 0.70]
    fr_vals = [0.40, 0.42, 0.45]
    for iw, fw in itertools.product(ic50_vals, fr_vals):
        tag = f"ic{int(iw*100)}_fr{int(fw*100)}"
        rows.append(eval_cfg(tag, iw, fw, X, Xt, y, ext, blocks, ref_scores))

    df = pd.DataFrame(rows).sort_values("mean_d")
    print("\n=== narrow grid (vs ref ic65_fr42) ===")
    print(df.to_string(index=False))
    out = "phase_e_tune_narrow.csv"
    df.to_csv(out, index=False)
    best = df.iloc[0]
    print(f"\nBest OOF: {best['name']} mean_d={best['mean_d']:.3f} ic50={best['ic50']:.2f}")
    print(f"Saved {out}")
    print(f"Время: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
