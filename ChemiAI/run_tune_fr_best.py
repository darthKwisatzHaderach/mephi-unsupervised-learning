"""
Дотюнинг best: w55_cc25 + fr_only IC50 CatBoost.
Public ref: 275.30064 (fr_w=0.35).
"""
import time

import pandas as pd

from run_local_signal_search import clear_fold_cache, competition_score, ensure_data, fit_oof, load_data, per_target_rmse
from run_phase_b_si_ic50 import SEEDS, frozen_cc50, load_ext_cols
from run_phase_e_structural import ExtraHead, blend_extra_head, feature_blocks

REF_FR_W = 0.35


def base_cfg(ext, **kw):
    d = dict(ic50_cat_w=0.55, cc50_cat_w=0.25)
    d.update(kw)
    return frozen_cc50(ext, **d)


def eval_cfg(name, base, head, X, Xt, y, ref_scores) -> dict:
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
        "s42": scores[0],
        "s2024": scores[1],
        "s7": scores[2],
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
    ref_base = base_cfg(ext)
    ref_head = ExtraHead("fr", blocks["fr_only"], 0, REF_FR_W)

    ref_scores = {}
    for seed in SEEDS:
        clear_fold_cache()
        bo, bt = fit_oof(X, Xt, y, ref_base, seed)
        oof, _ = blend_extra_head(X, Xt, y, bo, bt, ref_head.cols, 0, REF_FR_W, seed)
        ref_scores[seed] = competition_score(y, oof)

    print("ref fr_w0.35:", {s: round(ref_scores[s], 2) for s in SEEDS})
    t0 = time.perf_counter()
    rows = []

    # 1) fr weight
    for w in [0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.42, 0.45, 0.50]:
        head = ExtraHead(f"fr_w{int(w*100)}", blocks["fr_only"], 0, w)
        rows.append(eval_cfg(f"fr_w{int(w*100)}", ref_base, head, X, Xt, y, ref_scores))

    df1 = pd.DataFrame(rows).sort_values("mean_d")
    print("\n=== fr weight ===")
    print(df1.to_string(index=False))

    best_fr_w = float(df1.iloc[0]["name"].split("_w")[1]) / 100
    best_base = ref_base

    # 2) CC50 cat_w при лучшем fr_w
    rows2 = []
    for cw in [0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
        b = base_cfg(ext, cc50_cat_w=cw)
        head = ExtraHead("fr", blocks["fr_only"], 0, best_fr_w)
        rows2.append(eval_cfg(f"cc{cw:.2f}_fr{int(best_fr_w*100)}", b, head, X, Xt, y, ref_scores))

    df2 = pd.DataFrame(rows2).sort_values("mean_d")
    print("\n=== cc50_cat_w (vs fr at cc=0.25) ===")
    print(df2.to_string(index=False))

    # 3) IC50 ext cat_w
    rows3 = []
    for iw in [0.45, 0.50, 0.55, 0.60, 0.65]:
        b = base_cfg(ext, ic50_cat_w=iw)
        head = ExtraHead("fr", blocks["fr_only"], 0, best_fr_w)
        rows3.append(eval_cfg(f"ic50_{iw:.2f}_fr{int(best_fr_w*100)}", b, head, X, Xt, y, ref_scores))

    df3 = pd.DataFrame(rows3).sort_values("mean_d")
    print("\n=== ic50_cat_w (vs ref fr35) ===")
    print(df3.to_string(index=False))

    # 4) mordred vs fr same w
    for block in ["mordred", "fr_only"]:
        w = best_fr_w
        head = ExtraHead(block, blocks[block], 0, w)
        rows.append(eval_cfg(f"{block}_w{int(w*100)}", ref_base, head, X, Xt, y, ref_scores))

    pd.concat([df1, df2, df3]).drop_duplicates("name").sort_values("mean_d").to_csv(
        "phase_e_tune_best.csv", index=False
    )
    print(f"\nBest fr_w from grid: {best_fr_w}")
    print(f"Время: {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
