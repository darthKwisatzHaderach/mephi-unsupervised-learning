"""
Phase G: прорыв к ~260 — только НЕ закрытые ветки на fr42_ic65 base.
Ref OOF ~528.03, public 274.76.
"""
import time

import numpy as np
import pandas as pd

from run_local_signal_search import clear_fold_cache, competition_score, ensure_data, fit_oof, load_data, per_target_rmse
from run_phase_b_si_ic50 import SEEDS, frozen_cc50, load_ext_cols
from run_phase_e_structural import blend_extra_head, feature_blocks

IC50_W = 0.65
CC50_W = 0.25
FR_W = 0.42


def best_cfg(ext, **kw):
    d = dict(ic50_cat_w=IC50_W, cc50_cat_w=CC50_W)
    d.update(kw)
    return frozen_cc50(ext, **d)


def fit_fr_base(X, Xt, y, ext, blocks, seed):
    clear_fold_cache()
    oof, test = fit_oof(X, Xt, y, best_cfg(ext), seed)
    return blend_extra_head(X, Xt, y, oof, test, blocks["fr_only"], 0, FR_W, seed)


def eval_name(name, fn, X, Xt, y, ext, blocks, ref_scores):
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


def main():
    ensure_data()
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)
    t0 = time.perf_counter()

    ref_scores = {}
    for seed in SEEDS:
        oof, _ = fit_fr_base(X, Xt, y, ext, blocks, seed)
        ref_scores[seed] = competition_score(y, oof)
    print("ref fr42_ic65:", {s: round(ref_scores[s], 2) for s in SEEDS})

    rows = []

    # --- G1: второй IC50 head (mordred / morgan) поверх fr ---
    for block in ["mordred", "morgan"]:
        for w in [0.15, 0.20, 0.25, 0.30, 0.35]:
            def fn(X, Xt, y, ext, blocks, seed, b=block, w=w):
                oof, test = fit_fr_base(X, Xt, y, ext, blocks, seed)
                return blend_extra_head(X, Xt, y, oof, test, blocks[b], 0, w, seed)

            rows.extend(eval_name(f"ic50_{block}_w{int(w*100)}", fn, X, Xt, y, ext, blocks, ref_scores))

    # --- G2: fr + mordred sequential IC50 ---
    for fw in [0.35, 0.42]:
        for mw in [0.15, 0.20, 0.25]:
            def fn(X, Xt, y, ext, blocks, seed, fw=fw, mw=mw):
                oof, test = fit_fr_base(X, Xt, y, ext, blocks, seed)
                oof, test = blend_extra_head(
                    X, Xt, y, oof, test, blocks["mordred"], 0, mw, seed,
                )
                return oof, test

            rows.extend(eval_name(f"fr{int(fw*100)}_mord{int(mw*100)}", fn, X, Xt, y, ext, blocks, ref_scores))

    # --- G3: CC50 k / blend / cat на fr42_ic65 ---
    for k in [2, 4, 5]:
        for bw in [0.55, 0.65, 0.70]:
            def fn(X, Xt, y, ext, blocks, seed, k=k, bw=bw):
                clear_fold_cache()
                cfg = best_cfg(ext, cc50_trans_k=k, cc50_blend_w=bw)
                oof, test = fit_oof(X, Xt, y, cfg, seed)
                return blend_extra_head(
                    X, Xt, y, oof, test, blocks["fr_only"], 0, FR_W, seed,
                )

            rows.extend(eval_name(f"cc_k{k}_b{int(bw*100)}", fn, X, Xt, y, ext, blocks, ref_scores))

    # --- G4: seed ensemble (OOF нет, но проверим mean test stability) ---
    tests = []
    for seed in [42, 2024, 7, 0, 1, 13, 77, 123, 456, 999]:
        clear_fold_cache()
        _, test = fit_fr_base(X, Xt, y, ext, blocks, seed)
        tests.append(test)
    ens = np.mean(tests, axis=0)
    print(f"\nSeed ensemble: test mean SI={ens[:,2].mean():.2f} vs seed42={tests[0][:,2].mean():.2f}")

    df = pd.DataFrame(rows)
    sm = df.groupby("name").agg(
        oof_mean=("oof", "mean"),
        delta_mean=("delta", "mean"),
        wins=("delta", lambda s: int((s < -0.05).sum())),
        ic50=("ic50", "mean"),
        cc50=("cc50", "mean"),
        si=("si", "mean"),
    ).reset_index().sort_values("delta_mean")

    print("\n=== TOP 20 Phase G ===")
    print(sm.head(20).to_string(index=False))
    sm.to_csv("phase_g_breakthrough.csv", index=False)
    df.to_csv("phase_g_breakthrough.detail.csv", index=False)
    print(f"\nВремя: {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
