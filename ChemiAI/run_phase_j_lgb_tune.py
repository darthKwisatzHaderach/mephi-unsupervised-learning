"""
Phase J2 — микро-тюнинг LGB IC50 w поверх Phase I (public best lgb_ic20 = 270.49).

Запуск:
  python run_phase_j_lgb_tune.py
  python run_phase_j_lgb_tune.py --quick
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from run_local_signal_search import clear_fold_cache, competition_score, ensure_data, load_data, per_target_rmse
from run_phase_b_si_ic50 import SEEDS, load_ext_cols
from run_phase_e_structural import feature_blocks
from run_phase_j import blend_ic50_lgb, fit_phase_i

# public: ic15=270.86, ic20=270.49 — ищем оптимум ~0.18–0.28
LGB_WEIGHTS = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    seeds = [42] if args.quick else list(SEEDS)

    ensure_data()
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)
    t0 = time.perf_counter()

    ref_scores: dict[int, float] = {}
    for seed in seeds:
        clear_fold_cache()
        oof, _ = fit_phase_i(X, Xt, y, ext, blocks, seed)
        ref_scores[seed] = competition_score(y, oof)
    print(f"REF phase_i: " + ", ".join(f"s{s}={ref_scores[s]:.2f}" for s in seeds))

    rows: list[dict] = []
    for w in LGB_WEIGHTS:
        name = f"lgb_ic{int(round(w * 100)):02d}" if w < 0.1 else f"lgb_ic{int(round(w * 100))}"
        # единообразные имена: lgb_ic10, lgb_ic18, ...
        name = f"lgb_ic{int(round(w * 100))}"
        print(f"  w={w:.2f} ({name})...", flush=True, end=" ")
        for seed in seeds:
            clear_fold_cache()
            oof, test = fit_phase_i(X, Xt, y, ext, blocks, seed)
            oof, _ = blend_ic50_lgb(X, Xt, y, oof, test, w, seed)
            sc = competition_score(y, oof)
            rmse = per_target_rmse(y, oof)
            d = sc - ref_scores[seed]
            rows.append({
                "name": name,
                "weight": w,
                "seed": seed,
                "oof": sc,
                "delta": d,
                "ic50": rmse[0],
                "cc50": rmse[1],
                "si": rmse[2],
            })
            print(f"s{seed}={sc:.2f}({d:+.2f})", end=" ")
        print()

    df = pd.DataFrame(rows)
    sm = (
        df.groupby(["name", "weight"])
        .agg(
            oof_mean=("oof", "mean"),
            delta_mean=("delta", "mean"),
            min_delta=("delta", "min"),
            ic50=("ic50", "mean"),
        )
        .reset_index()
        .sort_values("delta_mean")
    )
    sm.to_csv("phase_j_lgb_tune.csv", index=False)
    df.to_csv("phase_j_lgb_tune.detail.csv", index=False)

    print(f"\n=== LGB tune ({time.perf_counter()-t0:.0f}s) ===")
    print(sm.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("\nPublic ref: ic20=270.49, ic28=269.99")
    print("  python make_submission_phase_j.py lgb_ic25")


if __name__ == "__main__":
    main()
