"""Тюнинг ic50_cat_w вокруг public best 280.13 (w=0.35). CC50/SI frozen."""
import time

from run_local_signal_search import (
    clear_fold_cache,
    competition_score,
    ensure_data,
    fit_oof,
    load_data,
    per_target_rmse,
)
from run_phase_b_si_ic50 import SEEDS, frozen_cc50, load_ext_cols

WEIGHTS = [0.30, 0.32, 0.33, 0.35, 0.37, 0.38, 0.40]
REF_W = 0.35


def main() -> None:
    ensure_data()
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    ref = frozen_cc50(ext, ic50_cat_w=REF_W)

    ref_scores = {}
    for seed in SEEDS:
        clear_fold_cache()
        oof, _ = fit_oof(X, Xt, y, ref, random_state=seed)
        ref_scores[seed] = competition_score(y, oof)

    print(f"ref ic50_w{REF_W}: " + " ".join(f"s{s}={ref_scores[s]:.2f}" for s in SEEDS))
    print(f"{'w':>6} " + " ".join(f"s{s:>7}" for s in SEEDS) + "  mean_d  wins  IC50 RMSE")
    print("-" * 62)

    t0 = time.perf_counter()
    results = []
    for w in WEIGHTS:
        cfg = frozen_cc50(ext, ic50_cat_w=w)
        scores, deltas = [], []
        for seed in SEEDS:
            oof, _ = fit_oof(X, Xt, y, cfg, random_state=seed)
            s = competition_score(y, oof)
            scores.append(s)
            deltas.append(s - ref_scores[seed])
        wins = sum(1 for d in deltas if d < -0.05)
        mean_d = sum(deltas) / len(deltas)
        oof42, _ = fit_oof(X, Xt, y, cfg, random_state=42)
        ic50_rmse = per_target_rmse(y, oof42)[0]
        results.append((mean_d, wins, w, scores, ic50_rmse))
        mark = " *" if w == REF_W else ""
        print(
            f"{w:6.2f} " + " ".join(f"{s:7.2f}" for s in scores)
            + f"  {mean_d:+6.2f}  {wins}/3  {ic50_rmse:7.1f}{mark}"
        )

    results.sort()
    best = results[0]
    print(f"\nЛучший OOF: w={best[2]} (mean_d={best[0]:+.2f}, wins={best[1]}/3)")
    print(f"Время: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
