"""Фаза C2: SI + combo w55/cc50 при ref w40 public 279.92."""
import time

from run_local_signal_search import clear_fold_cache, competition_score, ensure_data, fit_oof, load_data, per_target_rmse
from run_phase_b_si_ic50 import SEEDS, frozen_cc50, load_ext_cols

REF_W = 0.40


def fb(ext, **kw):
    base = dict(ic50_cat_w=REF_W)
    base.update(kw)
    return frozen_cc50(ext, **base)


def main() -> None:
    ensure_data()
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    ref = fb(ext)

    ref_s = {}
    for seed in SEEDS:
        clear_fold_cache()
        oof, _ = fit_oof(X, Xt, y, ref, random_state=seed)
        ref_s[seed] = competition_score(y, oof)

    print(f"ref w40: {ref_s}")
    print(f"{'name':<22} s42    s2024  s7     mean_d wins  SI_RMSE")

    candidates = []
    for w in [0.45, 0.50, 0.55, 0.60, 0.65]:
        candidates.append((f"ic50_w{w}", fb(ext, ic50_cat_w=w)))
    for a in [0.25, 0.28, 0.30, 0.32]:
        candidates.append((f"si_a{int(a*100)}", fb(ext, si_alpha=a)))
    for rw in [0.35, 0.40, 0.45, 0.50]:
        candidates.append((f"si_rw{int(rw*100)}", fb(ext, si_robust_w=rw)))
    candidates += [
        ("w50_cc25", fb(ext, ic50_cat_w=0.50, cc50_cat_w=0.25)),
        ("w55_cc25", fb(ext, ic50_cat_w=0.55, cc50_cat_w=0.25)),
        ("w55_cc20", fb(ext, ic50_cat_w=0.55, cc50_cat_w=0.20)),
        ("w50_cc20", fb(ext, ic50_cat_w=0.50, cc50_cat_w=0.20)),
        ("w50_a30", fb(ext, ic50_cat_w=0.50, si_alpha=0.30)),
    ]

    t0 = time.perf_counter()
    best = None
    for name, cfg in candidates:
        scs, ds = [], []
        for seed in SEEDS:
            oof, _ = fit_oof(X, Xt, y, cfg, random_state=seed)
            scs.append(competition_score(y, oof))
            ds.append(scs[-1] - ref_s[seed])
        wins = sum(1 for d in ds if d < -0.1)
        md = sum(ds) / 3
        si_rmse = per_target_rmse(y, fit_oof(X, Xt, y, cfg, 42)[0])[2]
        print(f"{name:<22} " + " ".join(f"{s:6.2f}" for s in scs) + f"  {md:+6.2f}  {wins}/3  {si_rmse:.0f}")
        if best is None or md < best[0]:
            best = (md, name)

    print(f"\nBest: {best[1]} ({best[0]:+.2f})")
    print(f"Время: {time.perf_counter()-t0:.0f}s")

    # оценка потолка: что нужно для OOF~500 (грубая цель LB 265)
    oof42, _ = fit_oof(X, Xt, y, ref, 42)
    r = per_target_rmse(y, oof42)
    print(f"\nRef per-target OOF42: IC50={r[0]:.0f} CC50={r[1]:.0f} SI={r[2]:.0f} mean={sum(r)/3:.0f}")
    print("Для mean OOF~500 нужно суммарно -80 RMSE; SI={:.0f} даёт ~{:.0f}% mean — главный рычаг.".format(
        r[2], r[2] / sum(r) * 100
    ))


if __name__ == "__main__":
    main()
