"""Фаза D2: SI = CatBoost top-k вместо robust HGB (не blend поверх)."""
import time

from run_local_signal_search import clear_fold_cache, competition_score, ensure_data, fit_oof, load_data, per_target_rmse
from run_phase_b_si_ic50 import SEEDS, frozen_cc50, load_ext_cols


def base(ext, **kw):
    d = dict(ic50_cat_w=0.55, cc50_cat_w=0.25, si_robust_w=0.0)
    d.update(kw)
    return frozen_cc50(ext, **d)


def main() -> None:
    ensure_data()
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    ref = frozen_cc50(ext, ic50_cat_w=0.55, cc50_cat_w=0.25)

    ref_s = {}
    for seed in SEEDS:
        clear_fold_cache()
        oof, _ = fit_oof(X, Xt, y, ref, seed)
        ref_s[seed] = competition_score(y, oof)

    print(f"ref w55_cc25: {ref_s}")
    print(f"{'name':<24} s42    s2024  s7     mean_d wins  SI")

    candidates = []
    for w in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 1.00]:
        for k in [90, 120, 150]:
            candidates.append((f"w{int(w*100)}_k{k}", base(ext, si_catboost_w=w, si_topk=k)))
    for k in [120, 150]:
        candidates.append((f"w50_k{k}_mae", base(ext, si_catboost_w=0.50, si_topk=k, si_catboost_mae=True)))

    t0 = time.perf_counter()
    best = None
    for name, cfg in candidates:
        scs, ds = [], []
        for seed in SEEDS:
            oof, _ = fit_oof(X, Xt, y, cfg, seed)
            scs.append(competition_score(y, oof))
            ds.append(scs[-1] - ref_s[seed])
        md = sum(ds) / 3
        wins = sum(1 for d in ds if d < -0.1)
        si = per_target_rmse(y, fit_oof(X, Xt, y, cfg, 42)[0])[2]
        print(f"{name:<24} " + " ".join(f"{s:6.2f}" for s in scs) + f"  {md:+6.2f}  {wins}/3  {si:.0f}")
        if best is None or md < best[0]:
            best = (md, name, cfg)

    print(f"\nBest: {best[1]} ({best[0]:+.2f})")
    print(f"Время: {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
