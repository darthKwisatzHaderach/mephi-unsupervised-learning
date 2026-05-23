"""Топ-комбо фазы B: ic50_cat_w35 + si_alpha."""
from run_phase_b_si_ic50 import SEEDS, frozen_cc50
from run_local_signal_search import (
    EXTENDED_IC50_FEATURES,
    clear_fold_cache,
    competition_score,
    ensure_data,
    fit_oof,
    load_data,
)

ensure_data()
X, Xt, y, cols = load_data()
ext = [c for c in EXTENDED_IC50_FEATURES if c in cols]
ref = frozen_cc50(ext)
cfgs = [
    ("ref_cat15", ref),
    ("ic50_cat_w35", frozen_cc50(ext, ic50_cat_w=0.35)),
    ("si_a45", frozen_cc50(ext, si_alpha=0.45)),
    ("combo_w35_a45", frozen_cc50(ext, ic50_cat_w=0.35, si_alpha=0.45)),
    ("combo_w35_a42", frozen_cc50(ext, ic50_cat_w=0.35, si_alpha=0.42)),
    ("combo_w32_a45", frozen_cc50(ext, ic50_cat_w=0.30, si_alpha=0.45)),
]

ref_scores = {}
for seed in SEEDS:
    clear_fold_cache()
    oof, _ = fit_oof(X, Xt, y, ref, random_state=seed)
    ref_scores[seed] = competition_score(y, oof)

print(f"{'name':<18} s42    s2024  s7     mean_d wins")
for name, cfg in cfgs:
    scores, deltas = [], []
    for seed in SEEDS:
        oof, _ = fit_oof(X, Xt, y, cfg, random_state=seed)
        s = competition_score(y, oof)
        scores.append(s)
        deltas.append(s - ref_scores[seed])
    wins = sum(1 for d in deltas if d < -0.05)
    print(
        f"{name:<18} "
        + " ".join(f"{s:6.2f}" for s in scores)
        + f"  {sum(deltas)/len(deltas):+6.2f}  {wins}/3"
    )
