from run_phase_e_structural import ExtraHead, blend_extra_head, feature_blocks
from run_phase_b_si_ic50 import SEEDS, frozen_cc50, load_ext_cols
from run_local_signal_search import clear_fold_cache, competition_score, ensure_data, fit_oof, load_data

ensure_data()
X, Xt, y, cols = load_data()
ext = load_ext_cols(cols)
base = frozen_cc50(ext, ic50_cat_w=0.55, cc50_cat_w=0.25)
blocks = feature_blocks(cols)
combos = [
    ("fr_ic50_w35", [ExtraHead("a", blocks["fr_only"], 0, 0.35)]),
    ("mordred_ic50_w35", [ExtraHead("b", blocks["mordred"], 0, 0.35)]),
    ("fr_ic50+mord_si", [
        ExtraHead("a", blocks["fr_only"], 0, 0.35),
        ExtraHead("b", blocks["mordred"], 2, 0.35),
    ]),
]
for name, heads in combos:
    ds = []
    for seed in SEEDS:
        clear_fold_cache()
        bo, bt = fit_oof(X, Xt, y, base, seed)
        ref = competition_score(y, bo)
        oof = bo.copy()
        for h in heads:
            oof, _ = blend_extra_head(X, Xt, y, oof, bt, h.cols, h.target, h.weight, seed)
        ds.append(competition_score(y, oof) - ref)
    print(name, f"mean_d={sum(ds)/3:+.2f}", f"wins={sum(1 for d in ds if d<-0.1)}/3", [round(d, 2) for d in ds])
