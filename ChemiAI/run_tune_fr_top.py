"""Топ-комбо после run_tune_fr_best.py"""
from run_local_signal_search import clear_fold_cache, competition_score, ensure_data, fit_oof, load_data
from run_phase_b_si_ic50 import SEEDS, frozen_cc50, load_ext_cols
from run_phase_e_structural import ExtraHead, blend_extra_head, feature_blocks


def run(name, base_kw, fr_w, ref_scores):
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)
    kw = dict(ic50_cat_w=0.55, cc50_cat_w=0.25)
    kw.update(base_kw)
    base = frozen_cc50(ext, **kw)
    head = ExtraHead("fr", blocks["fr_only"], 0, fr_w)
    ds = []
    for seed in SEEDS:
        clear_fold_cache()
        bo, bt = fit_oof(X, Xt, y, base, seed)
        oof, _ = blend_extra_head(X, Xt, y, bo, bt, head.cols, 0, fr_w, seed)
        ds.append(competition_score(y, oof) - ref_scores[seed])
    print(f"{name:<28} d={sum(ds)/3:+.3f} wins={sum(1 for d in ds if d<-0.05)}/3  {[round(d,2) for d in ds]}")


def main():
    ensure_data()
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)
    ref_scores = {}
    for seed in SEEDS:
        clear_fold_cache()
        bo, bt = fit_oof(X, Xt, y, frozen_cc50(ext, ic50_cat_w=0.55, cc50_cat_w=0.25), seed)
        oof, _ = blend_extra_head(X, Xt, y, bo, bt, blocks["fr_only"], 0, 0.35, seed)
        ref_scores[seed] = competition_score(y, oof)

    print("ref fr35:", ref_scores)
    run("fr42", {}, 0.42, ref_scores)
    run("fr40", {}, 0.40, ref_scores)
    run("fr42_ic65", {"ic50_cat_w": 0.65}, 0.42, ref_scores)
    run("fr42_ic60", {"ic50_cat_w": 0.60}, 0.42, ref_scores)
    run("fr42_cc30", {"cc50_cat_w": 0.30}, 0.42, ref_scores)
    run("fr42_ic65_cc28", {"ic50_cat_w": 0.65, "cc50_cat_w": 0.28}, 0.42, ref_scores)
    run("fr40_ic65", {"ic50_cat_w": 0.65}, 0.40, ref_scores)
    run("fr42_ic65_cc30", {"ic50_cat_w": 0.65, "cc50_cat_w": 0.30}, 0.42, ref_scores)


if __name__ == "__main__":
    main()
