"""Стабильность топ-кандидатов по OOF на нескольких seed."""
from run_local_signal_search import (
    EXTENDED_IC50_FEATURES,
    PipelineConfig,
    clear_fold_cache,
    competition_score,
    ensure_data,
    fit_oof,
    load_data,
    per_target_rmse,
)

SEEDS = [42, 2024, 7]

CANDIDATES = [
    ("baseline_final", PipelineConfig()),
    ("cc50_trans_k3_w60", PipelineConfig(cc50_trans_k=3)),
    ("cc50_trans_k3_w55", PipelineConfig(cc50_trans_k=3, cc50_blend_w=0.55)),
    ("cc50_trans_k3_w65", PipelineConfig(cc50_trans_k=3, cc50_blend_w=0.65)),
    ("cc50_cat_w15", PipelineConfig(cc50_cat_w=0.15)),
    ("cc50_cat_w20", PipelineConfig(cc50_cat_w=0.20)),
    ("cc50_cat_w10", PipelineConfig(cc50_cat_w=0.10)),
    ("combo_ext_ic50_cc15", PipelineConfig(
        ic50_cat_cols=None,  # patched below
        cc50_cat_w=0.15,
    )),
    ("cc50_k3_cat15", PipelineConfig(cc50_trans_k=3, cc50_cat_w=0.15)),
    ("ic50_ext_cc50_k3", PipelineConfig(cc50_trans_k=3)),  # ext patched
]


def main() -> None:
    ensure_data()
    X_train, X_test, y_train, all_cols = load_data()
    ext_cols = [c for c in EXTENDED_IC50_FEATURES if c in all_cols]

    print(f"{'candidate':<22} " + " ".join(f"s{s:>7}" for s in SEEDS) + "  mean_d  wins")
    print("-" * 70)

    base_scores = {}
    for seed in SEEDS:
        clear_fold_cache()
        oof, _ = fit_oof(X_train, X_test, y_train, PipelineConfig(), random_state=seed)
        base_scores[seed] = competition_score(y_train, oof)

    for name, cfg in CANDIDATES:
        if name == "combo_ext_ic50_cc15":
            cfg = PipelineConfig(ic50_cat_cols=ext_cols, cc50_cat_w=0.15)
        elif name == "ic50_ext_cc50_k3":
            cfg = PipelineConfig(ic50_cat_cols=ext_cols, cc50_trans_k=3)
        scores = []
        for seed in SEEDS:
            oof, _ = fit_oof(X_train, X_test, y_train, cfg, random_state=seed)
            scores.append(competition_score(y_train, oof))
        deltas = [scores[i] - base_scores[SEEDS[i]] for i in range(len(SEEDS))]
        wins = sum(1 for d in deltas if d < -0.05)
        line = f"{name:<22} " + " ".join(f"{s:7.2f}" for s in scores)
        line += f"  {sum(deltas)/len(deltas):+6.2f}  {wins}/{len(SEEDS)}"
        print(line)
        if wins >= 2:
            oof42, _ = fit_oof(X_train, X_test, y_train, cfg, random_state=42)
            rmse = per_target_rmse(y_train, oof42)
            print(f"    seed42 per-target: IC50={rmse[0]:.1f} CC50={rmse[1]:.1f} SI={rmse[2]:.1f}")


if __name__ == "__main__":
    main()
