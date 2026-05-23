"""
Фаза B: SI (alpha, robust w) и IC50 (ext/transductive) при замороженном CC50 ext_k3_cat15.
Public ref: 280.75531
"""
import time

import pandas as pd

from run_local_signal_search import (
    EXTENDED_IC50_FEATURES,
    SIZE_FEATURE_NAMES,
    PipelineConfig,
    clear_fold_cache,
    competition_score,
    ensure_data,
    fit_oof,
    load_data,
    per_target_rmse,
)

SEEDS = [42, 2024, 7]


def load_ext_cols(all_cols: list[str]) -> list[str]:
    return [c for c in EXTENDED_IC50_FEATURES if c in all_cols]


def frozen_cc50(ext_cols: list[str], **kw) -> PipelineConfig:
    """CC50 = ext_k3_cat15; меняем только IC50/SI через kw."""
    base = dict(
        ic50_cat_cols=ext_cols,
        ic50_cat_w=0.25,
        ic50_trans_w=0.0,
        cc50_trans_k=3,
        cc50_blend_w=0.60,
        cc50_cat_w=0.15,
        si_robust_w=0.30,
        si_alpha=0.35,
    )
    base.update(kw)
    return PipelineConfig(**base)


def eval_candidates(
    ref_cfg: PipelineConfig,
    candidates: list[tuple[str, PipelineConfig]],
    y_train,
    X_train,
    X_test,
) -> pd.DataFrame:
    ref_scores = {}
    rows = []
    for seed in SEEDS:
        clear_fold_cache()
        oof_ref, _ = fit_oof(X_train, X_test, y_train, ref_cfg, random_state=seed)
        ref_scores[seed] = competition_score(y_train, oof_ref)

        for name, cfg in candidates:
            oof, _ = fit_oof(X_train, X_test, y_train, cfg, random_state=seed)
            score = competition_score(y_train, oof)
            rmse = per_target_rmse(y_train, oof)
            rows.append({
                "candidate": name,
                "seed": seed,
                "oof": score,
                "delta_vs_ref": score - ref_scores[seed],
                "rmse_ic50": rmse[0],
                "rmse_cc50": rmse[1],
                "rmse_si": rmse[2],
            })

    df = pd.DataFrame(rows)
    summary = df.groupby("candidate").agg(
        oof_mean=("oof", "mean"),
        delta_mean=("delta_vs_ref", "mean"),
        delta_min=("delta_vs_ref", "min"),
        wins=("delta_vs_ref", lambda s: int((s < -0.05).sum())),
        rmse_ic50=("rmse_ic50", "mean"),
        rmse_cc50=("rmse_cc50", "mean"),
        rmse_si=("rmse_si", "mean"),
    ).reset_index().sort_values("delta_mean")
    return summary, df


def main() -> None:
    ensure_data()
    X_train, X_test, y_train, all_cols = load_data()
    ext_cols = [c for c in EXTENDED_IC50_FEATURES if c in all_cols]
    size_cols = [c for c in SIZE_FEATURE_NAMES if c in all_cols]

    ref = frozen_cc50(ext_cols)
    t0 = time.perf_counter()

    # --- SI grid ---
    si_candidates: list[tuple[str, PipelineConfig]] = []
    for alpha in [0.28, 0.30, 0.32, 0.33, 0.35, 0.37, 0.40, 0.42, 0.45]:
        si_candidates.append((f"si_a{int(alpha*100):02d}", frozen_cc50(ext_cols, si_alpha=alpha)))
    for w in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        si_candidates.append((f"si_rw{int(w*100):02d}", frozen_cc50(ext_cols, si_robust_w=w)))
    # пара alpha + robust
    for alpha, w in [(0.32, 0.25), (0.33, 0.25), (0.37, 0.35), (0.40, 0.35), (0.30, 0.20)]:
        si_candidates.append((
            f"si_a{int(alpha*100)}_rw{int(w*100)}",
            frozen_cc50(ext_cols, si_alpha=alpha, si_robust_w=w),
        ))

    print("=== SI grid (CC50 frozen) ===")
    si_sum, si_det = eval_candidates(ref, si_candidates, y_train, X_train, X_test)
    print(si_sum.head(12).to_string(index=False))
    si_sum.to_csv("phase_b_si_grid.csv", index=False)
    si_det.to_csv("phase_b_si_grid.detail.csv", index=False)

    # --- IC50 grid ---
    ic50_candidates: list[tuple[str, PipelineConfig]] = []
    ic50_candidates.append(("ic50_size_cols", frozen_cc50(size_cols)))
    ic50_candidates.append(("ic50_ext_cols", frozen_cc50(ext_cols)))  # ref IC50
    for w in [0.15, 0.20, 0.30, 0.35]:
        ic50_candidates.append((f"ic50_cat_w{int(w*100)}", frozen_cc50(ext_cols, ic50_cat_w=w)))
    for tw in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
        ic50_candidates.append((f"ic50_trans_w{int(tw*100)}", frozen_cc50(ext_cols, ic50_trans_w=tw)))
    for tw in [0.10, 0.15]:
        ic50_candidates.append((
            f"ic50_size_trans_w{int(tw*100)}",
            frozen_cc50(size_cols, ic50_trans_w=tw),
        ))
    ic50_candidates.append((
        "ic50_ext_trans10_cat20",
        frozen_cc50(ext_cols, ic50_trans_w=0.10, ic50_cat_w=0.20),
    ))

    print("\n=== IC50 grid (CC50 frozen) ===")
    ic50_sum, ic50_det = eval_candidates(ref, ic50_candidates, y_train, X_train, X_test)
    print(ic50_sum.head(12).to_string(index=False))
    ic50_sum.to_csv("phase_b_ic50_grid.csv", index=False)
    ic50_det.to_csv("phase_b_ic50_grid.detail.csv", index=False)

    best_si = si_sum.iloc[0]
    best_ic50 = ic50_sum.iloc[0]
    si_map = {n: c for n, c in si_candidates}
    ic50_map = {n: c for n, c in ic50_candidates}
    best_si_name = str(best_si["candidate"])
    best_ic50_name = str(best_ic50["candidate"])
    si_cfg = si_map[best_si_name]
    ic50_cfg = ic50_map[best_ic50_name]

    combo_candidates = [
        (f"combo_{best_si_name}", si_cfg),
        (f"combo_{best_ic50_name}", ic50_cfg),
        ("combo_best_si_ic50", frozen_cc50(
            ext_cols,
            ic50_cat_cols=ic50_cfg.ic50_cat_cols,
            ic50_cat_w=ic50_cfg.ic50_cat_w,
            ic50_trans_w=ic50_cfg.ic50_trans_w,
            si_robust_w=si_cfg.si_robust_w,
            si_alpha=si_cfg.si_alpha,
        )),
    ]

    print("\n=== Top combos ===")
    combo_sum, _ = eval_candidates(ref, combo_candidates, y_train, X_train, X_test)
    print(combo_sum.to_string(index=False))
    combo_sum.to_csv("phase_b_combos.csv", index=False)

    elapsed = time.perf_counter() - t0
    print(f"\nRef OOF seeds: ", end="")
    for seed in SEEDS:
        clear_fold_cache()
        oof, _ = fit_oof(X_train, X_test, y_train, ref, random_state=seed)
        print(f"{seed}={competition_score(y_train, oof):.2f} ", end="")
    print(f"\nВремя: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
