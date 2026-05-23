"""Узкая сетка вокруг ext_k3_cat15 (public 280.76)."""
import time

from run_local_signal_search import (
    EXTENDED_IC50_FEATURES,
    PipelineConfig,
    cache_stats,
    clear_fold_cache,
    competition_score,
    ensure_data,
    fit_oof,
    load_data,
)

SEEDS = [42, 2024, 7]


def with_ext(ext_cols: list[str], **kw) -> PipelineConfig:
    defaults = dict(
        ic50_cat_cols=ext_cols,
        ic50_cat_w=0.25,
        cc50_trans_k=3,
        cc50_blend_w=0.60,
        cc50_cat_w=0.15,
    )
    defaults.update(kw)
    return PipelineConfig(**defaults)


def with_size(**kw) -> PipelineConfig:
    defaults = dict(
        ic50_cat_cols=None,
        cc50_trans_k=3,
        cc50_blend_w=0.60,
        cc50_cat_w=0.15,
    )
    defaults.update(kw)
    return PipelineConfig(**defaults)


def get_candidates(ext_cols: list[str]) -> list[tuple[str, PipelineConfig]]:
    return [
        ("ext_k3_cat12", with_ext(ext_cols, cc50_cat_w=0.12)),
        ("ext_k3_cat18", with_ext(ext_cols, cc50_cat_w=0.18)),
        ("ext_k3_cat20", with_ext(ext_cols, cc50_cat_w=0.20)),
        ("ext_k3_w55", with_ext(ext_cols, cc50_blend_w=0.55)),
        ("ext_k3_w65", with_ext(ext_cols, cc50_blend_w=0.65)),
        ("ext_k3_w55_c12", with_ext(ext_cols, cc50_blend_w=0.55, cc50_cat_w=0.12)),
        ("ext_k3_w55_c18", with_ext(ext_cols, cc50_blend_w=0.55, cc50_cat_w=0.18)),
        ("ext_k3_w65_c12", with_ext(ext_cols, cc50_blend_w=0.65, cc50_cat_w=0.12)),
        ("ext_k3_w65_c18", with_ext(ext_cols, cc50_blend_w=0.65, cc50_cat_w=0.18)),
        ("k3_cat15_size", with_size()),
        ("k3_cat12_size", with_size(cc50_cat_w=0.12)),
        ("k3_cat18_size", with_size(cc50_cat_w=0.18)),
        ("k3_w55_cat15", with_size(cc50_blend_w=0.55)),
        ("k3_w65_cat15", with_size(cc50_blend_w=0.65)),
    ]


def main() -> None:
    ensure_data()
    _, _, _, all_cols = load_data()
    ext_cols = [c for c in EXTENDED_IC50_FEATURES if c in all_cols]
    candidates = get_candidates(ext_cols)

    ref_cfg = with_ext(ext_cols)
    X_train, X_test, y_train, _ = load_data()
    ref_by_seed = {}
    t0 = time.perf_counter()
    for seed in SEEDS:
        clear_fold_cache()
        oof, _ = fit_oof(X_train, X_test, y_train, ref_cfg, random_state=seed)
        ref_by_seed[seed] = competition_score(y_train, oof)

    print("ref ext_k3_cat15:", [round(ref_by_seed[s], 2) for s in SEEDS])
    print(f"{'name':<24} " + " ".join(f"s{s}" for s in SEEDS) + "  mean_d  wins")
    print("-" * 72)

    results = []
    for name, cfg in candidates:
        scores = []
        for seed in SEEDS:
            oof, _ = fit_oof(X_train, X_test, y_train, cfg, random_state=seed)
            scores.append(competition_score(y_train, oof))
        deltas = [scores[i] - ref_by_seed[SEEDS[i]] for i in range(len(SEEDS))]
        wins = sum(1 for d in deltas if d < -0.05)
        mean_d = sum(deltas) / len(deltas)
        results.append((mean_d, wins, name, scores, deltas))

    results.sort()
    for mean_d, wins, name, scores, _ in results:
        line = f"{name:<24} " + " ".join(f"{s:6.2f}" for s in scores)
        line += f"  {mean_d:+6.2f}  {wins}/3"
        print(line)

    elapsed = time.perf_counter() - t0
    print(f"\nВремя: {elapsed:.1f}s, cache entries (last seed): {cache_stats()}")

    best = results[0]
    if best[0] < -0.1:
        print(f"Лучше ref: {best[2]} (mean_d={best[0]:+.2f})")
    else:
        print("Ref ext_k3_cat15 остаётся оптимумом в сетке.")


if __name__ == "__main__":
    main()
