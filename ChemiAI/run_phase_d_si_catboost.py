"""
Фаза D: SI-only CatBoost top-k поверх public best w55_cc25 (279.81).
"""
import time

import pandas as pd

from run_local_signal_search import (
    PipelineConfig,
    clear_fold_cache,
    competition_score,
    ensure_data,
    fit_oof,
    load_data,
    per_target_rmse,
)
from run_phase_b_si_ic50 import SEEDS, frozen_cc50, load_ext_cols


def base_w55_cc25(ext: list[str], **kw) -> PipelineConfig:
    d = dict(ic50_cat_w=0.55, cc50_cat_w=0.25)
    d.update(kw)
    return frozen_cc50(ext, **d)


def eval_grid(ref, candidates, X, Xt, y) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        clear_fold_cache()
        oof_ref, _ = fit_oof(X, Xt, y, ref, random_state=seed)
        ref_s = competition_score(y, oof_ref)
        for name, cfg in candidates:
            oof, _ = fit_oof(X, Xt, y, cfg, random_state=seed)
            rmse = per_target_rmse(y, oof)
            rows.append({
                "candidate": name,
                "seed": seed,
                "oof": competition_score(y, oof),
                "delta": competition_score(y, oof) - ref_s,
                "si_rmse": rmse[2],
            })
    df = pd.DataFrame(rows)
    return df.groupby("candidate").agg(
        oof_mean=("oof", "mean"),
        delta_mean=("delta", "mean"),
        wins=("delta", lambda s: int((s < -0.1).sum())),
        si_rmse=("si_rmse", "mean"),
    ).reset_index().sort_values("delta_mean")


def main() -> None:
    ensure_data()
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    ref = base_w55_cc25(ext)
    t0 = time.perf_counter()

    # --- сетка top-k x weight ---
    grid = []
    for k in [60, 90, 120, 150]:
        for w in [0.25, 0.35, 0.45, 0.55]:
            grid.append((f"si_cb_k{k}_w{int(w*100)}", base_w55_cc25(ext, si_catboost_w=w, si_topk=k)))

    print("=== SI CatBoost top-k (ref w55_cc25) ===")
    s1 = eval_grid(ref, grid, X, Xt, y)
    print(s1.head(15).to_string(index=False))
    s1.to_csv("phase_d_si_catboost_grid.csv", index=False)

    # --- MAE + replace robust ---
    top_name = str(s1.iloc[0]["candidate"])
    top_k = int(top_name.split("_k")[1].split("_")[0])
    top_w = int(top_name.split("_w")[1]) / 100

    extras = [
        (f"si_cb_k{top_k}_w{int(top_w*100)}_mae", base_w55_cc25(
            ext, si_catboost_w=top_w, si_topk=top_k, si_catboost_mae=True)),
        ("si_cb_only_w50_k120", base_w55_cc25(
            ext, si_robust_w=0.0, si_catboost_w=0.50, si_topk=120)),
        ("si_cb_w45_no_robust", base_w55_cc25(
            ext, si_robust_w=0.0, si_catboost_w=0.45, si_topk=120)),
        ("si_cb_w55_k90", base_w55_cc25(ext, si_catboost_w=0.55, si_topk=90)),
        ("si_cb_w45_k150", base_w55_cc25(ext, si_catboost_w=0.45, si_topk=150)),
    ]
    print("\n=== MAE / ablation ===")
    s2 = eval_grid(ref, extras, X, Xt, y)
    print(s2.to_string(index=False))

    oof_ref, _ = fit_oof(X, Xt, y, ref, 42)
    r = per_target_rmse(y, oof_ref)
    print(f"\nRef seed42: IC50={r[0]:.0f} CC50={r[1]:.0f} SI={r[2]:.0f} mean={competition_score(y,oof_ref):.2f}")
    if len(s1) and s1.iloc[0]["delta_mean"] < -0.05:
        best = s1.iloc[0]
        print(f"Best grid: {best['candidate']} delta={best['delta_mean']:+.2f} SI_RMSE={best['si_rmse']:.0f}")
    print(f"Время: {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
