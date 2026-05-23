"""
Фаза C: поиск крупного сигнала (цель LB 264-270, текущий public 279.92).
Ref: ext_k3_cat15 + ic50_cat_w=0.40
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

REF_W = 0.40


def frozen_best(ext_cols: list[str], **kw) -> PipelineConfig:
    base = dict(ic50_cat_w=REF_W)
    base.update(kw)
    return frozen_cc50(ext_cols, **base)


def eval_grid(
    ref_cfg: PipelineConfig,
    candidates: list[tuple[str, PipelineConfig]],
    X, Xt, y,
) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        clear_fold_cache()
        oof_ref, _ = fit_oof(X, Xt, y, ref_cfg, random_state=seed)
        ref_s = competition_score(y, oof_ref)
        for name, cfg in candidates:
            oof, _ = fit_oof(X, Xt, y, cfg, random_state=seed)
            s = competition_score(y, oof)
            rmse = per_target_rmse(y, oof)
            rows.append({
                "candidate": name,
                "seed": seed,
                "oof": s,
                "delta": s - ref_s,
                "ic50": rmse[0],
                "cc50": rmse[1],
                "si": rmse[2],
            })
    df = pd.DataFrame(rows)
    return df.groupby("candidate").agg(
        oof_mean=("oof", "mean"),
        delta_mean=("delta", "mean"),
        wins=("delta", lambda s: int((s < -0.1).sum())),
        ic50=("ic50", "mean"),
        cc50=("cc50", "mean"),
        si=("si", "mean"),
    ).reset_index().sort_values("delta_mean")


def main() -> None:
    ensure_data()
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    ref = frozen_best(ext)
    t0 = time.perf_counter()

    # --- IC50 w продолжение ---
    ic50_w = [(f"ic50_w{int(w*100)}", frozen_best(ext, ic50_cat_w=w))
              for w in [0.42, 0.45, 0.50, 0.55, 0.60, 0.65]]
    print("=== IC50 cat_w (ref w40) ===")
    s1 = eval_grid(ref, ic50_w, X, Xt, y)
    print(s1.to_string(index=False))

    # --- CC50 при w40 ---
    cc50 = []
    for cw in [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
        cc50.append((f"cc50_cat{cw:.2f}", frozen_best(ext, cc50_cat_w=cw)))
    for k in [2, 4, 5, 7]:
        cc50.append((f"cc50_k{k}", frozen_best(ext, cc50_trans_k=k)))
    for tw in [0.50, 0.55, 0.65, 0.70]:
        cc50.append((f"cc50_tw{int(tw*100)}", frozen_best(ext, cc50_blend_w=tw)))
    print("\n=== CC50 (ref w40) ===")
    s2 = eval_grid(ref, cc50, X, Xt, y)
    print(s2.head(15).to_string(index=False))

    # --- комбо лучших направлений ---
    combos = [
        ("w45_cat12", frozen_best(ext, ic50_cat_w=0.45, cc50_cat_w=0.12)),
        ("w50_cat15", frozen_best(ext, ic50_cat_w=0.50, cc50_cat_w=0.15)),
        ("w45_k2", frozen_best(ext, ic50_cat_w=0.45, cc50_trans_k=2)),
        ("w50_k3_cat12", frozen_best(ext, ic50_cat_w=0.50, cc50_cat_w=0.12)),
        ("w55_cat15", frozen_best(ext, ic50_cat_w=0.55, cc50_cat_w=0.15)),
        ("w60_cat15", frozen_best(ext, ic50_cat_w=0.60, cc50_cat_w=0.15)),
        ("w50_tw55", frozen_best(ext, ic50_cat_w=0.50, cc50_blend_w=0.55)),
        ("w45_tw65", frozen_best(ext, ic50_cat_w=0.45, cc50_blend_w=0.65)),
    ]
    print("\n=== Combos ===")
    s3 = eval_grid(ref, combos, X, Xt, y)
    print(s3.to_string(index=False))

    # ref per-target OOF
    clear_fold_cache()
    oof_ref, _ = fit_oof(X, Xt, y, ref, random_state=42)
    rmse = per_target_rmse(y, oof_ref)
    print(f"\nRef w40 seed42 per-target RMSE: IC50={rmse[0]:.1f} CC50={rmse[1]:.1f} SI={rmse[2]:.1f} mean={competition_score(y,oof_ref):.2f}")
    print(f"Время: {time.perf_counter()-t0:.0f}s")

    pd.concat([s1, s2, s3]).drop_duplicates("candidate").sort_values("delta_mean").to_csv(
        "phase_c_search.csv", index=False
    )


if __name__ == "__main__":
    main()
