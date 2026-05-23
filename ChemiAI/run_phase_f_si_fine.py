"""Узкая сетка вокруг si_trans_k5_w30 (best OOF -1.21)."""
import time

import pandas as pd

from run_local_signal_search import clear_fold_cache, competition_score, ensure_data, load_data, per_target_rmse
from run_phase_b_si_ic50 import SEEDS, load_ext_cols
from run_phase_e_structural import blend_extra_head, feature_blocks
from run_phase_f_si_search import CachedBase, best_base_cfg, blend_si_transductive_head, eval_on_base, fit_best_base, summarize

BEST_FR_W = 0.42


def main() -> None:
    ensure_data()
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)
    t0 = time.perf_counter()

    bases = {}
    ref_scores = {}
    for seed in SEEDS:
        bases[seed] = fit_best_base(X, Xt, y, ext, blocks, seed)
        ref_scores[seed] = competition_score(y, bases[seed].oof)

    rows = []
    for k in [4, 5, 6, 7]:
        for w in [0.22, 0.25, 0.28, 0.30, 0.32, 0.35]:
            name = f"si_trans_k{k}_w{int(w*100)}"
            for seed in SEEDS:
                base = bases[seed]
                oof, _ = blend_si_transductive_head(
                    X, Xt, y, base.oof, base.test, k, w, seed,
                )
                rows.append(eval_on_base(name, base, oof, ref_scores, seed, y))

    df = pd.DataFrame(rows)
    sm = summarize(df)
    sm.to_csv("phase_f_si_fine.csv", index=False)
    print(sm.head(15).to_string(index=False))
    print(f"\nВремя: {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
