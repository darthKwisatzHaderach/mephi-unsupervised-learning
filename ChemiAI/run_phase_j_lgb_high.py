"""Быстрый OOF только для новых LGB w (30, 32, 35)."""
from run_local_signal_search import clear_fold_cache, competition_score, ensure_data, load_data
from run_phase_b_si_ic50 import load_ext_cols
from run_phase_e_structural import feature_blocks
from run_phase_j import blend_ic50_lgb, fit_phase_i

for w in [0.52, 0.55, 0.58]:
    ensure_data()
    X, Xt, y, cols = load_data()
    ext = load_ext_cols(cols)
    blocks = feature_blocks(cols)
    clear_fold_cache()
    oof, test = fit_phase_i(X, Xt, y, ext, blocks, 42)
    ref = competition_score(y, oof)
    clear_fold_cache()
    oof, test = fit_phase_i(X, Xt, y, ext, blocks, 42)
    oof, _ = blend_ic50_lgb(X, Xt, y, oof, test, w, 42)
    sc = competition_score(y, oof)
    print(f"lgb_ic{int(w*100)}: OOF={sc:.2f} delta={sc-ref:+.2f}")
