"""OOF для ratio_lgb w=52/55/58 (ref ratio_lgb50 public 269.12)."""
from phase_k_fe import engineer_features
from run_local_signal_search import clear_fold_cache, competition_score, ensure_data, load_data
from run_phase_b_si_ic50 import load_ext_cols
from run_phase_e_structural import feature_blocks
from run_phase_i_kaggle_ideas import blend_ic50_full, fit_phase_h
from run_phase_j import FULL_CB_W, blend_ic50_lgb

REF_W = 0.50

ensure_data()
X, Xt, y, cols = load_data()
ext = load_ext_cols(cols)
blocks = feature_blocks(cols)
Xh = engineer_features(X, ratios=True)
Xth = engineer_features(Xt, ratios=True)

ref = None
for w in [REF_W, 0.52, 0.55, 0.58]:
    clear_fold_cache()
    oof, test = fit_phase_h(X, Xt, y, ext, blocks, 42)
    oof, test = blend_ic50_full(X, Xt, y, oof, test, FULL_CB_W, 42)
    oof, _ = blend_ic50_lgb(Xh, Xth, y, oof, test, w, 42)
    sc = competition_score(y, oof)
    tag = f"ratio_lgb{int(round(w * 100))}"
    if w == REF_W:
        ref = sc
        print(f"ref {tag}: OOF={sc:.2f}")
    else:
        print(f"{tag}: OOF={sc:.2f} delta={sc - ref:+.2f}")
