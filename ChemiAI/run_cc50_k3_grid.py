"""Узкая сетка вокруг cc50_k3 + CatBoost blend."""
from run_local_signal_search import (
    PipelineConfig,
    competition_score,
    ensure_data,
    fit_oof,
    load_data,
)

SEEDS = [42, 2024, 7]


def eval_cfg(cfg: PipelineConfig) -> tuple[float, int]:
    X_train, X_test, y_train, _ = load_data()
    deltas = []
    for seed in SEEDS:
        oof_b, _ = fit_oof(X_train, X_test, y_train, PipelineConfig(), random_state=seed)
        oof_c, _ = fit_oof(X_train, X_test, y_train, cfg, random_state=seed)
        deltas.append(competition_score(y_train, oof_c) - competition_score(y_train, oof_b))
    return sum(deltas) / len(deltas), sum(1 for d in deltas if d < -0.05)


def main() -> None:
    ensure_data()
    grid = []
    for k in [3, 4]:
        for tw in [0.55, 0.60, 0.65]:
            for cw in [0.10, 0.12, 0.15, 0.18, 0.20]:
                cfg = PipelineConfig(cc50_trans_k=k, cc50_blend_w=tw, cc50_cat_w=cw)
                mean_d, wins = eval_cfg(cfg)
                grid.append((mean_d, wins, k, tw, cw))

    grid.sort()
    print("k  tw   cw   mean_d  wins")
    for mean_d, wins, k, tw, cw in grid[:12]:
        print(f"{k}  {tw:.2f} {cw:.2f}  {mean_d:+7.2f}  {wins}/3")


if __name__ == "__main__":
    main()
