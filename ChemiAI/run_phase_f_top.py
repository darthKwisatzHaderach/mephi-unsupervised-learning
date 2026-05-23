"""Топ Phase F + batch submission generation."""
import argparse

import pandas as pd

from make_submission_phase_f import apply_si_head


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--min-wins", type=int, default=2)
    p.add_argument("--generate", action="store_true")
    args = p.parse_args()

    try:
        df = pd.read_csv("phase_f_si_all.csv")
    except FileNotFoundError:
        print("Сначала: python run_phase_f_si_search.py")
        return

    filt = df[(df["delta_mean"] < -0.02) & (df["wins"] >= args.min_wins)]
    top = filt.head(args.top) if len(filt) else df.head(args.top)
    print("=== TOP candidates for submission ===")
    print(top[["name", "section", "delta_mean", "wins", "si_rmse", "oof_mean"]].to_string(index=False))

    if args.generate:
        from run_local_signal_search import DATA_DIR, SUBMISSION_COLS
        import numpy as np

        for name in top["name"]:
            out = f"submission_phase_f_{name}.csv"
            pred = apply_si_head(name, 42)
            sub = pd.read_csv(DATA_DIR / "sample_submission.csv").copy()
            sub[SUBMISSION_COLS] = pred
            sub.to_csv(out, index=False)
            print(f"  -> {out}")


if __name__ == "__main__":
    main()
