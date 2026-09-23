"""Command line entry point: simulate, analyse, render."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_all
from .report import build
from .simulate import simulate_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omicsdash",
        description="Generate a self-contained metabolomics dashboard from "
                    "synthetic data.",
    )
    parser.add_argument("-o", "--out", type=Path,
                        default=Path("dashboard.html"),
                        help="output HTML file (default: dashboard.html)")
    parser.add_argument("--subjects", type=int, default=60,
                        help="number of simulated subjects (default: 60)")
    parser.add_argument("--timepoints", type=int, default=2,
                        help="samples per subject (default: 2)")
    parser.add_argument("--features", type=int, default=400,
                        help="number of features (default: 400)")
    parser.add_argument("--planted", type=int, default=12,
                        help="true associations to plant (default: 12)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="FDR threshold (default: 0.05)")
    parser.add_argument("--seed", type=int, default=20260923,
                        help="random seed (default: 20260923)")
    args = parser.parse_args(argv)

    study = simulate_study(
        n_subjects=args.subjects,
        n_timepoints=args.timepoints,
        n_features=args.features,
        n_planted=args.planted,
        seed=args.seed,
    )
    curated, components, results = run_all(study, alpha=args.alpha)
    path = build(study, curated, components, results, args.out, alpha=args.alpha)

    n_sig = sum(int(df["significant"].sum()) for df in results.values())
    print(f"{study.n_samples} samples, {len(curated.kept)} features kept, "
          f"{n_sig} associations at FDR {args.alpha:g}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
