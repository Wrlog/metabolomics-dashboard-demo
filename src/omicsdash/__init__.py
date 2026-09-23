"""A small, dependency-light example of a reproducible analysis dashboard.

The package simulates a metabolomics study, runs a short association
pipeline over it, and renders the result as one self-contained HTML file.
All data is synthetic; see `omicsdash.simulate`.
"""

from .pipeline import associate, curate, pca, run_all
from .report import build as build_report
from .simulate import OUTCOMES, Study, simulate_study

__version__ = "0.1.0"

__all__ = [
    "OUTCOMES",
    "Study",
    "associate",
    "build_report",
    "curate",
    "pca",
    "run_all",
    "simulate_study",
]
