"""Curation, ordination and association testing.

The steps are the ones a metabolomics association analysis usually runs
through, kept deliberately small so each is readable in one sitting:

1. curation  -- drop sparse features, drop features that are unstable in the
   pooled QC samples, impute, log-transform and normalise;
2. PCA       -- an unsupervised look at what dominates the variance;
3. associations -- one linear model per feature per outcome, adjusted for
   age, sex and BMI, with Benjamini-Hochberg control across features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .simulate import OUTCOMES, Study


@dataclass(frozen=True)
class CurationResult:
    abundance: pd.DataFrame  # curated, log2, per-sample median normalised
    samples: pd.DataFrame  # study samples only
    kept: pd.Index
    dropped_sparse: pd.Index
    dropped_unstable: pd.Index
    qc_rsd: pd.Series
    n_imputed: int

    @property
    def summary(self) -> dict[str, int | float]:
        return {
            "features_in": len(self.kept)
            + len(self.dropped_sparse)
            + len(self.dropped_unstable),
            "features_kept": len(self.kept),
            "dropped_sparse": len(self.dropped_sparse),
            "dropped_unstable": len(self.dropped_unstable),
            "samples": len(self.samples),
            "values_imputed": self.n_imputed,
            "median_qc_rsd": float(self.qc_rsd.reindex(self.kept).median()),
        }


@dataclass(frozen=True)
class PCAResult:
    scores: pd.DataFrame
    explained: np.ndarray  # proportion of variance per component


def curate(
    study: Study,
    min_detection: float = 0.70,
    max_qc_rsd: float = 0.30,
) -> CurationResult:
    """Filter, impute, log-transform and normalise the feature table.

    Features must be detected in at least `min_detection` of study samples and
    have a pooled-QC relative standard deviation at or below `max_qc_rsd`.
    Remaining missing values are filled with half the feature minimum, the
    usual stand-in for "below the limit of detection".
    """
    abundance = study.abundance
    is_qc = (study.samples["sample_type"] == "qc").to_numpy()
    study_rows = abundance.loc[~is_qc]
    qc_rows = abundance.loc[is_qc]

    detection = study_rows.notna().mean(axis=0)
    sparse = detection.index[detection < min_detection]

    qc_rsd = (qc_rows.std(axis=0) / qc_rows.mean(axis=0)).fillna(np.inf)
    unstable = qc_rsd.index[(qc_rsd > max_qc_rsd) & (~qc_rsd.index.isin(sparse))]

    kept = abundance.columns.difference(sparse).difference(unstable)
    curated = study_rows[kept]

    n_imputed = int(curated.isna().to_numpy().sum())
    curated = curated.fillna(curated.min(axis=0) / 2.0)

    curated = np.log2(curated)
    # Per-sample median centring: removes overall intensity differences
    # between injections without touching between-feature structure.
    curated = curated.sub(curated.median(axis=1), axis=0)

    return CurationResult(
        abundance=curated,
        samples=study.samples.loc[~is_qc],
        kept=kept,
        dropped_sparse=sparse,
        dropped_unstable=unstable,
        qc_rsd=qc_rsd,
        n_imputed=n_imputed,
    )


def pca(curated: CurationResult, n_components: int = 4) -> PCAResult:
    """Principal components of the curated table, via SVD on centred data."""
    x = curated.abundance.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    k = min(n_components, s.size)
    scores = pd.DataFrame(
        u[:, :k] * s[:k],
        index=curated.abundance.index,
        columns=[f"PC{i + 1}" for i in range(k)],
    )
    explained = (s**2 / np.sum(s**2))[:k]
    return PCAResult(scores=scores, explained=explained)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Return BH-adjusted p-values (q-values) for a vector of p-values."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # Enforce monotonicity from the largest p-value down.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = np.clip(ranked, 0, 1)
    return q


def _design(samples: pd.DataFrame, outcome: str) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix [intercept, outcome, age, sex, bmi] and the row mask."""
    cols = [outcome, "age", "bmi"]
    frame = samples[cols + ["sex"]].copy()
    frame["sex_m"] = (frame["sex"] == "M").astype(float)
    frame = frame.drop(columns="sex")
    ok = frame.notna().all(axis=1).to_numpy()
    x = frame.loc[ok].to_numpy(dtype=float)
    x = np.column_stack([np.ones(x.shape[0]), x])
    return x, ok


def associate(
    curated: CurationResult, outcome: str, alpha: float = 0.05
) -> pd.DataFrame:
    """Fit one OLS model per feature for a single outcome.

    Each model is `feature ~ outcome + age + sex + bmi`; the reported estimate
    is the outcome coefficient, standardised so that it reads as log2 change
    in abundance per standard deviation of the outcome.
    """
    if outcome not in OUTCOMES:
        raise KeyError(f"unknown outcome {outcome!r}; expected one of {list(OUTCOMES)}")

    samples = curated.samples.copy()
    z = samples[outcome].astype(float)
    samples[outcome] = (z - z.mean()) / z.std()

    x, ok = _design(samples, outcome)
    y = curated.abundance.loc[ok].to_numpy(dtype=float)

    n, p = x.shape
    if n <= p:
        raise ValueError(f"not enough samples ({n}) for {p} model terms")

    # One least-squares solve for all features at once.
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    dof = n - p
    sigma2 = (resid**2).sum(axis=0) / dof
    se = np.sqrt(np.outer(np.diag(xtx_inv), sigma2))

    idx = 1  # the outcome column
    est = beta[idx]
    err = se[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(err > 0, est / err, np.nan)
    pval = 2 * stats.t.sf(np.abs(tstat), dof)

    out = pd.DataFrame(
        {
            "estimate": est,
            "std_error": err,
            "t": tstat,
            "p_value": pval,
            "q_value": benjamini_hochberg(pval),
            "n": dof + p,
        },
        index=curated.abundance.columns,
    )
    out["significant"] = out["q_value"] <= alpha
    return out.sort_values("p_value")


def run_all(
    study: Study, alpha: float = 0.05
) -> tuple[CurationResult, PCAResult, dict[str, pd.DataFrame]]:
    """Curate, run PCA, and test every outcome. Returns all three results."""
    curated = curate(study)
    components = pca(curated)
    results = {name: associate(curated, name, alpha) for name in OUTCOMES}
    return curated, components, results
