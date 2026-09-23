"""Synthetic metabolomics and phenotype data.

Everything this package analyses is generated here, from a seed. There is no
real study data anywhere in this repository, and none is required to run it.

The generator plants a known set of feature-outcome associations so that the
pipeline can be checked against ground truth: `simulate_study` returns the
planted effects alongside the data, and the tests assert that the analysis
recovers them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Phenotype domains the demo reports on, mirroring the kind of wearable
#: readouts a metabolomics study is often paired with.
OUTCOMES: dict[str, str] = {
    "activity_steps": "Daily step count",
    "sleep_efficiency": "Sleep efficiency (%)",
    "resting_hr": "Resting heart rate (bpm)",
}

#: Annotation confidence levels, following the Metabolomics Standards
#: Initiative convention: level 1 is a standard-confirmed identification,
#: level 4 an unknown feature.
ANNOTATION_LEVELS = (1, 2, 3, 4)


@dataclass(frozen=True)
class PlantedEffect:
    """A feature-outcome association deliberately written into the data."""

    feature_id: str
    outcome: str
    beta: float


@dataclass(frozen=True)
class Study:
    """A simulated study: sample metadata, feature table, and ground truth."""

    samples: pd.DataFrame
    features: pd.DataFrame
    abundance: pd.DataFrame
    planted: list[PlantedEffect] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_features(self) -> int:
        return len(self.features)


def _feature_table(rng: np.random.Generator, n_features: int) -> pd.DataFrame:
    """Build the feature metadata: m/z, retention time, annotation level."""
    level = rng.choice(
        ANNOTATION_LEVELS, size=n_features, p=[0.06, 0.14, 0.25, 0.55]
    )
    return pd.DataFrame(
        {
            "feature_id": [f"FT{i:04d}" for i in range(n_features)],
            "mz": np.round(rng.uniform(80, 900, n_features), 4),
            "rt_min": np.round(rng.uniform(0.4, 14.0, n_features), 2),
            "annotation_level": level,
        }
    ).set_index("feature_id")


def _sample_table(
    rng: np.random.Generator, n_subjects: int, n_timepoints: int
) -> pd.DataFrame:
    """Build subject metadata with repeated measures and QC samples."""
    subject_ids = [f"S{i:03d}" for i in range(n_subjects)]
    age = rng.normal(42, 12, n_subjects).clip(20, 75)
    sex = rng.choice(["F", "M"], size=n_subjects)
    bmi = rng.normal(25, 4, n_subjects).clip(17, 42)

    rows = []
    for i, sid in enumerate(subject_ids):
        for t in range(n_timepoints):
            rows.append(
                {
                    "sample_id": f"{sid}_T{t}",
                    "subject_id": sid,
                    "timepoint": t,
                    "sample_type": "study",
                    "age": round(float(age[i]), 1),
                    "sex": sex[i],
                    "bmi": round(float(bmi[i]), 1),
                }
            )

    # Pooled QC samples, injected every eighth position in the run order.
    n_qc = max(4, len(rows) // 8)
    for q in range(n_qc):
        rows.append(
            {
                "sample_id": f"QC{q:02d}",
                "subject_id": None,
                "timepoint": None,
                "sample_type": "qc",
                "age": np.nan,
                "sex": None,
                "bmi": np.nan,
            }
        )

    samples = pd.DataFrame(rows)
    # Randomise injection order, which the drift model below follows.
    order = rng.permutation(len(samples))
    samples["run_order"] = np.argsort(order) + 1
    return samples.sort_values("run_order").set_index("sample_id")


def _phenotypes(rng: np.random.Generator, samples: pd.DataFrame) -> pd.DataFrame:
    """Wearable-style outcomes, correlated with age and BMI as you would expect."""
    study = samples[samples["sample_type"] == "study"]
    n = len(study)
    age_z = (study["age"] - study["age"].mean()) / study["age"].std()
    bmi_z = (study["bmi"] - study["bmi"].mean()) / study["bmi"].std()

    pheno = pd.DataFrame(index=study.index)
    pheno["activity_steps"] = (
        8500 - 700 * bmi_z - 300 * age_z + rng.normal(0, 1800, n)
    ).clip(1200, 25000)
    pheno["sleep_efficiency"] = (
        87 - 1.4 * bmi_z - 0.8 * age_z + rng.normal(0, 4.0, n)
    ).clip(55, 99)
    pheno["resting_hr"] = (
        62 + 2.6 * bmi_z + 1.1 * age_z + rng.normal(0, 5.5, n)
    ).clip(40, 105)
    return pheno


def simulate_study(
    n_subjects: int = 60,
    n_timepoints: int = 2,
    n_features: int = 400,
    n_planted: int = 12,
    missing_rate: float = 0.04,
    seed: int = 20260923,
) -> Study:
    """Generate a complete synthetic study.

    The abundance matrix is built on a log2 scale from five additive parts:
    a per-feature baseline, batch and injection-order drift, biological
    variation, planted associations with the phenotypes, and analytical noise.

    Only the drift and the analytical noise reach the pooled QC samples, which
    is what makes them informative: a feature whose QC replicates disagree is
    unstable in the instrument, not variable in the population.
    """
    rng = np.random.default_rng(seed)

    features = _feature_table(rng, n_features)
    samples = _sample_table(rng, n_subjects, n_timepoints)
    pheno = _phenotypes(rng, samples)
    samples = samples.join(pheno)

    n_s, n_f = len(samples), len(features)
    baseline = rng.normal(16, 2.2, n_f)
    x = np.tile(baseline, (n_s, 1))

    # Injection-order drift: a slow per-feature trend across the run. A
    # minority of features drift hard enough to fail the QC filter later.
    progress = (samples["run_order"].to_numpy() / n_s)[:, None]
    drift_scale = np.where(rng.random(n_f) < 0.12, 0.90, 0.10)
    drift = rng.normal(0, 1.0, n_f)[None, :] * drift_scale[None, :] * progress
    x = x + drift

    # Two acquisition batches with a small offset.
    batch = (samples["run_order"].to_numpy() > n_s / 2).astype(float)
    x = x + batch[:, None] * rng.normal(0, 0.18, n_f)[None, :]

    is_study = (samples["sample_type"] == "study").to_numpy()

    # Biological variability, which QC samples do not have.
    x = x + rng.normal(0, 0.60, (n_s, n_f)) * is_study[:, None]

    # Planted biology, on study samples only.
    planted: list[PlantedEffect] = []
    outcome_names = list(OUTCOMES)
    chosen = rng.choice(n_f, size=n_planted, replace=False)
    for k, idx in enumerate(chosen):
        outcome = outcome_names[k % len(outcome_names)]
        beta = float(rng.uniform(0.45, 1.10) * rng.choice([-1.0, 1.0]))
        z = samples[outcome].to_numpy(dtype=float)
        z = (z - np.nanmean(z)) / np.nanstd(z)
        z = np.nan_to_num(z)
        x[:, idx] += beta * z * is_study
        planted.append(
            PlantedEffect(
                feature_id=str(features.index[idx]), outcome=outcome, beta=beta
            )
        )

    # Analytical noise, on every injection including the QC samples.
    x = x + rng.normal(0, 0.16, (n_s, n_f))

    abundance = pd.DataFrame(
        np.exp2(x), index=samples.index, columns=features.index
    )

    # Missing values, more likely in low-abundance features (as in practice).
    rank = abundance.mean(axis=0).rank(pct=True).to_numpy()
    p_missing = missing_rate * (1.8 - rank)[None, :]
    mask = rng.random(abundance.shape) < p_missing
    abundance = abundance.mask(mask)

    return Study(
        samples=samples, features=features, abundance=abundance, planted=planted
    )
