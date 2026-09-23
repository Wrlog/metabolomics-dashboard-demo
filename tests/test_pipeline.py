"""Tests for the analysis pipeline.

The simulator plants known associations, so these are not just smoke tests:
the pipeline is checked against ground truth for both power and false
positives, and the multiple-testing correction is checked against a case with
no signal at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from omicsdash.pipeline import associate, benjamini_hochberg, curate, pca, run_all
from omicsdash.simulate import OUTCOMES, simulate_study


@pytest.fixture(scope="module")
def study():
    return simulate_study(n_subjects=60, n_features=300, n_planted=9, seed=7)


def test_simulation_is_reproducible():
    a = simulate_study(n_subjects=10, n_features=40, n_planted=3, seed=123)
    b = simulate_study(n_subjects=10, n_features=40, n_planted=3, seed=123)
    assert a.abundance.equals(b.abundance)
    assert [e.feature_id for e in a.planted] == [e.feature_id for e in b.planted]


def test_simulation_differs_by_seed():
    a = simulate_study(n_subjects=10, n_features=40, seed=1)
    b = simulate_study(n_subjects=10, n_features=40, seed=2)
    assert not a.abundance.equals(b.abundance)


def test_curation_drops_features_and_keeps_only_study_samples(study):
    curated = curate(study)
    assert len(curated.kept) < study.n_features
    assert (curated.samples["sample_type"] == "study").all()
    assert not curated.abundance.isna().to_numpy().any()
    # Log2 and median-centred: every sample's median is ~0.
    assert np.allclose(curated.abundance.median(axis=1), 0, atol=1e-9)


def test_curation_thresholds_are_respected(study):
    curated = curate(study, min_detection=0.70, max_qc_rsd=0.30)
    assert (curated.qc_rsd.reindex(curated.kept) <= 0.30).all()
    detection = study.abundance.loc[
        study.samples["sample_type"] == "study", curated.kept
    ].notna().mean(axis=0)
    assert (detection >= 0.70).all()


def test_pca_explains_decreasing_variance(study):
    curated = curate(study)
    result = pca(curated, n_components=4)
    assert result.scores.shape == (len(curated.samples), 4)
    assert np.all(np.diff(result.explained) <= 0)
    assert 0 < result.explained.sum() <= 1.0


def test_benjamini_hochberg_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    q = benjamini_hochberg(p)
    assert q.min() >= 0 and q.max() <= 1
    assert np.all(q >= p - 1e-12)  # adjustment never decreases a p-value
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)  # monotone in p


def test_benjamini_hochberg_matches_known_case():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    q = benjamini_hochberg(p)
    expected = np.array([0.008, 0.032, 0.0672, 0.0672, 0.0672, 0.08, 0.0846, 0.205])
    assert np.allclose(q, expected, atol=1e-4)


def test_associations_recover_most_planted_effects(study):
    curated, _, results = run_all(study)
    kept = set(curated.kept)
    planted = [e for e in study.planted if e.feature_id in kept]
    assert planted, "curation removed every planted feature; check the fixture"

    recovered = [
        e for e in planted
        if bool(results[e.outcome].loc[e.feature_id, "significant"])
    ]
    assert len(recovered) >= 0.6 * len(planted)

    # Direction is recovered, not just presence.
    for e in recovered:
        estimate = results[e.outcome].loc[e.feature_id, "estimate"]
        assert np.sign(estimate) == np.sign(e.beta)


def test_false_positive_rate_is_controlled(study):
    curated, _, results = run_all(study)
    planted_ids = {e.feature_id for e in study.planted}
    for outcome, table in results.items():
        hits = set(table.index[table["significant"]])
        spurious = hits - planted_ids
        # At FDR 5% a handful of the several hundred nulls may slip through;
        # an order-of-magnitude break would mean the correction is broken.
        assert len(spurious) <= max(3, 0.05 * len(table)), outcome


def test_no_signal_gives_no_discoveries():
    flat = simulate_study(n_subjects=50, n_features=250, n_planted=0, seed=11)
    _, _, results = run_all(flat)
    for outcome, table in results.items():
        assert int(table["significant"].sum()) <= 2, outcome


def test_every_outcome_is_analysed(study):
    _, _, results = run_all(study)
    assert set(results) == set(OUTCOMES)


def test_unknown_outcome_is_rejected(study):
    curated = curate(study)
    with pytest.raises(KeyError):
        associate(curated, "not_an_outcome")
