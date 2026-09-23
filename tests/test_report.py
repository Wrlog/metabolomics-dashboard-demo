"""Tests for the rendered dashboard.

The contract for the output is that it is a single file that works offline,
so the tests assert exactly that: no external references, both themes
present, and the content actually rendered. The interactive controls are
checked structurally - every control the inline script binds to must exist,
and every table row must carry the data attributes the filters read.
"""

from __future__ import annotations

import re

import pytest

from omicsdash.cli import main
from omicsdash.pipeline import run_all
from omicsdash.report import build
from omicsdash.simulate import OUTCOMES, simulate_study


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    study = simulate_study(n_subjects=40, n_features=150, n_planted=6, seed=3)
    curated, components, results = run_all(study)
    out = tmp_path_factory.mktemp("report") / "dashboard.html"
    build(study, curated, components, results, out)
    return out.read_text(encoding="utf-8")


def test_report_is_self_contained(rendered):
    """Nothing is fetched at open time: no external URLs, no src on <script>."""
    external = re.findall(r'(?:src|href)="(?!data:|#)([^"]+)"', rendered)
    assert all(u.startswith("https://github.com/") for u in external), external
    assert not re.search(r"<script[^>]+src=", rendered)
    assert not re.search(r"<link[^>]+href=", rendered)


def test_every_figure_is_embedded_twice(rendered):
    """One light and one dark PNG per figure, so dark mode is not an inversion."""
    light = rendered.count('class="light-only"')
    dark = rendered.count('class="dark-only"')
    assert light == dark > 0
    assert rendered.count("data:image/png;base64,") == light + dark


def test_report_states_the_data_is_synthetic(rendered):
    assert "Synthetic data" in rendered
    assert "simulate" in rendered


def test_report_renders_all_sections(rendered):
    for heading in ("Data curation", "annotation levels",
                    "Principal component analysis", "Planted effects"):
        assert heading in rendered


def test_no_unrendered_placeholders(rendered):
    """Check the prose and tables, not the base64 blobs, which contain anything."""
    text = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", rendered)
    body = text.split("</style>", 1)[1].split("<script>", 1)[0]
    assert "nan" not in body.lower()
    assert "None" not in body
    assert "{" not in body and "}" not in body  # no unformatted f-string braces


# --- interactive controls ---------------------------------------------------

CONTROL_IDS = ("f-outcome", "f-search", "f-level", "f-sort", "f-sig",
               "f-reset", "f-theme")


def test_every_control_exists(rendered):
    for control_id in CONTROL_IDS:
        assert f'id="{control_id}"' in rendered, control_id


def test_script_only_binds_controls_that_exist(rendered):
    """Guard against a renamed control silently breaking a filter."""
    script = rendered.split("<script>", 1)[1]
    for control_id in re.findall(r"\$\('([^']+)'\)", script):
        assert f'id="{control_id}"' in rendered, control_id


def test_every_control_is_labelled(rendered):
    """Each input is reachable by label, and the buttons carry text."""
    for control_id in ("f-outcome", "f-search", "f-level", "f-sort", "f-sig"):
        assert f'for="{control_id}"' in rendered, control_id


def test_outcome_selector_offers_every_outcome(rendered):
    for outcome in OUTCOMES:
        assert f'<option value="{outcome}">' in rendered
    assert '<option value="all">' in rendered


def test_each_outcome_has_a_filterable_section_and_table(rendered):
    for outcome in OUTCOMES:
        assert f'<section data-outcome="{outcome}">' in rendered
        assert f'data-outcome="{outcome}"' in rendered
        assert f'id="count-{outcome}"' in rendered


def test_rows_carry_the_attributes_the_filters_read(rendered):
    rows = re.findall(r"<tr data-p=[^>]*>", rendered)
    assert rows, "no result rows rendered"
    for attr in ("data-p", "data-effect", "data-mz", "data-rt",
                 "data-level", "data-sig", "data-search"):
        assert all(attr in row for row in rows), attr


def test_tables_hold_every_feature_not_a_top_slice(rendered):
    """Search is only useful if the rows are actually present."""
    study = simulate_study(n_subjects=40, n_features=150, n_planted=6, seed=3)
    curated, _, _ = run_all(study)
    rows = len(re.findall(r"<tr data-p=", rendered))
    assert rows == len(curated.kept) * len(OUTCOMES)


def test_nothing_starts_hidden(rendered):
    """With JavaScript off the page must still show everything."""
    assert "<section data-outcome" in rendered
    assert not re.search(r"<section[^>]*\shidden", rendered)
    assert not re.search(r"<tr[^>]*\shidden", rendered)


def test_cli_writes_a_file(tmp_path, capsys):
    out = tmp_path / "cli.html"
    code = main(["--out", str(out), "--subjects", "25", "--features", "80",
                 "--planted", "4"])
    assert code == 0
    assert out.exists() and out.stat().st_size > 50_000
    assert "wrote" in capsys.readouterr().out
