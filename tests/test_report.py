"""Tests for the rendered dashboard.

The contract for the output is that it is a single file that works offline,
so the tests assert exactly that: no external references, both themes
present, and the content actually rendered.
"""

from __future__ import annotations

import re

import pytest

from omicsdash.cli import main
from omicsdash.pipeline import run_all
from omicsdash.report import build
from omicsdash.simulate import simulate_study


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    study = simulate_study(n_subjects=40, n_features=150, n_planted=6, seed=3)
    curated, components, results = run_all(study)
    out = tmp_path_factory.mktemp("report") / "dashboard.html"
    build(study, curated, components, results, out)
    return out.read_text(encoding="utf-8")


def test_report_is_self_contained(rendered):
    """No network references: every image is inline, no external CSS or JS."""
    assert "<script" not in rendered
    assert "<link" not in rendered
    external = re.findall(r'(?:src|href)="(?!data:|#)([^"]+)"', rendered)
    assert all(u.startswith("https://github.com/") for u in external), external


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
    body = text.split("</style>", 1)[1]
    assert "nan" not in body.lower()
    assert "None" not in body
    assert "{" not in body and "}" not in body  # no unformatted f-string braces


def test_cli_writes_a_file(tmp_path, capsys):
    out = tmp_path / "cli.html"
    code = main(["--out", str(out), "--subjects", "25", "--features", "80",
                 "--planted", "4"])
    assert code == 0
    assert out.exists() and out.stat().st_size > 50_000
    assert "wrote" in capsys.readouterr().out
