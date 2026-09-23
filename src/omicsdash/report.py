"""Render the analysis into one self-contained HTML file.

Self-contained is the point: figures are embedded as base64 PNGs and the CSS
is inline, so the output is a single file that can be emailed, attached to a
ticket or opened from a USB stick with no server and no network. Each figure
is embedded twice, light and dark, and CSS picks - so the dark version is a
set of marks chosen for a dark surface rather than an inverted image.
"""

from __future__ import annotations

import base64
import datetime as dt
import html
from pathlib import Path

import pandas as pd

from . import figures
from .pipeline import CurationResult, PCAResult
from .simulate import OUTCOMES, Study
from .theme import DARK, LIGHT

CSS = """
:root {
  --surface: #fcfcfb; --sunken: #f4f3f0; --border: #e2e1dc;
  --ink: #0b0b0b; --ink-2: #52514e; --ink-3: #78766f;
  --accent: #2a78d6; --warn: #eb6834;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --surface: #1a1a19; --sunken: #232322; --border: #3a3a37;
    --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #95948b;
    --accent: #3987e5; --warn: #d95926;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --surface: #1a1a19; --sunken: #232322; --border: #3a3a37;
  --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #95948b;
  --accent: #3987e5; --warn: #d95926;
  color-scheme: dark;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--surface); color: var(--ink);
  font: 15px/1.6 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1040px; margin: 0 auto; padding: 40px 16px 80px; }
header {
  border-bottom: 1px solid var(--border); padding-bottom: 20px;
  margin-bottom: 8px;
}
h1 { font-size: 1.7rem; line-height: 1.25; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 1.15rem; margin: 44px 0 4px; letter-spacing: -0.005em; }
h3 {
  font-size: 0.95rem; margin: 26px 0 4px; color: var(--ink-2); font-weight: 600;
}
p, li { color: var(--ink-2); }
.lede { color: var(--ink-2); margin: 0; }
.note {
  background: var(--sunken); border: 1px solid var(--border);
  border-left: 3px solid var(--warn); border-radius: 8px;
  padding: 12px 16px; margin: 20px 0; font-size: 0.92rem;
}
.note strong { color: var(--ink); }
.tiles { display: flex; flex-wrap: wrap; gap: 10px; margin: 18px 0 6px; }
.tile {
  flex: 1 1 140px; background: var(--sunken); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 14px;
}
.tile .v {
  font-size: 1.5rem; font-weight: 650; color: var(--ink); letter-spacing: -0.02em;
}
.tile .k {
  font-size: 0.78rem; color: var(--ink-3);
  text-transform: uppercase; letter-spacing: 0.04em;
}
figure { margin: 18px 0 6px; }
figure img { width: 100%; height: auto; display: block; border-radius: 8px; }
figcaption { font-size: 0.86rem; color: var(--ink-3); margin-top: 8px; }
img.dark-only { display: none; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) img.light-only { display: none; }
  :root:not([data-theme="light"]) img.dark-only { display: block; }
}
:root[data-theme="dark"] img.light-only { display: none; }
:root[data-theme="dark"] img.dark-only { display: block; }
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; margin: 14px 0; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--border); }
th:first-child, td:first-child { text-align: left; }
th {
  color: var(--ink-3); font-weight: 600; text-transform: uppercase;
  font-size: 0.74rem; letter-spacing: 0.04em;
}
td { color: var(--ink-2); font-variant-numeric: tabular-nums; }
td.hit { color: var(--ink); font-weight: 600; }
.up { color: var(--warn); } .down { color: var(--accent); }
footer {
  margin-top: 56px; padding-top: 18px; border-top: 1px solid var(--border);
  font-size: 0.85rem; color: var(--ink-3);
}
code {
  background: var(--sunken); padding: 1px 5px; border-radius: 4px;
  font-size: 0.88em;
}
@media (max-width: 640px) {
  .wrap { padding: 24px 16px 60px; }
  h1 { font-size: 1.4rem; }
}
"""


def _fig(light: bytes, dark: bytes, alt: str, caption: str) -> str:
    def b64(data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")

    safe = html.escape(alt)
    return "\n".join([
        "<figure>",
        f'  <img class="light-only" alt="{safe}"',
        f'       src="data:image/png;base64,{b64(light)}">',
        f'  <img class="dark-only" alt="{safe}"',
        f'       src="data:image/png;base64,{b64(dark)}">',
        f"  <figcaption>{caption}</figcaption>",
        "</figure>",
    ])


def _tiles(items: list[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div class="tile"><div class="v">{html.escape(v)}</div>'
        f'<div class="k">{html.escape(k)}</div></div>'
        for v, k in items
    )
    return f'<div class="tiles">{cells}</div>'


def _hits_table(assoc: pd.DataFrame, features: pd.DataFrame, top_n: int = 10) -> str:
    top = assoc.head(top_n)
    rows = []
    for fid, row in top.iterrows():
        meta = features.loc[fid]
        direction = "up" if row["estimate"] > 0 else "down"
        arrow = "+" if row["estimate"] > 0 else "−"
        flag = ' class="hit"' if row["significant"] else ""
        rows.append(
            f"<tr><td{flag}>{fid}</td>"
            f"<td>{meta['mz']:.4f}</td>"
            f"<td>{meta['rt_min']:.2f}</td>"
            f"<td>L{int(meta['annotation_level'])}</td>"
            f'<td class="{direction}">{arrow}{abs(row["estimate"]):.3f}</td>'
            f"<td>{row['std_error']:.3f}</td>"
            f"<td>{row['p_value']:.2e}</td>"
            f"<td>{row['q_value']:.3f}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Feature</th><th>m/z</th><th>RT (min)</th>"
        "<th>Level</th><th>Estimate</th><th>SE</th><th>p</th><th>q</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _annotation_table(features: pd.DataFrame, kept: pd.Index) -> str:
    counts = features.loc[kept, "annotation_level"].value_counts().sort_index()
    total = int(counts.sum())
    names = {
        1: "Confirmed against a standard",
        2: "Putative, spectral match",
        3: "Putative class",
        4: "Unknown",
    }
    rows = "".join(
        f"<tr><td>Level {lvl} — {names[lvl]}</td><td>{int(n)}</td>"
        f"<td>{n / total * 100:.1f}%</td></tr>"
        for lvl, n in counts.items()
    )
    return (
        "<table><thead><tr><th>Annotation level</th><th>Features</th>"
        f"<th>Share</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def build(
    study: Study,
    curated: CurationResult,
    components: PCAResult,
    results: dict[str, pd.DataFrame],
    out_path: Path,
    alpha: float = 0.05,
) -> Path:
    """Write the dashboard and return the path it was written to."""
    s = curated.summary
    generated = dt.datetime.now().strftime("%d %B %Y, %H:%M")

    total_hits = sum(int(df["significant"].sum()) for df in results.values())

    parts: list[str] = []
    parts.append(f"""<header>
  <h1>Metabolomics association dashboard</h1>
  <p class="lede">Synthetic demonstration study · {s["samples"]} samples ·
  {s["features_kept"]} curated features · {len(OUTCOMES)} outcomes</p>
</header>
<div class="note">
  <strong>Synthetic data.</strong> Every number on this page comes from a seeded
  simulation in <code>omicsdash.simulate</code>. No real study, subject or
  measurement is represented here, and none is needed to reproduce it.
</div>""")

    # 1. Curation
    parts.append("<h2>1 · Data curation</h2>")
    parts.append(
        "<p>Features are kept when they are detected in at least 70% of study "
        "samples and vary by no more than 30% RSD across the pooled QC "
        "injections. Remaining gaps are filled at half the feature minimum, "
        "then the table is log2-transformed and median-centred per sample.</p>"
    )
    parts.append(_tiles([
        (f"{s['features_in']:,}", "features in"),
        (f"{s['features_kept']:,}", "features kept"),
        (f"{s['dropped_sparse']:,}", "dropped: sparse"),
        (f"{s['dropped_unstable']:,}", "dropped: QC RSD"),
        (f"{s['median_qc_rsd'] * 100:.1f}%", "median QC RSD"),
        (f"{s['values_imputed']:,}", "values imputed"),
    ]))
    parts.append(_fig(
        figures.qc_drift(study, LIGHT), figures.qc_drift(study, DARK),
        "Median intensity against injection order, with pooled QC samples marked",
        "Median log2 intensity by injection order. Pooled QC samples (diamonds) "
        "carry the analytical drift without the biology, so their trend line is "
        "the drift estimate.",
    ))

    # 2. Feature set
    parts.append("<h2>2 · Feature set and annotation levels</h2>")
    parts.append(
        "<p>Annotation level records how confidently a feature is identified. "
        "Most untargeted features stay unknown, which is expected and is the "
        "reason association results are reported at feature level rather than "
        "as named metabolites.</p>"
    )
    parts.append(_annotation_table(study.features, curated.kept))

    # 3. PCA
    parts.append("<h2>3 · Principal component analysis</h2>")
    parts.append(
        "<p>An unsupervised check on what dominates the curated table. Batch "
        "separation along an early component is a sign that acquisition, not "
        "biology, is the largest source of variance.</p>"
    )
    parts.append(_fig(
        figures.pca_scores(components, curated, LIGHT),
        figures.pca_scores(components, curated, DARK),
        "PC1 against PC2, coloured by acquisition batch",
        "Sample scores on the first two components, split by acquisition batch.",
    ))
    parts.append(_fig(
        figures.scree(components, LIGHT), figures.scree(components, DARK),
        "Variance explained by each principal component",
        "Variance explained by each retained component.",
    ))

    # 4+. One section per outcome
    for i, (outcome, label) in enumerate(OUTCOMES.items(), start=4):
        assoc = results[outcome]
        n_sig = int(assoc["significant"].sum())
        parts.append(f"<h2>{i} · {html.escape(label)}</h2>")
        parts.append(
            f"<p>One linear model per feature: "
            f"<code>feature ~ {outcome} + age + sex + bmi</code>, with "
            f"Benjamini-Hochberg control across the "
            f"{s['features_kept']:,} curated features. Estimates read as log2 "
            f"abundance change per standard deviation of the outcome.</p>"
        )
        parts.append(_tiles([
            (f"{n_sig}", f"features at FDR {alpha:g}"),
            (f"{assoc['p_value'].min():.1e}", "smallest p"),
            (f"{int(assoc['n'].iloc[0])}", "samples modelled"),
        ]))
        parts.append(_fig(
            figures.volcano(assoc, outcome, LIGHT, alpha),
            figures.volcano(assoc, outcome, DARK, alpha),
            f"Volcano plot of feature associations with {label}",
            "Effect size against evidence. Colour encodes direction only; the "
            "dashed line is the FDR boundary.",
        ))
        if n_sig:
            parts.append(_fig(
                figures.forest(assoc, outcome, LIGHT),
                figures.forest(assoc, outcome, DARK),
                f"Effect estimates with 95% intervals for {label}",
                "Strongest associations with 95% confidence intervals.",
            ))
        parts.append("<h3>Top features</h3>")
        parts.append(_hits_table(assoc, study.features))

    # Ground truth, which is only knowable because the data is simulated.
    parts.append("<h2>Appendix · Planted effects</h2>")
    parts.append(
        "<p>The simulator writes a known set of associations into the data. "
        "Listing them here turns the dashboard into something checkable: the "
        "pipeline should recover these and not much else.</p>"
    )
    def _verdict(effect) -> str:
        table = results[effect.outcome]
        hits = table.index[table["significant"]]
        return "recovered" if effect.feature_id in hits else "missed"

    rows = "".join(
        f"<tr><td>{e.feature_id}</td>"
        f"<td>{html.escape(OUTCOMES[e.outcome])}</td>"
        f"<td>{e.beta:+.3f}</td><td>{_verdict(e)}</td></tr>"
        for e in study.planted
    )
    parts.append(
        "<table><thead><tr><th>Feature</th><th>Outcome</th>"
        "<th>True effect</th><th>Result</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )

    parts.append(f"""<footer>
  Generated {generated} by
  <a href="https://github.com/Wrlog/metabolomics-dashboard-demo">omicsdash</a>.
  {total_hits} associations at FDR {alpha:g} across {len(OUTCOMES)} outcomes.
  Synthetic data throughout.
</footer>""")

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Metabolomics association dashboard</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
{"".join(parts)}
</div>
</body>
</html>"""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path
