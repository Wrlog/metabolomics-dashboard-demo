"""Render the analysis into one self-contained, interactive HTML file.

Self-contained is the point: figures are embedded as base64 PNGs, and the CSS
and the small amount of JavaScript are inline, so the output is a single file
that can be emailed, attached to a ticket or opened from a USB stick with no
server and no network. Each figure is embedded twice, light and dark, and CSS
picks - so the dark version is a set of marks chosen for a dark surface rather
than an inverted image.

The controls at the top filter and re-sort the result tables in the browser.
Figures are rendered images and cannot redraw, so the outcome selector shows
and hides whole sections rather than pretending to re-plot.
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
.wrap { max-width: 1040px; margin: 0 auto; padding: 32px 16px 80px; }
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

/* Controls: one row above the content, sticky so they stay reachable. */
.controls {
  position: sticky; top: 0; z-index: 10;
  display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: flex-end;
  background: var(--surface); border-bottom: 1px solid var(--border);
  padding: 12px 0; margin-bottom: 4px;
}
.control { display: flex; flex-direction: column; gap: 3px; }
.control > label {
  font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--ink-3); font-weight: 600;
}
.controls select, .controls input[type="search"] {
  font: inherit; font-size: 0.9rem; color: var(--ink);
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 7px; padding: 7px 10px; min-height: 38px;
}
.controls input[type="search"] { min-width: 210px; }
.controls select:focus-visible, .controls input:focus-visible,
.btn:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 2px;
}
.check {
  display: flex; align-items: center; gap: 7px; min-height: 38px;
  font-size: 0.9rem; color: var(--ink-2); cursor: pointer;
}
.check input { width: 16px; height: 16px; accent-color: var(--accent); }
.btn {
  font: inherit; font-size: 0.88rem; color: var(--ink-2); cursor: pointer;
  background: var(--sunken); border: 1px solid var(--border);
  border-radius: 7px; padding: 8px 13px; min-height: 38px;
}
.btn:hover { color: var(--ink); border-color: var(--ink-3); }
.spacer { flex: 1 1 auto; }
.count { font-size: 0.85rem; color: var(--ink-3); margin: 10px 0 0; }

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
tr[hidden] { display: none; }
.scroll { max-height: 460px; overflow-y: auto; border-radius: 8px; }
.scroll thead th {
  position: sticky; top: 0; background: var(--surface);
  box-shadow: inset 0 -1px 0 var(--border);
}
section[hidden] { display: none; }
footer {
  margin-top: 56px; padding-top: 18px; border-top: 1px solid var(--border);
  font-size: 0.85rem; color: var(--ink-3);
}
code {
  background: var(--sunken); padding: 1px 5px; border-radius: 4px;
  font-size: 0.88em;
}
@media (max-width: 640px) {
  .wrap { padding: 20px 16px 60px; }
  h1 { font-size: 1.4rem; }
  .controls { position: static; }
  .controls input[type="search"] { min-width: 100%; }
}
@media print {
  .controls { display: none; }
  section[hidden], tr[hidden] { display: revert !important; }
  .scroll { max-height: none; overflow: visible; }
}
"""

# Filtering and sorting run over the rendered tables. No dependencies, and
# the page is fully readable with JavaScript off: nothing starts out hidden,
# so the only thing lost is the controls themselves.
SCRIPT = """
(function () {
  var $ = function (id) { return document.getElementById(id); };
  var outcome = $('f-outcome'), search = $('f-search'), level = $('f-level');
  var sigOnly = $('f-sig'), sort = $('f-sort'), reset = $('f-reset');
  var theme = $('f-theme');
  var sections = Array.prototype.slice.call(
    document.querySelectorAll('section[data-outcome]'));
  var tables = Array.prototype.slice.call(
    document.querySelectorAll('table[data-outcome]'));

  var SORTS = {
    p:      function (r) { return +r.dataset.p; },
    effect: function (r) { return -Math.abs(+r.dataset.effect); },
    mz:     function (r) { return +r.dataset.mz; },
    rt:     function (r) { return +r.dataset.rt; },
    level:  function (r) { return +r.dataset.level; }
  };

  function apply() {
    var want = outcome.value;
    var needle = search.value.trim().toLowerCase();
    var lvl = level.value;
    var onlySig = sigOnly.checked;
    var key = SORTS[sort.value] || SORTS.p;

    sections.forEach(function (s) {
      s.hidden = want !== 'all' && s.dataset.outcome !== want;
    });

    tables.forEach(function (table) {
      var body = table.tBodies[0];
      var rows = Array.prototype.slice.call(body.rows);
      var shown = 0;

      rows.forEach(function (row) {
        var ok = true;
        if (onlySig && row.dataset.sig !== '1') { ok = false; }
        if (ok && lvl !== 'all' && row.dataset.level !== lvl) { ok = false; }
        if (ok && needle) { ok = row.dataset.search.indexOf(needle) !== -1; }
        row.hidden = !ok;
        if (ok) { shown++; }
      });

      rows.sort(function (a, b) { return key(a) - key(b); })
          .forEach(function (row) { body.appendChild(row); });

      var readout = document.getElementById('count-' + table.dataset.outcome);
      if (readout) {
        readout.textContent = shown + ' of ' + rows.length + ' features shown' +
          (shown === 0 ? '. Try clearing the filters.' : '');
      }
    });
  }

  [outcome, level, sort].forEach(function (el) {
    el.addEventListener('change', apply);
  });
  search.addEventListener('input', apply);
  sigOnly.addEventListener('change', apply);

  reset.addEventListener('click', function () {
    outcome.value = 'all'; search.value = ''; level.value = 'all';
    sigOnly.checked = false; sort.value = 'p';
    apply();
    search.focus();
  });

  function currentTheme() {
    var set = document.documentElement.getAttribute('data-theme');
    if (set) { return set; }
    return window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark' : 'light';
  }
  function paintToggle() {
    var now = currentTheme();
    theme.textContent = now === 'dark' ? 'Light mode' : 'Dark mode';
    theme.setAttribute('aria-pressed', now === 'dark' ? 'true' : 'false');
  }
  theme.addEventListener('click', function () {
    document.documentElement.setAttribute(
      'data-theme', currentTheme() === 'dark' ? 'light' : 'dark');
    paintToggle();
  });

  paintToggle();
  apply();
})();
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


def _controls() -> str:
    """The filter bar: one row, above everything it controls."""
    options = "".join(
        f'<option value="{key}">{html.escape(label)}</option>'
        for key, label in OUTCOMES.items()
    )
    levels = "".join(f'<option value="{i}">Level {i}</option>' for i in (1, 2, 3, 4))
    return f"""<div class="controls">
  <div class="control">
    <label for="f-outcome">Outcome</label>
    <select id="f-outcome">
      <option value="all">All outcomes</option>{options}
    </select>
  </div>
  <div class="control">
    <label for="f-search">Search</label>
    <input id="f-search" type="search" placeholder="Feature ID or m/z…"
           autocomplete="off">
  </div>
  <div class="control">
    <label for="f-level">Annotation</label>
    <select id="f-level">
      <option value="all">Any level</option>{levels}
    </select>
  </div>
  <div class="control">
    <label for="f-sort">Sort by</label>
    <select id="f-sort">
      <option value="p">p-value</option>
      <option value="effect">Effect size</option>
      <option value="mz">m/z</option>
      <option value="rt">Retention time</option>
      <option value="level">Annotation level</option>
    </select>
  </div>
  <label class="check" for="f-sig">
    <input type="checkbox" id="f-sig"> Significant only
  </label>
  <button type="button" class="btn" id="f-reset">Reset</button>
  <div class="spacer"></div>
  <button type="button" class="btn" id="f-theme" aria-pressed="false">
    Dark mode
  </button>
</div>"""


def _results_table(assoc: pd.DataFrame, features: pd.DataFrame, outcome: str) -> str:
    """Every curated feature for one outcome, filterable and sortable.

    The whole table is in the document rather than a top-N slice, so the
    search box can actually find things.
    """
    rows = []
    for fid, row in assoc.iterrows():
        meta = features.loc[fid]
        est = row["estimate"]
        sig = bool(row["significant"])
        direction = "up" if est > 0 else "down"
        arrow = "+" if est > 0 else "−"
        hit_class = ' class="hit"' if sig else ""
        needle = f"{fid} {meta['mz']:.4f} {meta['rt_min']:.2f}".lower()
        rows.append(
            f'<tr data-p="{row["p_value"]:.6g}" data-effect="{est:.6g}" '
            f'data-mz="{meta["mz"]:.4f}" data-rt="{meta["rt_min"]:.2f}" '
            f'data-level="{int(meta["annotation_level"])}" '
            f'data-sig="{1 if sig else 0}" data-search="{html.escape(needle)}">'
            f"<td{hit_class}>{fid}</td>"
            f"<td>{meta['mz']:.4f}</td>"
            f"<td>{meta['rt_min']:.2f}</td>"
            f"<td>L{int(meta['annotation_level'])}</td>"
            f'<td class="{direction}">{arrow}{abs(est):.3f}</td>'
            f"<td>{row['std_error']:.3f}</td>"
            f"<td>{row['p_value']:.2e}</td>"
            f"<td>{row['q_value']:.3f}</td>"
            f"<td>{'yes' if sig else '—'}</td></tr>"
        )
    return (
        f'<p class="count" id="count-{outcome}">'
        f"{len(rows)} of {len(rows)} features shown</p>"
        f'<div class="scroll"><table class="results" data-outcome="{outcome}">'
        "<thead><tr><th>Feature</th><th>m/z</th><th>RT (min)</th>"
        "<th>Level</th><th>Estimate</th><th>SE</th><th>p</th><th>q</th>"
        "<th>FDR hit</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
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
        f"<tr><td>Level {lvl}: {names[lvl]}</td><td>{int(n)}</td>"
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
</header>""")
    parts.append(_controls())
    parts.append("""<div class="note">
  <strong>Synthetic data.</strong> Every number on this page comes from a seeded
  simulation in <code>omicsdash.simulate</code>. There's no real study, subject
  or measurement behind it, and you don't need any to reproduce it.
</div>""")

    # 1. Curation
    parts.append("<h2>1 · Data curation</h2>")
    parts.append(
        "<p>Features are kept if they're detected in at least 70% of study "
        "samples and their RSD across the pooled QC injections is 30% or less. "
        "Remaining gaps are filled with half the feature minimum, then the "
        "table is log2-transformed and median-centred per sample.</p>"
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
        "have analytical drift but no biological variation, so their trend "
        "line is used as the drift estimate.",
    ))

    # 2. Feature set
    parts.append("<h2>2 · Feature set and annotation levels</h2>")
    parts.append(
        "<p>Annotation level is how confidently a feature has been identified. "
        "Most untargeted features stay unknown, which is normal, and it's why "
        "the results below are reported by feature rather than as named "
        "metabolites. The annotation filter at the top can limit the result "
        "tables to the confidently identified ones.</p>"
    )
    parts.append(_annotation_table(study.features, curated.kept))

    # 3. PCA
    parts.append("<h2>3 · Principal component analysis</h2>")
    parts.append(
        "<p>A quick unsupervised check on the curated table. If samples "
        "separate by batch along an early component, acquisition is "
        "contributing more variance than biology.</p>"
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

    # 4+. One section per outcome, each hideable by the outcome selector.
    for i, (outcome, label) in enumerate(OUTCOMES.items(), start=4):
        assoc = results[outcome]
        n_sig = int(assoc["significant"].sum())
        body = [
            f"<h2>{i} · {html.escape(label)}</h2>",
            f"<p>One linear model per feature: "
            f"<code>feature ~ {outcome} + age + sex + bmi</code>, with "
            f"Benjamini-Hochberg control across the {s['features_kept']:,} "
            f"curated features. Estimates are log2 abundance change per "
            f"standard deviation of the outcome.</p>",
            _tiles([
                (f"{n_sig}", f"features at FDR {alpha:g}"),
                (f"{assoc['p_value'].min():.1e}", "smallest p"),
                (f"{int(assoc['n'].iloc[0])}", "samples modelled"),
            ]),
            _fig(
                figures.volcano(assoc, outcome, LIGHT, alpha),
                figures.volcano(assoc, outcome, DARK, alpha),
                f"Volcano plot of feature associations with {label}",
                "Effect size against -log10 p-value. Colour only shows "
                "direction, and the dashed line marks the FDR cutoff.",
            ),
        ]
        if n_sig:
            body.append(_fig(
                figures.forest(assoc, outcome, LIGHT),
                figures.forest(assoc, outcome, DARK),
                f"Effect estimates with 95% intervals for {label}",
                "Strongest associations with 95% confidence intervals.",
            ))
        body.append("<h3>All features</h3>")
        body.append(_results_table(assoc, study.features, outcome))
        parts.append(
            f'<section data-outcome="{outcome}">' + "".join(body) + "</section>"
        )

    # Ground truth, which is only knowable because the data is simulated.
    parts.append("<h2>Appendix · Planted effects</h2>")
    parts.append(
        "<p>These are the associations the simulator wrote into the data, "
        "listed so the results can be checked against them. The pipeline "
        "should recover these and not much else.</p>"
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
<script>{SCRIPT}</script>
</body>
</html>"""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path
