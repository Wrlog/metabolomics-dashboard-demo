"""Chart and page theme, defined once for both light and dark rendering.

Colours come from a validated reference palette. Only the first three
categorical slots are used: that subset is the documented all-pairs safe set
for scatter-type charts, where every series can end up adjacent to every
other. Diverging comparisons use the blue/red pair with a neutral grey
midpoint, never a rainbow.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    surface_sunken: str
    border: str
    text_primary: str
    text_secondary: str
    text_muted: str
    grid: str
    series: tuple[str, str, str]
    diverging_low: str  # negative pole
    diverging_high: str  # positive pole
    neutral: str  # not significant / midpoint

    @property
    def is_dark(self) -> bool:
        return self.name == "dark"


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    surface_sunken="#f4f3f0",
    border="#e2e1dc",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    text_muted="#78766f",
    grid="#e8e7e2",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    diverging_low="#2a78d6",
    diverging_high="#e34948",
    neutral="#b6b4ad",
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    surface_sunken="#232322",
    border="#3a3a37",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    text_muted="#95948b",
    grid="#2f2f2d",
    series=("#3987e5", "#d95926", "#199e70"),
    diverging_low="#3987e5",
    diverging_high="#e66767",
    neutral="#5c5b55",
)

THEMES = (LIGHT, DARK)
