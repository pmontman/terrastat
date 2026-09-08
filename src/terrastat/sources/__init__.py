"""Source plugins. Each module exposes a ``Source`` subclass registered here by name."""
from __future__ import annotations

from terrastat.sources.base import Source


def get_source(name: str) -> Source:
    name = name.lower()
    if name == "fred":
        from terrastat.sources.fred import FredSource

        return FredSource()
    if name == "eurostat":
        from terrastat.sources.eurostat import EurostatSource

        return EurostatSource()
    if name == "oecd":
        from terrastat.sources.oecd import OecdSource

        return OecdSource()
    raise ValueError(f"unknown source {name!r}; choose fred, eurostat or oecd")


SOURCES = ("fred", "eurostat", "oecd")
