"""Frequencies and time periods, normalised across sources.

Canonical frequency codes used everywhere in terrastat, finest to coarsest:

    H  hourly
    D  daily (every calendar day)
    B  daily, business week only (SDMX "B")
    W  weekly
    BW biweekly
    M  monthly
    Q  quarterly
    S  semiannual / half-yearly
    A  annual
    A3 the source's own "triannual" bucket, kept verbatim because the label is
       ambiguous in English (three times a year vs every three years)
    P  pluriannual: any fixed multi-year step (FRED "5 Year"/"10 Year", Eurostat "P")
    I  irregular / a-periodic: real observations, no fixed step
    OTHER  the source declares no applicable frequency, or a code we do not know

This vocabulary is the union of SDMX's CL_FREQ (which Eurostat and OECD both build on)
and the strings FRED reports; the codes are SDMX's own letters wherever one exists.
The raw string from the source is always kept next to the canonical code in
``frequency_raw``, so nothing here loses information: a mapping only ever adds a
coarser label on top of what the source said.
"""
from __future__ import annotations

import datetime as dt
import re

CANONICAL = ("H", "D", "B", "W", "BW", "M", "Q", "S", "A", "A3", "P", "I", "OTHER")

# FRED reports a human string. These are matched as prefixes, because FRED qualifies many of
# them ("Weekly, Ending Friday", "Daily, 7-Day"). Verified complete against all 845,518 series
# of all 331 releases on 2026-09-04: the only strings FRED uses are Annual, Monthly, Quarterly,
# Daily, Weekly, Semiannual, 5 Year, 10 Year, Biweekly and "Not Applicable".
_FRED_PREFIXES = [
    ("hourly", "H"),
    ("daily", "D"),
    ("biweekly", "BW"),
    ("bi-weekly", "BW"),
    ("weekly", "W"),
    ("monthly", "M"),
    ("quarterly", "Q"),
    ("semiannual", "S"),
    ("semi-annual", "S"),
    ("annual", "A"),
    ("5 year", "P"),
    ("10 year", "P"),
    ("not applicable", "OTHER"),
]

# Eurostat codelist ESTAT:FREQ (v3.9, 12 codes) and OECD's CL_FREQ, in full.
_SDMX_CODES = {
    "H": "H",      # hourly
    "D": "D",      # daily
    "B": "B",      # daily, business week
    "W": "W",      # weekly
    "M": "M",      # monthly
    "Q": "Q",      # quarterly
    "S": "S",      # half-yearly, semesterly
    "A": "A",      # annual
    "A3": "A3",    # Eurostat "Triannual"
    "P": "P",      # pluri-annual
    "I": "I",      # irregular / a-periodic
    "NAP": "OTHER",  # not applicable
    "N": "OTHER",  # SDMX "minutely"; no economic series uses it, kept from being mistaken
}
_EUROSTAT_CODES = _SDMX_CODES
_OECD_CODES = _SDMX_CODES


def canonical_frequency(raw: str | None, source: str = "") -> str:
    """Map a source-specific frequency label or code to the canonical vocabulary.

    SDMX sources send a code ("M", "A3"); FRED sends a human string ("Weekly, Ending Friday").
    Anything unrecognised becomes OTHER rather than a guess, and ``frequency_raw`` keeps the
    original either way.
    """
    if raw is None:
        return "OTHER"
    s = str(raw).strip()
    if not s:
        return "OTHER"
    if source in ("eurostat", "oecd") and s.upper() in _SDMX_CODES:
        return _SDMX_CODES[s.upper()]
    low = s.lower()
    for prefix, code in _FRED_PREFIXES:
        if low.startswith(prefix):
            return code
    if s.upper() in _SDMX_CODES:  # a bare SDMX code from any source
        return _SDMX_CODES[s.upper()]
    return "OTHER"


_RE_YEAR = re.compile(r"^(\d{4})$")
_RE_SEM = re.compile(r"^(\d{4})-?S([12])$")
_RE_QTR = re.compile(r"^(\d{4})-?Q([1-4])$")
_RE_MON = re.compile(r"^(\d{4})(?:-|M)(\d{2})$")
_RE_WEEK = re.compile(r"^(\d{4})-?W(\d{2})$")
_RE_DAY = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_RE_DAY2 = re.compile(r"^(\d{4})M(\d{2})D(\d{2})$")
_RE_DAYCOMPACT = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_RE_HOUR = re.compile(r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2})(?::\d{2})*")

# Periods that look like a year but are not one. Eurostat writes "9999" for a long-term average
# (env_wat_ltaa) and "LTAA" elsewhere for the same idea; read as a year, 9999 would poison any
# max(end_date). The observation is still kept, with a null date and frequency OTHER.
SENTINEL_PERIODS = frozenset({"9999", "0000", "LTAA", "TOTAL", "_T", "NAP", "NA"})


def period_to_date(period: str | None) -> dt.date | None:
    """First calendar day of the period. Accepts the Eurostat, SDMX and ISO spellings.

    2015 -> 2015-01-01; 2015-S2 / 2015S2 -> 2015-07-01; 2015-Q4 / 2015Q4 -> 2015-10-01;
    2015-02 / 2015M02 -> 2015-02-01; 2015-W05 / 2015W05 -> Monday of ISO week 5;
    2015-12-31 / 2015M12D31 -> 2015-12-31. Anything else -> None.
    """
    if period is None:
        return None
    s = str(period).strip()
    if s in SENTINEL_PERIODS:
        return None
    try:
        if m := _RE_YEAR.match(s):
            return dt.date(int(m[1]), 1, 1)
        if m := _RE_MON.match(s):
            return dt.date(int(m[1]), int(m[2]), 1)
        if m := _RE_QTR.match(s):
            return dt.date(int(m[1]), (int(m[2]) - 1) * 3 + 1, 1)
        if m := _RE_DAY.match(s):
            return dt.date(int(m[1]), int(m[2]), int(m[3]))
        if m := _RE_SEM.match(s):
            return dt.date(int(m[1]), 1 if m[2] == "1" else 7, 1)
        if m := _RE_WEEK.match(s):
            return dt.date.fromisocalendar(int(m[1]), int(m[2]), 1)
        if m := _RE_DAY2.match(s):
            return dt.date(int(m[1]), int(m[2]), int(m[3]))
        if m := _RE_HOUR.match(s):  # hourly periods collapse to their day
            return dt.date(int(m[1]), int(m[2]), int(m[3]))
        if m := _RE_DAYCOMPACT.match(s):
            return dt.date(int(m[1]), int(m[2]), int(m[3]))
    except ValueError:
        return None
    return None


def frequency_of_period(period: str | None) -> str:
    """Canonical frequency implied by how a period is written ('2015-Q4' -> 'Q')."""
    if period is None:
        return "OTHER"
    s = str(period).strip()
    if s in SENTINEL_PERIODS:
        return "OTHER"
    if _RE_YEAR.match(s):
        return "A"
    if _RE_MON.match(s):
        return "M"
    if _RE_QTR.match(s):
        return "Q"
    if _RE_HOUR.match(s):
        return "H"
    if _RE_DAY.match(s) or _RE_DAY2.match(s) or _RE_DAYCOMPACT.match(s):
        return "D"
    if _RE_SEM.match(s):
        return "S"
    if _RE_WEEK.match(s):
        return "W"
    return "OTHER"
