"""Bounded reconstruction of sparse series; storage remains unchanged."""
from __future__ import annotations

import datetime as dt

import numpy as np


def regularize(dates, values, frequency: str, max_periods: int = 10_000) -> dict:
    """Return dates, values and observed mask on a regular grid, with gaps as NaN.

    Expand only between the earliest and latest stored dates. M/Q/S/A dates map to
    calendar period starts. W/BW dates must align to the first date. B means weekdays,
    not an exchange holiday calendar. H/I/P/A3/OTHER cannot be inferred safely here.
    Reject duplicate periods, null dates, year 9999 and infinite values. Count periods
    before allocating the expanded arrays; raise when max_periods would be exceeded.
    """
    if not isinstance(max_periods, int) or isinstance(max_periods, bool) or max_periods < 1:
        raise ValueError("max_periods must be a positive integer")
    months = {"M": 1, "Q": 3, "S": 6, "A": 12}
    days = {"D": 1, "W": 7, "BW": 14}
    if frequency not in {*months, *days, "B"}:
        raise ValueError(f"no unambiguous calendar grid for frequency {frequency!r}")
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or len(dates) != v.size:
        raise ValueError("dates and values must be one-dimensional and have equal lengths")
    if np.isinf(v).any():
        raise ValueError("infinite observations are not valid missing values")
    if not v.size:
        return {"dates": [], "values": v, "observed": np.isfinite(v)}
    if any(not isinstance(d, dt.date) or isinstance(d, dt.datetime) or d.year == 9999 for d in dates):
        raise ValueError("dates must be calendar dates, without nulls or the sentinel year 9999")
    if frequency in months:
        step = months[frequency]
        codes = np.array([(d.year * 12 + d.month - 1) // step for d in dates], dtype=np.int64)

        def decode(code):
            year, month = divmod(int(code) * step, 12)
            return dt.date(year, month + 1, 1)
    elif frequency == "B":
        raw = np.array(dates, dtype="datetime64[D]")
        if not np.is_busday(raw).all():
            raise ValueError("business-day observations must fall on weekdays")
        codes = np.busday_count(np.datetime64("1970-01-01"), raw)

        def decode(code):
            return np.busday_offset(np.datetime64("1970-01-01"), int(code)).astype(dt.date)
    else:
        step = days[frequency]
        anchor = min(dates).toordinal()
        offsets = np.array([d.toordinal() - anchor for d in dates], dtype=np.int64)
        if (offsets % step).any():
            raise ValueError(f"dates are not aligned to a {frequency} grid")
        codes = offsets // step

        def decode(code):
            return dt.date.fromordinal(anchor + int(code) * step)
    lo, hi = int(codes.min()), int(codes.max())
    count = hi - lo + 1
    if count > max_periods:
        raise ValueError(f"calendar expansion needs {count:,} periods; max_periods={max_periods:,}")
    if np.unique(codes).size != codes.size:
        raise ValueError("duplicate observations in the same calendar period")
    expanded = np.full(count, np.nan)
    expanded[codes - lo] = v
    return {"dates": [decode(c) for c in range(lo, hi + 1)],
            "values": expanded, "observed": np.isfinite(expanded)}
