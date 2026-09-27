"""Only documented, sufficiently precise catalogue markers may skip a refresh."""
from __future__ import annotations

import copy
import datetime as dt

import pytest

from terrastat.sources import get_source
from terrastat.sources.base import DatasetRef, ref_from_row
from terrastat.sources.eurostat import EurostatSource


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setattr("terrastat.sources.eurostat._utc_today", lambda: dt.date(2026, 9, 21))


@pytest.fixture
def ref():
    return DatasetRef(
        "eurostat", "example", "Example dataset", ["Q", "A"], 120,
        last_updated="18.09.2026",
        extra={
            "last_structure_change": "12.09.2026",
            "type": "dataset", "data_start": "2000-Q1", "data_end": "2026-Q2",
            "source": "Eurostat", "short_description": "Quarterly observations",
            "metadata": {"html": "https://example.test/metadata", "sdmx": "https://example.test/sdmx"},
            "dimensions": ["freq", "unit", "geo", "time"],
            "paths": ["Economy > Output", "Quarterly accounts"],
        },
    )


def test_eligible_token_is_stable_across_serialization_order_and_retrieval_dates(ref):
    source = EurostatSource()
    baseline = source.refresh_token(ref)
    assert baseline and baseline.startswith("eurostat-toc-v1:")
    assert len(baseline.rsplit(":", 1)[1]) == 64
    other = copy.deepcopy(ref)
    other.extra = dict(reversed(list(other.extra.items())))
    other.extra["metadata"] = dict(reversed(list(other.extra["metadata"].items())))
    other.extra["paths"].reverse()
    other.frequencies.reverse()
    other.extra["retrieved_at"] = "2026-09-21T16:00:00Z"
    other.extra["checked_at"] = "2026-09-21T17:00:00Z"
    assert source.refresh_token(other) == baseline
    # Equivalent known date representations do not create artificial changes.
    other.last_updated = "2026-09-18"
    other.extra["last_structure_change"] = "2026-09-12"
    assert source.refresh_token(other) == baseline


@pytest.mark.parametrize("field,value", [
    ("last_updated", "17.09.2026"),
    ("last_structure_change", "13.09.2026"),
    ("title", "Revised dataset title"),
    ("dataset_id", "another_dataset"),
    ("n_values", 121),
    ("frequencies", ["Q"]),
    ("data_end", "2026-Q3"),
    ("short_description", "Revised description"),
    ("metadata", {"html": "https://example.test/new-metadata"}),
    ("dimensions", ["freq", "geo", "unit", "time"]),
])
def test_data_structure_and_catalogue_metadata_changes_change_the_token(ref, field, value):
    baseline = EurostatSource().refresh_token(ref)
    if hasattr(ref, field):
        setattr(ref, field, value)
    else:
        ref.extra[field] = value
    assert EurostatSource().refresh_token(ref) != baseline


@pytest.mark.parametrize("marker", [None, "", " ", "unknown", "31.02.2026", "2026", 123])
@pytest.mark.parametrize("field", ["data", "structure"])
def test_missing_or_malformed_either_marker_requires_download(ref, field, marker):
    if field == "data":
        ref.last_updated = marker
    else:
        ref.extra["last_structure_change"] = marker
    assert EurostatSource().refresh_token(ref) is None


def test_absent_structure_marker_requires_download(ref):
    ref.extra.pop("last_structure_change")
    assert EurostatSource().refresh_token(ref) is None


@pytest.mark.parametrize("marker", ["20.09.2026", "21.09.2026", "22.09.2026"])
@pytest.mark.parametrize("field", ["data", "structure"])
def test_recent_and_future_markers_cannot_hide_another_intraday_release(ref, field, marker):
    if field == "data":
        ref.last_updated = marker
    else:
        ref.extra["last_structure_change"] = marker
    assert EurostatSource().refresh_token(ref) is None


def test_old_marker_token_stays_stable_as_time_passes(ref, monkeypatch):
    baseline = EurostatSource().refresh_token(ref)
    monkeypatch.setattr("terrastat.sources.eurostat._utc_today", lambda: dt.date(2026, 12, 21))
    assert EurostatSource().refresh_token(ref) == baseline


def test_two_days_old_markers_are_eligible(ref):
    ref.last_updated = ref.extra["last_structure_change"] = "19.09.2026"
    assert EurostatSource().refresh_token(ref) is not None


@pytest.mark.parametrize("source,extra", [
    ("fred", {"realtime_start": "2026-09-18", "realtime_end": "2026-09-18"}),
    ("oecd", {"version": "1.2", "agency": "OECD", "last_structure_change": "12.09.2026"}),
])
def test_sources_without_observation_revision_markers_always_require_download(source, extra):
    ref = DatasetRef(source, "example", "Example", last_updated="18.09.2026", extra=extra)
    assert get_source(source).refresh_token(ref) is None


def test_catalogue_row_conversion_keeps_the_two_eurostat_markers():
    ref = ref_from_row("eurostat", {
        "dataset_id": "example", "title": "Example", "last_updated": "18.09.2026",
        "extra": '{"last_structure_change": "12.09.2026"}',
    })
    assert EurostatSource().refresh_token(ref) is not None
