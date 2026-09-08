"""The contract every source plugin fulfils, and the dataset-level metadata record.

A *dataset* is the natural bulk unit of a source: a FRED release, a Eurostat dataset, an OECD
dataflow. Layer 1 stores one folder per dataset with:

``observations.parquet`` -- long table with at least these columns::

    series_key  str   unique within the dataset (FRED series id; SDMX key like "A.CP_MEUR.B1GQ.LU")
    date        date  first calendar day of the period
    period      str   the period exactly as the source wrote it ("2015-Q4", "2015-02", ...)
    value       f64   null only when a flag explains why (confidential, not applicable ...)
    flag        str   source's observation status code, null when none

plus one column per dimension for SDMX sources (codes, not labels).

``series.parquet`` -- optional per-series metadata when the source provides it (FRED).

``dataset.json`` -- the :class:`DatasetMeta` record below.
"""
from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field

import polars as pl

from terrastat.http import PoliteClient
from terrastat.pacing import Pacer

CATALOG_SCHEMA = {
    "dataset_id": pl.String,  # folder-safe id
    "title": pl.String,
    "frequencies": pl.List(pl.String),  # canonical codes known before download (may be empty)
    "n_values": pl.Int64,  # size hint: number of stored values, null when unknown
    "last_updated": pl.String,  # as reported by the source
    "extra": pl.String,  # JSON with anything else the catalogue told us
}


class NotTimeSeries(Exception):
    """The payload has no time dimension: a questionnaire or a cross-section, not a time series."""


class NoData(Exception):
    """The catalogue lists it but the source has no observations for it (SDMX answers 404).

    Terminal, like NotTimeSeries: retrying cannot conjure data that the publisher never released.
    """


# Column names terrastat uses in observations.parquet. A source dimension that collides with one
# of these (Eurostat has four datasets with a dimension literally named "value") is suffixed.
RESERVED_COLUMNS = frozenset({"series_key", "period", "date", "value", "flag"})


def safe_dim_name(name: str) -> str:
    return f"{name}_dim" if name.lower() in RESERVED_COLUMNS else name


@dataclass
class DatasetRef:
    source: str
    dataset_id: str
    title: str
    frequencies: list[str] = field(default_factory=list)
    n_values: int | None = None
    last_updated: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class License:
    id: str  # short stable id, e.g. "fred-public-domain-citation-requested"
    name: str  # human name
    url: str  # where the terms live
    attribution: str  # the citation the source asks for
    notes: str = ""  # caveats (exceptions, pending clarifications)


@dataclass
class Dimension:
    id: str  # column name in observations.parquet
    name: str  # human name of the concept
    codes: dict  # code -> label


@dataclass
class DatasetMeta:
    source: str
    dataset_id: str
    title: str
    description: str = ""
    source_url: str = ""  # landing page of the dataset at the source
    api_url: str = ""  # exact request(s) used
    metadata_urls: dict = field(default_factory=dict)  # e.g. {"esms_html": ..., "esms_sdmx": ...}
    license: dict = field(default_factory=dict)  # License as dict; FRED: summary, per-series in series.parquet
    origin: dict = field(default_factory=dict)  # {"agencies": [...], "us_federal": bool|None}
    frequencies: list = field(default_factory=list)
    dimensions: list = field(default_factory=list)  # [Dimension as dict]
    attributes: list = field(default_factory=list)  # [Dimension as dict], e.g. OBS_FLAG codes
    time_start: str | None = None
    time_end: str | None = None
    n_series: int = 0
    n_obs: int = 0
    last_updated: str | None = None
    retrieved_at: str | None = None
    vintage: str = "latest"  # terrastat stores the latest revision only; see README on vintages
    raw_files: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetResult:
    dataset_id: str
    n_series: int
    n_obs: int
    frequencies: list[str]


class Source(abc.ABC):
    name: str = ""
    default_min_interval: float = 2.0
    default_max_per_minute: int | None = 30

    def make_client(self, min_interval: float | None = None, max_per_minute: int | None = None) -> PoliteClient:
        pacer = Pacer(
            min_interval=self.default_min_interval if min_interval is None else min_interval,
            max_per_minute=self.default_max_per_minute if max_per_minute is None else max_per_minute,
        )
        return PoliteClient(pacer, headers=self.headers())

    def headers(self) -> dict:
        return {}

    @abc.abstractmethod
    def catalog(self, client: PoliteClient) -> pl.DataFrame:
        """Everything the source offers, as a frame with CATALOG_SCHEMA columns."""

    @abc.abstractmethod
    def fetch_and_tidy(self, ref: DatasetRef, client: PoliteClient, keep_raw: bool = True) -> DatasetResult:
        """Download one dataset and write its layer-1 folder. Must be idempotent and atomic.

        When the raw payload is already on disk (from ``fetch_raw``) no data request is made."""

    @abc.abstractmethod
    def fetch_raw(self, ref: DatasetRef, client: PoliteClient) -> list:
        """Download only the raw payload of one dataset (the cheapest possible crawl). Returns the files."""

    def license_for_series(self, meta: dict, series_row: dict | None) -> dict:
        """License record for one series (sources with per-series licensing override this)."""
        return meta.get("license", {})


def ref_from_row(source: str, row: dict) -> DatasetRef:
    import json

    return DatasetRef(
        source=source,
        dataset_id=row["dataset_id"],
        title=row.get("title") or "",
        frequencies=list(row.get("frequencies") or []),
        n_values=row.get("n_values"),
        last_updated=row.get("last_updated"),
        extra=json.loads(row["extra"]) if row.get("extra") else {},
    )
