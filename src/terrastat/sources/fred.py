"""FRED (Federal Reserve Bank of St. Louis), through the v2 bulk endpoint.

Catalogue : ``/fred/releases`` (v1). Every FRED series belongs to exactly one release, so walking
            the releases once covers the whole database, at every frequency.
Data      : ``/fred/v2/release/observations`` returns every series of a release with its
            observations and metadata (title, units, frequency, seasonal adjustment, notes,
            last_updated and, importantly, ``copyright_id``), paginated by a cursor. Pages are
            stored raw and the cursor is checkpointed, so an interrupted release resumes.

Licence: FRED assigns one of three statuses to each series, recorded per series in
``series.parquet``. The FRED Terms of Use also restrict archiving and machine-learning use of
content obtained through FRED irrespective of that status. See README, section "Licences".
"""
from __future__ import annotations

import gzip
import json
import logging
import re
from collections import Counter

import polars as pl

from terrastat.config import fred_api_key
from terrastat.http import PoliteClient, StopRequested
from terrastat.sources.base import (
    CATALOG_SCHEMA,
    DatasetMeta,
    DatasetRef,
    DatasetResult,
    License,
    Source,
)
from terrastat.storage import (
    ParquetBatchWriter,
    dataset_dir,
    now_iso,
    raw_dir,
    read_json,
    write_json,
    write_parquet,
)
from terrastat.timeparse import canonical_frequency

log = logging.getLogger(__name__)

RELEASES_URL = "https://api.stlouisfed.org/fred/releases"
V2_OBS_URL = "https://api.stlouisfed.org/fred/v2/release/observations"
SERIES_PAGE = "https://fred.stlouisfed.org/series/{id}"
RELEASE_PAGE = "https://fred.stlouisfed.org/release/?rid={id}"

FRED_TERMS = "https://fred.stlouisfed.org/legal/"
FRED_NOTES = (
    "Status assigned by FRED. FRED's Terms of Use (https://fred.stlouisfed.org/legal/) restrict "
    "storing/archiving FRED content and its use for training machine-learning systems regardless of "
    "this status, and describe 'Public Domain: Citation Requested' series as ones that 'may be under "
    "copyright or in the public domain'. Works of the US federal government are public domain by "
    "statute (17 U.S.C. 105) at their origin; see origin.us_federal."
)

LICENSES = {
    "public domain: citation requested": License(
        id="fred-public-domain-citation-requested",
        name="FRED: Public Domain, Citation Requested",
        url=FRED_TERMS,
        attribution="{agency}, {title} [{series_id}], retrieved from FRED, Federal Reserve Bank of St. Louis; {url}, {date}.",
        notes=FRED_NOTES,
    ),
    "copyrighted: citation required": License(
        id="fred-copyrighted-citation-required",
        name="FRED: Copyrighted, Citation Required",
        url=FRED_TERMS,
        attribution="{agency}, {title} [{series_id}], retrieved from FRED, Federal Reserve Bank of St. Louis; {url}, {date}.",
        notes="Under copyright of the data provider; use with attribution of the source and of FRED. " + FRED_NOTES,
    ),
    "copyrighted: pre-approval required": License(
        id="fred-copyrighted-preapproval-required",
        name="FRED: Copyrighted, Pre-approval Required",
        url=FRED_TERMS,
        attribution="{agency}, {title} [{series_id}], retrieved from FRED, Federal Reserve Bank of St. Louis; {url}, {date}.",
        notes="Non-commercial educational or personal use only without written permission of the copyright holder. " + FRED_NOTES,
    ),
}
UNKNOWN_LICENSE = License(
    id="fred-unknown",
    name="FRED: no copyright status reported",
    url=FRED_TERMS,
    attribution="{agency}, {title} [{series_id}], retrieved from FRED, Federal Reserve Bank of St. Louis; {url}, {date}.",
    notes=FRED_NOTES,
)

# Heuristic: names of US federal bodies whose works carry no copyright of their own, so that a
# series they originate can be redistributed. Checked against every agency name appearing on a
# FRED public-domain series (76 of them) rather than guessed.
#
# One judgement call is recorded here explicitly: the twelve regional Federal Reserve Banks are
# quasi-public corporations rather than federal agencies, so 17 U.S.C. 105 does not apply to them
# automatically. They are included because FRED itself tags their series "Public Domain: Citation
# Requested" and because in practice these are transformations of federal source data (the largest
# group is "U.S. Bureau of Labor Statistics + Federal Reserve Bank of St. Louis"). If you would
# rather exclude them, drop the "federal reserve bank of" line and rebuild with
# `terrastat series fred --force`.
US_FEDERAL_PATTERNS = [
    r"bureau of labor statistics",
    r"bureau of economic analysis",
    r"census bureau",
    r"board of governors of the federal reserve",
    r"federal reserve bank of",
    r"federal open market committee",
    r"federal bureau of investigation",
    r"federal highway administration",
    r"council of economic advisers",
    r"u\.?s\.? department of",
    r"u\.?s\.? office of",
    r"department of the treasury",
    r"department of housing and urban development",
    r"employment and training administration",
    r"bureau of transportation statistics",
    r"energy information administration",
    r"congressional budget office",
    r"office of management and budget",
    r"social security administration",
    r"federal housing finance agency",
    r"federal deposit insurance corporation",
    r"national credit union administration",
    r"internal revenue service",
    r"small business administration",
    r"bureau of the fiscal service",
    r"centers for disease control",
    r"national center for",
    r"federal financial institutions examination council",
    r"environmental protection agency",
    r"bureau of the census",
    r"u\.?s\.? bureau",
    r"office of the comptroller of the currency",
    r"federal emergency management",
    r"u\.?s\.? patent",
]
_US_FED_RE = re.compile("|".join(US_FEDERAL_PATTERNS), re.IGNORECASE)


def is_us_federal(agencies: list[str]) -> bool | None:
    """True if every source of the release matches the federal-agency list; None if no sources."""
    if not agencies:
        return None
    return all(bool(_US_FED_RE.search(a)) for a in agencies)


SERIES_META_SCHEMA = {
    "series_id": pl.String,
    "title": pl.String,
    "frequency_raw": pl.String,
    "frequency": pl.String,
    "units": pl.String,
    "seasonal_adjustment": pl.String,
    "last_updated": pl.String,
    "copyright_id": pl.String,
    "notes": pl.String,
    "n_obs": pl.Int64,
    "n_missing": pl.Int64,
    "obs_start": pl.String,
    "obs_end": pl.String,
}


class FredSource(Source):
    name = "fred"
    default_min_interval = 2.0
    default_max_per_minute = 20
    page_limit = 250_000  # observations per request (API maximum is 500,000)

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {fred_api_key()}"}

    # -- catalogue ----------------------------------------------------------------------------

    def catalog(self, client: PoliteClient) -> pl.DataFrame:
        rows, offset = [], 0
        while True:
            data = client.get_json(RELEASES_URL, params={"api_key": fred_api_key(), "file_type": "json", "limit": 1000, "offset": offset})
            batch = data.get("releases", [])
            for r in batch:
                rows.append(
                    {
                        "dataset_id": str(r["id"]),
                        "title": r.get("name", ""),
                        "frequencies": [],
                        "n_values": None,
                        "last_updated": r.get("realtime_start"),
                        "extra": json.dumps({"link": r.get("link"), "press_release": r.get("press_release"), "notes": r.get("notes", "")}, ensure_ascii=False),
                    }
                )
            if len(batch) < 1000:
                break
            offset += 1000
        df = pl.DataFrame(rows, schema=CATALOG_SCHEMA)
        log.info("fred catalogue: %d releases", df.height)
        return df

    # -- one release --------------------------------------------------------------------------

    def fetch_raw(self, ref: DatasetRef, client: PoliteClient) -> list:
        return self._download_pages(ref.dataset_id, raw_dir(self.name, ref.dataset_id), client)

    def fetch_and_tidy(self, ref: DatasetRef, client: PoliteClient, keep_raw: bool = True) -> DatasetResult:
        rid = ref.dataset_id
        raw = raw_dir(self.name, rid)
        out = dataset_dir(self.name, rid)
        pages = self._download_pages(rid, raw, client)

        writer = ParquetBatchWriter(out / "observations.parquet")
        series: dict[str, dict] = {}
        release_info: dict = {}
        try:
            for page_path in pages:
                with gzip.open(page_path, "rt", encoding="utf-8") as fh:
                    page = json.load(fh)
                release_info = page.get("release", release_info) or release_info
                keys, dates, values = [], [], []
                for s in page.get("series", []):
                    sid = s["series_id"]
                    rec = series.get(sid)
                    if rec is None:
                        rec = series[sid] = {
                            "series_id": sid,
                            "title": s.get("title"),
                            "frequency_raw": s.get("frequency"),
                            "frequency": canonical_frequency(s.get("frequency"), "fred"),
                            "units": s.get("units"),
                            "seasonal_adjustment": s.get("seasonal_adjustment"),
                            "last_updated": s.get("last_updated"),
                            "copyright_id": s.get("copyright_id"),
                            "notes": s.get("notes") or None,
                            "n_obs": 0,
                            "n_missing": 0,
                            "obs_start": None,
                            "obs_end": None,
                        }
                    for o in s.get("observations", []):
                        v = o.get("value")
                        if v is None or v == ".":
                            rec["n_missing"] += 1
                            continue
                        try:
                            fv = float(v)
                        except ValueError:
                            rec["n_missing"] += 1
                            continue
                        d = o["date"]
                        keys.append(sid)
                        dates.append(d)
                        values.append(fv)
                        rec["n_obs"] += 1
                        if rec["obs_start"] is None or d < rec["obs_start"]:
                            rec["obs_start"] = d
                        if rec["obs_end"] is None or d > rec["obs_end"]:
                            rec["obs_end"] = d
                if keys:
                    df = pl.DataFrame({"series_key": keys, "period": dates, "value": values}).with_columns(
                        pl.col("period").str.to_date("%Y-%m-%d").alias("date"),
                        pl.lit(None, pl.String).alias("flag"),
                    )
                    writer.write(df.select("series_key", "period", "date", "value", "flag").sort("series_key", "date"))
        except Exception:
            writer.abort()
            raise
        writer.close()

        sdf = pl.DataFrame(list(series.values()), schema=SERIES_META_SCHEMA)
        write_parquet(sdf, out / "series.parquet")

        agencies = [s.get("name", "") for s in release_info.get("sources", []) if s.get("name")]
        lic_counts = Counter(series[s]["copyright_id"] or "unknown" for s in series)
        meta = DatasetMeta(
            source=self.name,
            dataset_id=rid,
            title=release_info.get("name") or ref.title,
            description=ref.extra.get("notes", ""),
            source_url=release_info.get("url") or ref.extra.get("link") or RELEASE_PAGE.format(id=rid),
            api_url=f"{V2_OBS_URL}?release_id={rid}&format=json&limit={self.page_limit}",
            metadata_urls={"release": RELEASE_PAGE.format(id=rid)},
            license={
                "id": "fred-per-series",
                "name": "FRED copyright status is recorded per series in series.parquet (copyright_id)",
                "url": FRED_TERMS,
                "attribution": "See per-series attribution; general form: '{source}, {title} [{series_id}], retrieved from FRED, Federal Reserve Bank of St. Louis'",
                "notes": FRED_NOTES,
                "counts": dict(lic_counts),
            },
            origin={"agencies": agencies, "sources": release_info.get("sources", []), "us_federal": is_us_federal(agencies)},
            frequencies=sorted({series[s]["frequency"] for s in series}),
            time_start=min((series[s]["obs_start"] for s in series if series[s]["obs_start"]), default=None),
            time_end=max((series[s]["obs_end"] for s in series if series[s]["obs_end"]), default=None),
            n_series=len(series),
            n_obs=int(sdf.get_column("n_obs").sum()) if series else 0,
            last_updated=max((series[s]["last_updated"] or "" for s in series), default=None) or None,
            retrieved_at=now_iso(),
            raw_files=[str(p.relative_to(raw.parent.parent.parent)) for p in pages] if keep_raw else [],
            extra={"catalogue": ref.extra, "frequency_counts": dict(Counter(series[s]["frequency_raw"] for s in series))},
        )
        write_json(out / "dataset.json", meta.to_dict())
        if not keep_raw:
            for p in pages:
                p.unlink(missing_ok=True)
        return DatasetResult(rid, len(series), meta.n_obs, meta.frequencies)

    def _download_pages(self, rid: str, raw, client: PoliteClient) -> list:
        """Download all pages of a release, resuming from the last checkpointed cursor."""
        raw.mkdir(parents=True, exist_ok=True)
        ckpt = raw / "_pages.json"
        state = read_json(ckpt) if ckpt.exists() else {"pages": [], "complete": False}
        if state.get("complete"):
            return [raw / p["file"] for p in state["pages"]]
        cursor = state["pages"][-1]["next_cursor"] if state["pages"] else None
        n = len(state["pages"])
        while True:
            if n > 0 and client.stop_requested():
                raise StopRequested(f"release {rid}: stopped after page {n}; the next run resumes from its cursor")
            params = {"release_id": rid, "format": "json", "limit": self.page_limit}
            if cursor:
                params["next_cursor"] = cursor
            resp = client.get(V2_OBS_URL, params=params)
            if resp.status_code != 200:
                raise RuntimeError(f"FRED v2 HTTP {resp.status_code} for release {rid}: {resp.text[:200]}")
            data = resp.json()
            n += 1
            fname = f"page_{n:04d}.json.gz"
            with gzip.open(raw / fname, "wt", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            has_more = str(data.get("has_more", False)).lower() == "true"
            cursor = data.get("next_cursor") if has_more else None
            state["pages"].append({"file": fname, "next_cursor": cursor, "n_series": len(data.get("series", []))})
            state["complete"] = not has_more
            write_json(ckpt, state)
            if not has_more:
                break
        return [raw / p["file"] for p in state["pages"]]

    # -- per-series licence -------------------------------------------------------------------

    def license_for_series(self, meta: dict, series_row: dict | None) -> dict:
        cid = (series_row or {}).get("copyright_id")
        lic = LICENSES.get((cid or "").strip().lower(), UNKNOWN_LICENSE)
        d = lic.__dict__.copy()
        agencies = meta.get("origin", {}).get("agencies") or ["FRED"]
        sid = (series_row or {}).get("series_id", "")
        d["attribution"] = lic.attribution.format(
            agency="; ".join(agencies),
            title=(series_row or {}).get("title", ""),
            series_id=sid,
            url=SERIES_PAGE.format(id=sid),
            date=(meta.get("retrieved_at") or "")[:10],
        )
        d["detail"] = cid
        return d
