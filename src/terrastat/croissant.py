"""Croissant metadata: the snapshot described in a form machines already read.

`Croissant <https://mlcommons.org/croissant/>`_ is MLCommons' JSON-LD vocabulary for ML datasets,
read by Hugging Face, Kaggle and OpenML. Publishing a ``croissant.json`` beside the shards means
those platforms show the licence, the citation and every column's meaning without anyone
retyping them, and a loader can find the files without being told the layout.

It is written automatically by ``terrastat export``, from the manifest, so it can never disagree
with the shards it sits next to.

Two parts of the vocabulary do real work here beyond ticking a box:

**Per-field descriptions.** Thirty-six columns is a lot to meet cold. Croissant carries a
description per field, so a dataset viewer can explain ``origin_us_federal`` or ``n_flagged`` at
the point the reader sees the column.

**The Responsible AI extension.** ``rai:dataLimitations`` is the right home for the two things
that will otherwise bite someone quietly — the corpus stores latest revisions only, so it cannot
support a genuine real-time exercise, and 78% of datasets publish totals alongside their parts,
so a random train/test split is contaminated. Those belong in the machine-readable metadata, not
only in a document nobody opens.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from terrastat import __version__
from terrastat.export import LICENSE_SUMMARY

log = logging.getLogger(__name__)

CROISSANT_VERSION = "http://mlcommons.org/croissant/1.0"

# The standard Croissant JSON-LD context, verbatim: platforms match on it.
CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "equivalentProperty": "sc:equivalentProperty",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}

# (Croissant dataType, repeated, description). Grouped as in docs/schema.md so a viewer that
# renders them in order tells a coherent story rather than an alphabet.
FIELDS: dict[str, tuple[str, bool, str]] = {
    # identity
    "series_uid": ("sc:Text", False, "Globally unique identifier, '<source>:<source_id>'."),
    "source": ("sc:Text", False, "Which statistical agency published it: fred, eurostat or oecd."),
    "source_id": ("sc:Text", False, "The source's own identifier: a FRED series id, or '<dataset>:<SDMX key>'."),
    "dataset_id": ("sc:Text", False, "The dataset this series belongs to. Split train/test on this, not on series_uid."),
    # human-readable
    "dataset_title": ("sc:Text", False, "Title of the dataset."),
    "title": ("sc:Text", False, "Title of the series: FRED's own, or the dataset title for SDMX sources."),
    "description": ("sc:Text", False, "One paragraph assembled from every dimension label, ready for a text encoder."),
    "notes": ("sc:Text", False, "Free-text notes from the source."),
    # classification
    "frequency": ("sc:Text", False, "Canonical frequency: H D B W BW M Q S A A3 P I OTHER. Comparable across sources."),
    "frequency_raw": ("sc:Text", False, "The source's own frequency label, kept because the mapping is lossy."),
    "units": ("sc:Text", False, "Unit of measure. Decides whether aggregation is meaningful: percentages and indices do not sum."),
    "seasonal_adjustment": ("sc:Text", False, "Whether the series is seasonally adjusted. Adjusted series embed future information at the sample end."),
    "geo": ("sc:Text", False, "Geographic code as given by the source."),
    "geo_label": ("sc:Text", False, "Human-readable name for the geographic code."),
    "dimensions": ("sc:Text", True, "Every SDMX dimension as {id, name, code, label}. Codes ending _T denote a total over that dimension."),
    "tags": ("sc:Text", True, "FRED tags where fetched; empty for other sources."),
    # licence
    "license_id": ("sc:Text", False, "Which licence applies to this series. Filter on it before redistributing."),
    "license_name": ("sc:Text", False, "Name of that licence."),
    "license_url": ("sc:URL", False, "Where to read the licence terms."),
    "license_detail": ("sc:Text", False, "The source's own status string, such as FRED's copyright field."),
    "attribution": ("sc:Text", False, "The exact citation this source asks for. Travels per row so it survives any slicing."),
    "license_notes": ("sc:Text", False, "Caveats attached to the licence."),
    # provenance
    "origin_agencies": ("sc:Text", True, "Who actually produced the numbers. FRED republishes OECD, IMF, World Bank and Eurostat series."),
    "origin_us_federal": ("sc:Boolean", False, "Whether a US federal body produced it, which decides whether it carries copyright (17 USC 105)."),
    "source_url": ("sc:URL", False, "Landing page for the series at the source."),
    "last_updated": ("sc:Text", False, "When the source last updated it."),
    "retrieved_at": ("sc:Text", False, "When this copy was downloaded."),
    "vintage": ("sc:Text", False, "Which revision this is. Latest only: a 2024 value may have been restated later."),
    # extent
    "start_date": ("sc:Date", False, "Date of the first observation."),
    "end_date": ("sc:Date", False, "Date of the last observation."),
    "n_points": ("sc:Integer", False, "Length of the dates/values/flags lists."),
    "n_obs": ("sc:Integer", False, "Non-missing values. n_points minus n_obs is the number of gaps."),
    "n_flagged": ("sc:Integer", False, "Observations carrying a status flag."),
    # the data
    "dates": ("sc:Date", True, "Observation dates, ascending. Same length as values and flags."),
    "values": ("sc:Float", True, "Observed values, null where the source published a flag but no number."),
    "flags": ("sc:Text", True, "Per-observation status: provisional, estimated, break in series, and so on."),
}

LIMITATIONS = (
    "Vintages: every observation is the most recent revision, not the figure available at the "
    "time. A value labelled 2024 may have been restated in 2026 using later information, so this "
    "corpus cannot support a genuine real-time forecasting exercise without external vintage data. "
    "Duplication: the three sources republish one another (FRED alone carries 80,274 OECD-origin "
    "series), 78% of datasets publish an aggregate alongside its parts, and many indicators appear "
    "at several frequencies. A train/test split assigned at random over series is therefore "
    "contaminated; split on whole datasets, themes or countries. "
    "Length: series counts are not a measure of how much data there is. Dimension cross-products "
    "generate very many very short series, and the median series in some large collections holds "
    "two observations."
)

COLLECTION = (
    "Downloaded from each agency's public API with a single paced connection per source, honouring "
    "rate limits and Retry-After, and recorded so an interrupted crawl resumes rather than "
    "re-requesting. Payloads are kept as served; the tidy layers are derived from them offline."
)


def sha256(path: Path) -> str:
    """Checksum of a file, streamed so a multi-gigabyte shard does not have to fit in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(manifest: dict, *, url: str | None = None, cite_as: str | None = None,
          name: str | None = None, index_sha256: str | None = None) -> dict:
    """The Croissant record for a snapshot, from its manifest.

    ``index_sha256`` is required by the specification for any ``cr:FileObject`` — the reference
    validator rejects the record without it — so ``write()`` computes it from the file on disk
    rather than letting the caller assert one.
    """
    lic = manifest["counts"].get("license_id", {})
    src = manifest["counts"].get("source", {})
    freq = manifest["counts"].get("frequency", {})
    cols = manifest.get("columns") or list(FIELDS)
    snap = name or manifest["name"]
    public = manifest["selection"].get("public_only")

    lic_lines = []
    for lid, n in sorted(lic.items(), key=lambda kv: -kv[1]):
        who, terms, u = LICENSE_SUMMARY.get(lid, (lid, "See the source's terms.", ""))
        lic_lines.append(f"{who} ({n:,} series): {terms}".strip())
    # schema.org licence wants a single value; a mixed corpus has to say so rather than pick one
    license_value = (
        "Mixed, per series. " + " ".join(lic_lines) +
        (" Every series here may be redistributed with acknowledgement."
         if public else
         " NO LICENCE FILTER WAS APPLIED: some series may not be redistributable. "
         "Check the license_id column.")
    )

    description = (
        f"{manifest['rows']:,} public economic time series from "
        f"{', '.join(sorted(src))}, one row per series with dates and values as parallel lists "
        f"and {len(cols)} columns of metadata carrying each series' own licence, attribution and "
        f"provenance. Frequencies: {', '.join(f'{k} ({v:,})' for k, v in sorted(freq.items(), key=lambda kv: -kv[1]))}. "
        f"Built with terrastat {manifest.get('terrastat_version', __version__)}."
    )

    fields = []
    for c in cols:
        dtype, repeated, doc = FIELDS.get(c, ("sc:Text", False, ""))
        f = {
            "@type": "cr:Field",
            "@id": f"series/{c}",
            "name": c,
            "description": doc,
            "dataType": dtype,
            "source": {"fileSet": {"@id": "shards"}, "extract": {"column": c}},
        }
        if repeated:
            f["repeated"] = True
        fields.append(f)

    return {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": CROISSANT_VERSION,
        "name": snap,
        "description": description,
        "version": manifest.get("terrastat_version", __version__),
        "datePublished": manifest.get("created_at"),
        "license": license_value,
        "url": url or "https://github.com/USER/terrastat",
        "citeAs": cite_as or (
            "Rosales Saiz, I. and Montero-Manso, P. terrastat: public economic time series from "
            "FRED, Eurostat and the OECD. See CITATION.cff. Each series additionally carries the "
            "citation its own source requires, in the attribution column."
        ),
        "keywords": ["time series", "forecasting", "economics", "FRED", "Eurostat", "OECD"],
        "isLiveDataset": False,
        "rai:dataCollection": COLLECTION,
        "rai:dataLimitations": LIMITATIONS,
        "rai:dataUseCases": (
            "Training and evaluating forecasting models, including global models trained across "
            "many series; hierarchical forecasting and reconciliation, since aggregates and their "
            "parts are both present and identifiable; and imputation, since gaps are marked "
            "rather than filled."
        ),
        "distribution": [
            {
                "@type": "cr:FileSet",
                "@id": "shards",
                "name": "shards",
                "description": (
                    f"{manifest['n_shards']} Parquet shards of about "
                    f"{manifest['bytes'] / max(manifest['n_shards'], 1) / 1e6:.0f} MB. Every series is "
                    "assigned to a shard at random and rows are shuffled inside each, so any single "
                    "shard is a uniform sample of the whole snapshot."
                ),
                "encodingFormat": "application/x-parquet",
                "includes": "shard-*.parquet",
            },
            {
                "@type": "cr:FileObject",
                "@id": "index",
                "name": "index.parquet",
                "description": "series_uid to shard and row, with source, dataset, frequency, licence and length.",
                "encodingFormat": "application/x-parquet",
                "contentUrl": "index.parquet",
                **({"sha256": index_sha256} if index_sha256 else {}),
            },
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "series",
                "name": "series",
                "description": (
                    "One record is one complete time series. dates, values and flags are "
                    "equal-length parallel lists in date order; values may be null where the "
                    "source published a status flag without a number."
                ),
                "key": {"@id": "series/series_uid"},
                "field": fields,
            }
        ],
    }


def write(out: Path, manifest: dict, **kw) -> Path:
    """Write ``croissant.json`` beside the shards."""
    out = Path(out)
    idx = out / "index.parquet"
    if idx.exists() and "index_sha256" not in kw:
        kw["index_sha256"] = sha256(idx)
    rec = build(manifest, **kw)
    path = Path(out) / "croissant.json"
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s (%d fields)", path, len(rec["recordSet"][0]["field"]))
    return path
