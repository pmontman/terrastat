"""Finding series by subject rather than by statistics.

Everything else in this package asks *how good* a series is. This asks *what it is about*, which
is the question you have when you want the poultry data, or fish landings by species, or anything
to do with prices, and you do not yet know which of 8,000 datasets holds it.

The search runs over an index built from the ``dataset.json`` files, one per dataset, and that
choice is the whole design. Series titles live in a 27 GB tree of 956 million rows; dataset
metadata is a few thousand small files holding the title, the description, every dimension and
**every code label inside it**. Searching those answers the question in under a second and, more
importantly, answers it better: "chicken" is not in the title of Eurostat's ``apro_ec_poulm``,
which is called *Poultry farming*, but it is one of the code labels of its ``animals`` dimension.
A title-only search would miss the dataset you actually wanted.

So a match reports *where* it matched, and when it matched on a code, which code. Once a dataset
looks right, ``series_of`` pulls its series out of the series layer.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import polars as pl
from tqdm import tqdm

from terrastat.storage import data_dir

log = logging.getLogger(__name__)

INDEX_SCHEMA = {
    "source": pl.String,
    "dataset_id": pl.String,
    "title": pl.String,
    "description": pl.String,
    "dimensions": pl.String,     # "id=name" for each, joined
    "labels": pl.String,         # every code label, joined; this is what makes "chicken" findable
    "n_series": pl.Int64,
    "n_obs": pl.Int64,
    "frequencies": pl.String,
    "time_start": pl.String,
    "time_end": pl.String,
    "source_url": pl.String,
}


def index_path(root: Path | str | None = None) -> Path:
    return (Path(root) if root else data_dir()) / "index" / "datasets.parquet"


def build_index(root: Path | str | None = None, force: bool = False) -> Path:
    """Read every ``dataset.json`` into one searchable table.

    Cheap enough to redo whenever the crawl has moved on: a few thousand small files, seconds.
    """
    out = index_path(root)
    base = (Path(root) if root else data_dir()) / "datasets"
    if out.exists() and not force:
        return out
    rows = []
    files = sorted(base.rglob("dataset.json"))
    for f in tqdm(files, unit="dataset", dynamic_ncols=True, smoothing=0):
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:                      # a half-written file from an interrupted run
            log.warning("skipping %s: %s", f, exc)
            continue
        dims = m.get("dimensions") or []
        labels = []
        for d in dims:
            labels.extend(str(v) for v in (d.get("codes") or {}).values())
        rows.append({
            "source": m.get("source") or f.parent.parent.name,
            "dataset_id": m.get("dataset_id") or f.parent.name,
            "title": m.get("title") or "",
            "description": (m.get("description") or "")[:4000],
            "dimensions": " | ".join(f"{d.get('id')}={d.get('name')}" for d in dims),
            "labels": " | ".join(labels)[:200_000],
            "n_series": m.get("n_series"),
            "n_obs": m.get("n_obs"),
            "frequencies": ",".join(m.get("frequencies") or []),
            "time_start": str(m.get("time_start") or ""),
            "time_end": str(m.get("time_end") or ""),
            "source_url": m.get("source_url") or "",
        })
    df = pl.DataFrame(rows, schema=INDEX_SCHEMA)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out, compression="zstd")
    log.info("indexed %d datasets to %s", df.height, out)
    return out


def load_index(root: Path | str | None = None) -> pl.DataFrame:
    p = index_path(root)
    if not p.exists():
        build_index(root)
    return pl.read_parquet(p)


def search(query: str, root: Path | str | None = None, sources: list[str] | None = None,
           frequencies: list[str] | None = None, in_labels: bool = True,
           min_series: int = 1) -> pl.DataFrame:
    """Datasets whose title, description, dimension names or code labels match ``query``.

    ``query`` is a regular expression, so alternatives work: ``"chicken|poultry|egg"``. Matching is
    case-insensitive.

    ``where`` in the result says which field matched, and ``matched_labels`` shows the first few
    code labels that did — the difference between "this dataset is about fish" and "this dataset
    has a species dimension containing tilapia".
    """
    idx = load_index(root)
    if sources:
        idx = idx.filter(pl.col("source").is_in(sources))
    if frequencies:
        pat = "|".join(frequencies)
        idx = idx.filter(pl.col("frequencies").str.contains(pat))
    if min_series:
        idx = idx.filter(pl.col("n_series").fill_null(0) >= min_series)

    q = f"(?i){query}"
    fields = ["title", "description", "dimensions"] + (["labels"] if in_labels else [])
    hit = {f: pl.col(f).str.contains(q) for f in fields}
    matched = idx.filter(pl.any_horizontal(list(hit.values())))
    where = pl.concat_str(
        [pl.when(hit[f]).then(pl.lit(f)).otherwise(pl.lit("")) for f in fields], separator=" ",
    ).str.strip_chars()
    # Relevance, not size. Sorting by series count buries the dataset actually about the subject
    # under every large one that merely mentions it: searching "poultry" put a 2.3 million-series
    # structural business statistics table above Eurostat's poultry farming data, because one of
    # its NACE codes is "Processing and preserving of poultry meat". A title match is a statement
    # about what the dataset *is*; a code label is a statement about something it contains.
    score = (
        pl.when(hit["title"]).then(4).otherwise(0)
        + pl.when(hit["dimensions"]).then(2).otherwise(0)
        + pl.when(hit["description"]).then(1).otherwise(0)
        + (pl.when(hit["labels"]).then(1).otherwise(0) if in_labels else 0)
    )
    return (
        matched.with_columns(
            where.alias("where"),
            score.alias("relevance"),
            # show why it matched: the code labels that hit, not the whole codelist
            pl.col("labels").str.split(" | ").list.eval(
                pl.element().filter(pl.element().str.contains(q))
            ).list.head(4).list.join("; ").alias("matched_labels"),
        )
        .drop("labels", "description")
        .sort(["relevance", "n_series"], descending=[True, True])
    )


def series_of(dataset_ids: list[str], root: Path | str | None = None,
              columns: list[str] | None = None) -> pl.LazyFrame:
    """The series belonging to these datasets, straight from the series layer."""
    from terrastat.quality import scan

    cols = columns or ["series_uid", "source", "dataset_id", "frequency", "title",
                       "start_date", "end_date", "n_obs"]
    return scan(root, columns=list(dict.fromkeys([*cols, "dataset_id"]))).filter(
        pl.col("dataset_id").is_in(dataset_ids)
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="terrastat search", description=__doc__.splitlines()[0])
    p.add_argument("query", help="regular expression, case-insensitive, e.g. \"chicken|poultry\"")
    p.add_argument("--sources", nargs="*", default=None)
    p.add_argument("--freq", nargs="*", default=None, help="only datasets published at these frequencies")
    p.add_argument("--min-series", type=int, default=1)
    p.add_argument("--titles-only", action="store_true", help="do not search dimension code labels")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--rebuild", action="store_true", help="rebuild the index before searching")
    p.add_argument("--root", default=None)
    args = p.parse_args(argv)

    if args.rebuild:
        build_index(args.root, force=True)
    hits = search(args.query, args.root, args.sources, args.freq,
                  in_labels=not args.titles_only, min_series=args.min_series)
    print(f"{hits.height} datasets match {args.query!r}\n")
    for r in hits.head(args.limit).iter_rows(named=True):
        span = f"{r['time_start'][:7]}..{r['time_end'][:7]}" if r["time_start"] else ""
        print(f"{r['source']:9} {r['dataset_id'][:44]:44} {str(r['n_series'] or 0):>10} series  "
              f"{r['frequencies'] or '?':<7} {span}")
        print(f"          {r['title'][:96]}")
        if r["matched_labels"]:
            print(f"          matched {r['where']}: {r['matched_labels'][:96]}")
        elif r["where"]:
            print(f"          matched {r['where']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
