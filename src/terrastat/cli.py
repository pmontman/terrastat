"""Command line: terrastat catalog | fetch | series | status | peek."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from terrastat.config import data_dir
from terrastat.sources import SOURCES


def _setup_logging(verbose: bool, quiet_console: bool = False) -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252; titles are UTF-8
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logs = data_dir() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    console = logging.StreamHandler(sys.stderr)
    if quiet_console and not verbose:
        console.setLevel(logging.WARNING)  # keep the progress bars readable; the log file has everything
    handlers = [console, logging.FileHandler(logs / "terrastat.log", encoding="utf-8")]
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format=fmt, handlers=handlers)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="terrastat", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="the simple way: fetch everything from every source, all frequencies, build the series view; re-run to resume")
    r.add_argument("--sources", nargs="*", choices=SOURCES, default=None, help="default: all three")
    r.add_argument("--max-hours", type=float, default=None, help="stop cleanly after this many hours (re-run to continue)")
    r.add_argument("--sequential", action="store_true", help="one source after another instead of one connection per source side by side")
    r.add_argument("--no-series", action="store_true", help="skip building the per-series view")
    r.add_argument("--no-tags", action="store_true", help="skip the FRED tags step that follows the FRED releases")
    r.add_argument("--min-obs", type=int, default=1)
    r.add_argument("--no-raw", action="store_true", help="delete raw payloads after tidying")
    r.add_argument("--max-download-gb", type=float, default=None, help="abandon any single download past this size and record it as too_large (OECD default 20)")

    tg = sub.add_parser("tags", help="FRED tags: list all tags, then the series of each tag (resumable); attach them to the series view")
    tg.add_argument("source", choices=["fred"])
    tg.add_argument("--groups", nargs="*", default=None, help="tag groups to fetch: gen geo geot src rls, or all (default: gen geo geot)")
    tg.add_argument("--limit-tags", type=int, default=None, help="stop after this many tags (tests)")
    tg.add_argument("--max-hours", type=float, default=None)
    tg.add_argument("--min-interval", type=float, default=2.0)
    tg.add_argument("--max-per-minute", type=int, default=30)
    tg.add_argument("--no-rebuild", action="store_true", help="do not rebuild the FRED series view afterwards")

    c = sub.add_parser("catalog", help="download (or show) what a source offers")
    c.add_argument("source", choices=SOURCES)
    c.add_argument("--refresh", action="store_true", help="re-download the catalogue")
    c.add_argument("--freq", nargs="*", help="only count datasets with these frequencies (M A Q ...)")
    c.add_argument("--csv", metavar="PATH", help="also write the full list (id, title, frequencies, size, status) to this CSV")

    f = sub.add_parser("fetch", help="download datasets one by one, politely, writing a usable folder after each")
    f.add_argument("source", choices=SOURCES)
    f.add_argument("--freq", nargs="*", help="canonical frequencies to include (e.g. M A Q). Default: all")
    f.add_argument("--ids", nargs="*", help="only these dataset ids")
    f.add_argument("--max-values", type=int, default=None, help="skip datasets with more stored values than this (size hint from the catalogue)")
    f.add_argument("--types", nargs="*", default=None, help="eurostat only: dataset and/or table (default: dataset)")
    f.add_argument("--order", choices=["size", "catalog"], default="size", help="smallest first (default) or catalogue order")
    f.add_argument("--limit", type=int, default=None, help="stop after this many new datasets")
    f.add_argument("--max-hours", type=float, default=None, help="stop cleanly after this many hours (resume later)")
    f.add_argument("--min-interval", type=float, default=None, help="seconds between requests (source default if omitted)")
    f.add_argument("--max-per-minute", type=int, default=None, help="cap on requests per minute (source default if omitted)")
    f.add_argument("--raw-only", action="store_true", help="download the raw payload only (one request per dataset); run fetch again without this flag to tidy from the local files")
    f.add_argument("--no-raw", action="store_true", help="delete raw payloads after tidying")
    f.add_argument("--force", action="store_true", help="redo datasets already done")
    f.add_argument("--retry-failed", action="store_true", help="retry datasets that failed before")
    f.add_argument("--with-series", action="store_true", help="also build the per-series view right after each dataset")
    f.add_argument("--min-obs", type=int, default=1, help="with --with-series: drop series with fewer non-missing values")

    s = sub.add_parser("series", help="build the one-row-per-series view from the dataset folders")
    s.add_argument("source", choices=SOURCES)
    s.add_argument("--ids", nargs="*")
    s.add_argument("--min-obs", type=int, default=1)
    s.add_argument("--force", action="store_true")

    st = sub.add_parser("status", help="progress of a source")
    st.add_argument("source", choices=SOURCES)

    pk = sub.add_parser("peek", help="show one dataset folder")
    pk.add_argument("source", choices=SOURCES)
    pk.add_argument("dataset_id")

    q = sub.add_parser("quality", help="how much data there is and how usable it is: length, recency, gaps, constants")
    q.add_argument("--sources", nargs="*", choices=SOURCES, default=None)
    q.add_argument("--freq", nargs="*", default=None, help="canonical frequencies, e.g. M Q A")
    q.add_argument("--asof", default=None, help="reference date for recency (default: today; set it to the crawl date to measure publication lag alone)")
    q.add_argument("--until", default="2025-12-01", help="cutoff date for the 'reaches' column")
    q.add_argument("--no-deep", action="store_true", help="skip the sampled value-level pass (constants, gap position)")
    q.add_argument("--sample", type=int, default=200_000, help="series per source/frequency in the deep pass")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--datasets", action="store_true", help="also break the numbers down per dataset, not just per source")
    q.add_argument("--origin", default=None, help="forecast time for the aligned cohort (default: the most recent 31 December)")
    q.add_argument("--folds", type=int, default=3, help="rolling-origin folds to report the aligned cohort for")
    q.add_argument("--history-years", type=float, default=1.0, help="history required before the earliest origin (default 1)")
    q.add_argument("--out", default=None, help="directory to write the tables (parquet + csv) and summary.md into")

    co = sub.add_parser("corpus", help="regenerate docs/corpus.md: the one-page summary of how much data there is and what shape it is in")
    co.add_argument("--out", default="docs/corpus.md", help="where to write the markdown annex")
    co.add_argument("--html", default="docs/index.html", help="where to write the project page")
    co.add_argument("--no-html", action="store_true", help="write only the markdown")
    co.add_argument("--sources", nargs="*", choices=SOURCES, default=None)
    co.add_argument("--freq", nargs="*", default=None, help="canonical frequencies, e.g. M Q A")
    co.add_argument("--years", type=float, default=2.0, help="the length bar, in years of each series' own frequency")
    co.add_argument("--sample", type=int, default=100_000, help="series per frequency for the sampled value statistics")
    co.add_argument("--no-values", action="store_true", help="skip the sampled value pass (distinctness, sign)")
    co.add_argument("--no-concentration", action="store_true", help="skip the per-dataset pass")
    co.add_argument("--tables", default="docs/corpus_tables.json", help="where to keep the computed figures")
    co.add_argument("--check", action="store_true",
                    help="do not read the corpus: re-render from --tables and fail if the committed pages differ")
    co.add_argument("--render", action="store_true",
                    help="do not read the corpus: rewrite both pages from --tables (seconds, not minutes)")

    hi = sub.add_parser("hierarchy", help="where the corpus is hierarchical: series that are exact sums of other series")
    hi.add_argument("--sources", nargs="*", choices=SOURCES, default=None)
    hi.add_argument("--dataset", default=None, help="inspect one dataset instead of summarising all")
    hi.add_argument("--dim", default=None, help="with --dataset: the dimension holding the total, e.g. sex")
    hi.add_argument("--check", action="store_true", help="with --dataset and --dim: verify parent == sum(children)")
    hi.add_argument("--out", default=None, help="write the dataset-level table to this parquet path")

    se = sub.add_parser("search", help="find datasets by subject: title, description, dimensions and every code label")
    se.add_argument("query", help='regular expression, case-insensitive, e.g. "chicken|poultry"')
    se.add_argument("--sources", nargs="*", choices=SOURCES, default=None)
    se.add_argument("--freq", nargs="*", default=None, help="only datasets published at these frequencies")
    se.add_argument("--min-series", type=int, default=1)
    se.add_argument("--titles-only", action="store_true", help="do not search dimension code labels")
    se.add_argument("--limit", type=int, default=25)
    se.add_argument("--rebuild", action="store_true", help="rebuild the index first (after a crawl has moved on)")

    sel = sub.add_parser("select", help="score series for information and pick the best on recency, length and information together")
    sel.add_argument("--sources", nargs="*", choices=SOURCES, default=None)
    sel.add_argument("--freq", nargs="*", default=None)
    sel.add_argument("--asof", default=None, help="reference date for recency (default: today)")
    sel.add_argument("--min-obs", type=int, default=3, help="technical floor: fewer points and the features are undefined")
    sel.add_argument("--max-lag", type=float, default=None, help="OPTIONAL cost cut: skip series staler than this (off by default; recency is a ranking axis)")
    sel.add_argument("--min-horizons", type=float, default=None, help="OPTIONAL cost cut: skip series shorter than this many horizons")
    sel.add_argument("--max-missing", type=float, default=None, help="OPTIONAL cost cut: skip series with more missing than this")
    sel.add_argument("--floor", action="store_true", help="also apply a hard degeneracy cut before ranking (off by default)")
    sel.add_argument("--min-entropy", type=float, default=0.3, help="with --floor: minimum normalised entropy")
    sel.add_argument("--max-repeat", type=float, default=0.5, help="with --floor: maximum share of consecutive equal values")
    sel.add_argument("--max-flat-run", type=float, default=0.3, help="with --floor: maximum share in one carried-forward run")
    sel.add_argument("--min-rank", type=float, default=None, help="keep series above this percentile on every ranked axis")
    sel.add_argument("--out", default=None, help="where to keep the scores (default: data/scores)")
    sel.add_argument("--force", action="store_true", help="rescore files already done")
    sel.add_argument("--report", action="store_true", help="skip scoring; summarise the scores already on disk")

    ex = sub.add_parser("export", help="repack the series view into fixed-size shuffled Parquet shards for training")
    ex.add_argument("name", help="snapshot name; written to data/snapshot/<name>/")
    ex.add_argument("--sources", nargs="*", choices=SOURCES, default=None)
    ex.add_argument("--freq", nargs="*", default=None)
    ex.add_argument("--min-obs", type=int, default=1)
    ex.add_argument("--license-ids", nargs="*", default=None, help="keep only these license ids")
    ex.add_argument("--public-only", action="store_true", help="Eurostat, OECD, and FRED public-domain series from US federal sources")
    ex.add_argument("--target-mb", type=float, default=128.0, help="approximate shard size (default 128)")
    ex.add_argument("--seed", type=int, default=0)
    ex.add_argument("--float64", action="store_true", help="keep values as float64 (default: float32)")
    ex.add_argument("--columns", nargs="*", default=None, help="subset of series columns to keep")
    ex.add_argument("--limit-rows", type=int, default=None, help="for tests: stop after this many series (truncates in catalogue order, so not a sample)")
    ex.add_argument("--per-dataset", type=int, default=None, help="sample at most this many series from each dataset; use this, not --limit-rows, for a representative subset")

    args = p.parse_args(argv)
    _setup_logging(args.verbose, quiet_console=(args.cmd == "run"))
    from terrastat import pipeline

    if args.cmd == "run":
        sources = args.sources or list(SOURCES)
        if args.max_download_gb:
            from terrastat.sources.oecd import OecdSource

            OecdSource.max_download_bytes = int(args.max_download_gb * 1e9)
        print(f"terrastat run: {', '.join(sources)}; all frequencies; {'sequential' if args.sequential else 'one connection per source'};"
              f" budget {args.max_hours or 'unlimited'} h. Ctrl+C stops within seconds; re-run to resume. Log: data/logs/terrastat.log")
        for s in sources:
            pipeline.load_catalog(s)  # download the catalogues up front so the bars start with real counts
        results = pipeline.run_all(
            sources=sources,
            max_hours=args.max_hours,
            with_series=not args.no_series,
            min_obs=args.min_obs,
            keep_raw=not args.no_raw,
            parallel=not args.sequential,
            with_tags=not args.no_tags,
        )
        print(json.dumps(results, indent=2, default=str))
        print("\nwhere things stand now:")
        for s in sources:
            st = pipeline.status(s)
            print(f"  {s}: {st['done']} done, {st['failed']} failed, {st['remaining']} remaining of {st['catalogue_datasets']}")
        return 0

    if args.cmd == "catalog":
        cat = pipeline.load_catalog(args.source, refresh=args.refresh)
        if args.freq:
            refs = pipeline.select_refs(args.source, cat, freqs=args.freq)
            print(f"{len(refs)} datasets with frequencies {args.freq}")
        print(cat.select("dataset_id", "title", "frequencies", "n_values", "last_updated").head(20))
        print(f"{cat.height} entries in the {args.source} catalogue ({pipeline.catalog_path(args.source)})")
        if args.csv:
            import polars as pl

            from terrastat.storage import State

            state = State(args.source)
            listing = cat.with_columns(
                pl.col("frequencies").list.join(" ").alias("frequencies"),
                pl.col("dataset_id").map_elements(lambda d: state.status(d) or "todo", return_dtype=pl.String).alias("status"),
            ).select("dataset_id", "title", "frequencies", "n_values", "last_updated", "status")
            listing.write_csv(args.csv)
            print(f"wrote {listing.height} rows to {args.csv}")
        return 0

    if args.cmd == "fetch":
        cat = pipeline.load_catalog(args.source)
        types = args.types
        if args.source == "eurostat" and types is None:
            types = ["dataset"]
        refs = pipeline.select_refs(args.source, cat, freqs=args.freq, ids=args.ids, max_values=args.max_values, types=types, order=args.order)
        summary = pipeline.run_fetch(
            args.source,
            refs,
            min_interval=args.min_interval,
            max_per_minute=args.max_per_minute,
            keep_raw=not args.no_raw,
            limit=args.limit,
            force=args.force,
            retry_failed=args.retry_failed,
            max_hours=args.max_hours,
            with_series=args.with_series,
            min_obs=args.min_obs,
            raw_only=args.raw_only,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["failed"] == 0 else 1

    if args.cmd == "series":
        from terrastat.series import build_series

        print(json.dumps(build_series(args.source, dataset_ids=args.ids, min_obs=args.min_obs, force=args.force), indent=2))
        return 0

    if args.cmd == "status":
        print(json.dumps(pipeline.status(args.source), indent=2, default=str))
        return 0

    if args.cmd == "peek":
        print(pipeline.peek(args.source, args.dataset_id))
        return 0

    if args.cmd == "tags":
        from terrastat.tags import run_tags

        res = run_tags(
            groups=args.groups,
            limit_tags=args.limit_tags,
            min_interval=args.min_interval,
            max_per_minute=args.max_per_minute,
            max_hours=args.max_hours,
            rebuild_series=not args.no_rebuild,
        )
        print(json.dumps(res, indent=2, default=str))
        return 0

    if args.cmd == "hierarchy":
        from terrastat.hierarchy import main as hier_main

        argv2 = []
        if args.sources:
            argv2 += ["--sources", *args.sources]
        for flag, val in (("--dataset", args.dataset), ("--dim", args.dim), ("--out", args.out)):
            if val:
                argv2 += [flag, str(val)]
        if args.check:
            argv2 += ["--check"]
        if args.render:
            argv2 += ["--render"]
        return hier_main(argv2)

    if args.cmd == "search":
        from terrastat.search import main as search_main

        argv2 = [args.query, "--limit", str(args.limit), "--min-series", str(args.min_series)]
        if args.sources:
            argv2 += ["--sources", *args.sources]
        if args.freq:
            argv2 += ["--freq", *args.freq]
        for flag in ("titles_only", "rebuild"):
            if getattr(args, flag):
                argv2 += ["--" + flag.replace("_", "-")]
        return search_main(argv2)

    if args.cmd == "select":
        from terrastat.selection import main as select_main

        argv2 = []
        for flag, val in (("--asof", args.asof), ("--out", args.out)):
            if val:
                argv2 += [flag, str(val)]
        if args.sources:
            argv2 += ["--sources", *args.sources]
        if args.freq:
            argv2 += ["--freq", *args.freq]
        for flag in ("force", "report", "floor"):
            if getattr(args, flag):
                argv2 += [f"--{flag}"]
        argv2 += ["--min-obs", str(args.min_obs), "--min-entropy", str(args.min_entropy),
                  "--max-repeat", str(args.max_repeat), "--max-flat-run", str(args.max_flat_run)]
        for flag, val in (("--max-lag", args.max_lag), ("--min-horizons", args.min_horizons),
                          ("--max-missing", args.max_missing), ("--min-rank", args.min_rank)):
            if val is not None:
                argv2 += [flag, str(val)]
        return select_main(argv2)

    if args.cmd == "quality":
        from terrastat.quality import main as quality_main

        argv2 = []
        for flag, val in (("--asof", args.asof), ("--until", args.until), ("--out", args.out)):
            if val:
                argv2 += [flag, str(val)]
        if args.sources:
            argv2 += ["--sources", *args.sources]
        if args.freq:
            argv2 += ["--freq", *args.freq]
        if args.no_deep:
            argv2 += ["--no-deep"]
        if args.datasets:
            argv2 += ["--datasets"]
        if args.origin:
            argv2 += ["--origin", args.origin]
        argv2 += ["--folds", str(args.folds), "--history-years", str(args.history_years)]
        argv2 += ["--sample", str(args.sample), "--seed", str(args.seed)]
        return quality_main(argv2)

    if args.cmd == "corpus":
        from terrastat.corpus import main as corpus_main

        argv2 = ["--out", args.out, "--html", args.html,
                 "--years", str(args.years), "--sample", str(args.sample)]
        if args.no_html:
            argv2 += ["--no-html"]
        if args.sources:
            argv2 += ["--sources", *args.sources]
        if args.freq:
            argv2 += ["--frequencies", *args.freq]
        if args.no_values:
            argv2 += ["--no-values"]
        if args.no_concentration:
            argv2 += ["--no-concentration"]
        argv2 += ["--tables", args.tables]
        if args.check:
            argv2 += ["--check"]
        if args.render:
            argv2 += ["--render"]
        return corpus_main(argv2)

    if args.cmd == "export":
        from terrastat.export import export_shards

        m = export_shards(
            args.name,
            sources=args.sources,
            freqs=args.freq,
            min_obs=args.min_obs,
            license_ids=args.license_ids,
            public_only=args.public_only,
            target_mb=args.target_mb,
            seed=args.seed,
            float32=not args.float64,
            columns=args.columns,
            limit_rows=args.limit_rows,
            per_file_cap=args.per_dataset,
        )
        print(json.dumps({k: m[k] for k in ("name", "n_shards", "rows", "bytes", "counts")}, indent=2))
        print(m["licensing_notice"])
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
