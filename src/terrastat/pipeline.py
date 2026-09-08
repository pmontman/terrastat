"""Walk a source's catalogue politely, one dataset at a time, and leave a usable folder after each.

Everything here is resumable: the catalogue is cached, finished datasets are recorded in a state
file and skipped next time, failures are recorded with their error, and a time budget lets a run
stop cleanly after N hours so it can be resumed later. ``run_all`` strings it all together for
every source at once, one connection per source.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import traceback

import polars as pl
from tqdm import tqdm

from terrastat.http import StopRequested, TooLarge
from terrastat.sources import SOURCES, get_source
from terrastat.sources.base import DatasetRef, NoData, NotTimeSeries, ref_from_row
from terrastat.storage import State, catalog_path, dataset_dir, read_json, write_parquet

log = logging.getLogger(__name__)


def load_catalog(source_name: str, refresh: bool = False, client=None) -> pl.DataFrame:
    path = catalog_path(source_name)
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    src = get_source(source_name)
    own = client is None
    client = client or src.make_client()
    try:
        df = src.catalog(client)
    finally:
        if own:
            client.close()
    write_parquet(df, path)
    return df


def select_refs(
    source_name: str,
    catalog: pl.DataFrame,
    freqs: list[str] | None = None,
    ids: list[str] | None = None,
    max_values: int | None = None,
    types: list[str] | None = None,
    order: str = "size",
) -> list[DatasetRef]:
    """Filter and order the catalogue. Unknown frequencies are kept when a freq filter is given
    only if the source cannot tell frequencies up front (FRED, OECD)."""
    df = catalog
    if ids:
        df = df.filter(pl.col("dataset_id").is_in(ids))
    if freqs:
        wanted = [f.upper() for f in freqs]
        df = df.filter(pl.col("frequencies").list.eval(pl.element().is_in(wanted)).list.any() | (pl.col("frequencies").list.len() == 0))
    if max_values is not None:
        df = df.filter(pl.col("n_values").is_null() | (pl.col("n_values") <= max_values))
    if types:
        df = df.filter(pl.col("extra").str.json_path_match("$.type").is_in(types) | pl.col("extra").str.json_path_match("$.type").is_null())
    if order == "size":
        df = df.sort("n_values", nulls_last=True)
    return [ref_from_row(source_name, row) for row in df.iter_rows(named=True)]


def run_fetch(
    source_name: str,
    refs: list[DatasetRef],
    min_interval: float | None = None,
    max_per_minute: int | None = None,
    keep_raw: bool = True,
    limit: int | None = None,
    force: bool = False,
    retry_failed: bool = False,
    max_hours: float | None = None,
    with_series: bool = False,
    min_obs: int = 1,
    raw_only: bool = False,
    deadline: float | None = None,
    stop: threading.Event | None = None,
    bar_position: int | None = None,
) -> dict:
    """Fetch every dataset in ``refs`` that is not done yet. Stops at ``max_hours`` (from now),
    at ``deadline`` (a ``time.monotonic()`` value) or when ``stop`` is set; re-running resumes."""
    src = get_source(source_name)
    state = State(source_name)
    t_start = time.monotonic()
    if max_hours is not None:
        deadline = min(deadline or float("inf"), t_start + max_hours * 3600)
    done = state.ids_with("done")
    failed = state.ids_with("failed")
    too_large = state.ids_with("too_large")
    not_ts = state.ids_with("not_timeseries")
    no_data = state.ids_with("no_data")
    raw_done = state.ids_with("raw")
    todo = []
    for r in refs:
        if r.dataset_id in done and not force:
            continue
        if r.dataset_id in too_large and not force:
            continue  # a bigger --max-download-gb or a per-dimension split is needed, not a retry
        if r.dataset_id in not_ts and not force:
            continue  # there is no time series in there; retrying cannot change that
        if r.dataset_id in no_data and not force:
            continue  # the source itself says it holds no observations for this one
        if r.dataset_id in failed and not (retry_failed or force):
            continue
        if r.dataset_id in raw_done and raw_only and not force:
            continue
        todo.append(r)
    if limit:
        todo = todo[:limit]
    log.info(
        "%s: %d datasets selected, %d done, %d raw only, %d failed, %d to do%s",
        source_name, len(refs), len(done), len(raw_done), len(failed), len(todo), " (raw only)" if raw_only else "",
    )
    summary = {"done": 0, "raw": 0, "failed": 0, "skipped": len(refs) - len(todo), "series": 0, "observations": 0, "stopped_early": False}
    if not todo:
        return summary
    def should_stop() -> bool:
        return (deadline is not None and time.monotonic() > deadline) or (stop is not None and stop.is_set())

    with src.make_client(min_interval, max_per_minute) as client:
        client.stop_check = should_stop
        # the bar counts over the whole selection, starting from what earlier runs finished
        # smoothing=0 makes the rate a plain average over the run rather than an EMA weighted to
        # the last item. Dataset sizes here differ by four orders of magnitude, and one five-hour
        # split was enough to push tqdm's estimate to 46 days remaining when the truth was hours.
        bar = tqdm(todo, unit="dataset", dynamic_ncols=True, position=bar_position, leave=True,
                   total=len(refs), initial=len(refs) - len(todo), smoothing=0)
        # a multi-GB dataset can take half an hour: show the bytes so it never looks stuck
        client.on_progress = lambda n: bar.set_postfix_str(f"{n / 1e6:,.0f} MB" if n else "", refresh=True)
        for ref in bar:
            if should_stop():
                log.info("%s: stopping cleanly (time budget or stop request); re-run to resume", source_name)
                summary["stopped_early"] = True
                break
            bar.set_description(f"{source_name}:{ref.dataset_id[:28]}")
            t0 = time.monotonic()
            try:
                if raw_only:
                    files = src.fetch_raw(ref, client)
                    state.mark(ref.dataset_id, "raw", files=len(files), seconds=round(time.monotonic() - t0, 1))
                    summary["raw"] += 1
                    continue
                res = src.fetch_and_tidy(ref, client, keep_raw=keep_raw)
                state.mark(ref.dataset_id, "done", n_series=res.n_series, n_obs=res.n_obs, frequencies=res.frequencies, seconds=round(time.monotonic() - t0, 1))
                summary["done"] += 1
                summary["series"] += res.n_series
                summary["observations"] += res.n_obs
                if with_series:
                    from terrastat.series import build_dataset_series

                    build_dataset_series(source_name, ref.dataset_id, min_obs=min_obs, force=True)
            except StopRequested as exc:
                log.info("%s: %s", source_name, exc)
                summary["stopped_early"] = True
                break
            except NoData as exc:
                log.info("%s: %s -- skipped", ref.dataset_id, exc)
                state.mark(ref.dataset_id, "no_data", error=str(exc)[:300])
                summary["no_data"] = summary.get("no_data", 0) + 1
            except NotTimeSeries as exc:
                log.info("%s: %s -- skipped", ref.dataset_id, exc)
                state.mark(ref.dataset_id, "not_timeseries", error=str(exc)[:300])
                summary["not_timeseries"] = summary.get("not_timeseries", 0) + 1
            except TooLarge as exc:
                log.warning("%s: %s -- needs splitting by dimension; skipped", ref.dataset_id, exc)
                state.mark(ref.dataset_id, "too_large", error=str(exc)[:300], seconds=round(time.monotonic() - t0, 1))
                summary["too_large"] = summary.get("too_large", 0) + 1
            except KeyboardInterrupt:
                log.warning("interrupted; %s left incomplete and will be redone next run", ref.dataset_id)
                raise
            except Exception as exc:  # noqa: BLE001 - record and move on
                log.error("%s failed: %s", ref.dataset_id, exc)
                log.debug(traceback.format_exc())
                state.mark(ref.dataset_id, "failed", error=f"{type(exc).__name__}: {str(exc)[:300]}", seconds=round(time.monotonic() - t0, 1))
                summary["failed"] += 1
        bar.close()
        log.info("%s transfer stats: %s", source_name, client.stats.summary())
    return summary


def default_refs(source_name: str) -> list[DatasetRef]:
    """Everything a source offers, all frequencies (Eurostat: datasets, not their derived tables)."""
    cat = load_catalog(source_name)
    types = ["dataset"] if source_name == "eurostat" else None
    return select_refs(source_name, cat, types=types)


def run_all(
    sources: list[str] | None = None,
    max_hours: float | None = None,
    with_series: bool = True,
    min_obs: int = 1,
    keep_raw: bool = True,
    parallel: bool = True,
    retry_failed: bool = True,
    with_tags: bool = True,
) -> dict:
    """Fetch every dataset of every source, all frequencies, then build the series view.

    One thread per source (each with its own single paced connection), a shared time budget and a
    shared stop signal: Ctrl+C stops every source within seconds. Run the same command again to
    continue; when ``status`` shows nothing to do, it is done. For FRED, the tags step runs once
    the releases are complete (same connection budget), and the FRED series view is rebuilt with
    the tags attached.
    """
    sources = list(sources or SOURCES)
    deadline = time.monotonic() + max_hours * 3600 if max_hours else None
    stop = threading.Event()
    results: dict[str, dict] = {}

    def work(i: int, name: str) -> None:
        try:
            refs = default_refs(name)
            res = run_fetch(name, refs, keep_raw=keep_raw, with_series=with_series, min_obs=min_obs, deadline=deadline, stop=stop, bar_position=i)
            if retry_failed and not stop.is_set() and not res.get("stopped_early") and State(name).ids_with("failed"):
                log.info("%s: retrying failed datasets once", name)
                res2 = run_fetch(name, refs, keep_raw=keep_raw, with_series=with_series, min_obs=min_obs, retry_failed=True, deadline=deadline, stop=stop, bar_position=i)
                res["retry"] = {k: res2[k] for k in ("done", "failed")}
            if with_series and not stop.is_set():
                from terrastat.series import build_series

                res["series_build"] = build_series(name, min_obs=min_obs)
            if name == "fred" and with_tags and not stop.is_set() and not res.get("stopped_early"):
                from terrastat.tags import run_tags

                res["tags"] = run_tags(deadline=deadline, stop=stop, rebuild_series=with_series, bar_position=i)
            results[name] = res
        except Exception as exc:  # noqa: BLE001
            log.error("%s: run aborted: %s", name, exc)
            log.debug(traceback.format_exc())
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}

    if parallel and len(sources) > 1:
        threads = [threading.Thread(target=work, args=(i, name), name=f"terrastat-{name}", daemon=True) for i, name in enumerate(sources)]
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(timeout=0.5)
        except KeyboardInterrupt:
            stop.set()
            log.warning("stopping: each source finishes the dataset it is on, then exits (re-run to resume)")
            for t in threads:
                t.join()
    else:
        for i, name in enumerate(sources):
            try:
                work(i, name)
            except KeyboardInterrupt:
                stop.set()
                break
    return {name: results.get(name, {"error": "not started"}) for name in sources}


def status(source_name: str) -> dict:
    state = State(source_name)
    done = state.ids_with("done")
    failed = state.ids_with("failed")
    n_series = sum(int(state.events[d].get("n_series", 0) or 0) for d in done)
    n_obs = sum(int(state.events[d].get("n_obs", 0) or 0) for d in done)
    freq_counts: dict[str, int] = {}
    for d in done:
        for f in state.events[d].get("frequencies", []) or []:
            freq_counts[f] = freq_counts.get(f, 0) + 1
    cat = catalog_path(source_name)
    total = pl.read_parquet(cat).height if cat.exists() else None
    if cat.exists():
        types = pl.read_parquet(cat).select(pl.col("extra").str.json_path_match("$.type").alias("t"))
        if source_name == "eurostat":
            total = int(types.filter(pl.col("t") == "dataset").height)
    too_large = state.ids_with("too_large")
    not_ts = state.ids_with("not_timeseries")
    no_data = state.ids_with("no_data")
    remaining = None if total is None else max(0, total - len(done) - len(failed) - len(too_large) - len(not_ts) - len(no_data))
    extra = {}
    if source_name == "fred":
        from terrastat.tags import tags_status

        extra["tags"] = tags_status()
    return {
        **extra,
        "source": source_name,
        "catalogue_datasets": total,
        "done": len(done),
        "raw_only": len(state.ids_with("raw")),
        "failed": len(failed),
        "too_large": len(too_large),
        "not_timeseries": len(not_ts),
        "no_data": len(no_data),
        "remaining": remaining,
        "series": n_series,
        "observations": n_obs,
        "datasets_by_frequency": dict(sorted(freq_counts.items())),
        "failures": {d: state.events[d].get("error") for d in sorted(failed)[:20]},
    }


def peek(source_name: str, dataset_id: str, n: int = 8) -> str:
    d = dataset_dir(source_name, dataset_id)
    meta = read_json(d / "dataset.json")
    lines = [json.dumps({k: v for k, v in meta.items() if k not in ("dimensions", "attributes", "extra")}, indent=2, ensure_ascii=False, default=str)]
    for dim in meta.get("dimensions", []):
        codes = dim["codes"]
        sample = dict(list(codes.items())[:5])
        lines.append(f"dimension {dim['id']} ({dim['name']}): {len(codes)} codes, e.g. {sample}")
    obs = d / "observations.parquet"
    if obs.exists():
        lines.append(str(pl.read_parquet(obs).head(n)))
    ser = d / "series.parquet"
    if ser.exists():
        lines.append(str(pl.read_parquet(ser).head(n)))
    return "\n".join(lines)
