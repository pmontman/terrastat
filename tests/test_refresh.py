"""Offline refresh integration tests: fresh inputs, exact comparisons and safe publication."""
import datetime as dt
import json
import os
from pathlib import Path

import polars as pl
import pytest

from terrastat import config, refresh, series, storage
from terrastat.http import StopRequested
from terrastat.sources.base import CATALOG_SCHEMA, DatasetMeta, DatasetRef, DatasetResult, Source

_OBS_SCHEMA = {"series_key": pl.String, "freq": pl.String, "geo": pl.String,
               "period": pl.String, "date": pl.Date, "value": pl.Float64, "flag": pl.String}


def _observations(values=(1.0, None, 3.0, 4.0), flags=(None, "c", None, None)):
    return pl.DataFrame([
        ("Q.AT", "Q", "AT", f"2020-Q{i + 1}", dt.date(2020, 1 + i * 3, 1), value, flag)
        for i, (value, flag) in enumerate(zip(values, flags))
    ], schema=_OBS_SCHEMA, orient="row")


class FakeClient:
    def __init__(self):
        self.stop_check = None
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def close(self):
        self.closed = True

    def stop_requested(self):
        return bool(self.stop_check and self.stop_check())


class FakeSource(Source):
    """A cached raw payload really is reused unless refresh supplies a clean workspace."""
    def __init__(self, name):
        self.name = name
        self.current = {"alpha": 0, "beta": 0}
        self.payloads = {(did, 0): (_observations(), f"Dataset {did}") for did in self.current}
        self.tokens = {did: "v1" for did in self.current}
        self.failure = None
        self.catalog_calls = 0
        self.fetch_calls = []
        self.downloads = []
        self.cache_hits = 0
        self.clients = []
        self.client_options = []

    def make_client(self, min_interval=None, max_per_minute=None):
        client = FakeClient()
        self.clients.append(client)
        self.client_options.append((min_interval, max_per_minute))
        return client

    def catalog(self, client):
        self.catalog_calls += 1
        return pl.DataFrame([
            {"dataset_id": did, "title": self.payloads[(did, self.current[did])][1],
             "frequencies": ["Q"], "n_values": 4, "last_updated": self.tokens[did], "extra": "{}"}
            for did in sorted(self.current)
        ], schema=CATALOG_SCHEMA)

    def refresh_token(self, ref):
        return self.tokens[ref.dataset_id]

    def update(self, did="alpha", *, observations=None, title=None, token="v2"):
        previous, previous_title = self.payloads[(did, self.current[did])]
        self.current[did] += 1
        self.payloads[(did, self.current[did])] = (
            previous if observations is None else observations,
            previous_title if title is None else title,
        )
        self.tokens[did] = token

    def reset_calls(self):
        self.fetch_calls.clear()
        self.downloads.clear()
        self.cache_hits = 0
        self.catalog_calls = 0

    def fetch_raw(self, ref, client):
        path = storage.raw_dir(self.name, ref.dataset_id) / "payload.json"
        if path.exists():
            self.cache_hits += 1
        else:
            storage.write_json(path, {"generation": self.current[ref.dataset_id]})
            self.downloads.append((ref.dataset_id, config.data_dir()))
        return [path]

    def fetch_and_tidy(self, ref, client, keep_raw=True):
        self.fetch_calls.append((ref.dataset_id, config.data_dir()))
        raw = self.fetch_raw(ref, client)[0]
        directory = storage.dataset_dir(self.name, ref.dataset_id)
        if self.failure is not None:
            storage.write_json(directory / "partial.json", {"incomplete": True})
            raise self.failure
        generation = storage.read_json(raw)["generation"]
        frame, title = self.payloads[(ref.dataset_id, generation)]
        storage.write_parquet(frame, directory / "observations.parquet")
        meta = DatasetMeta(
            source=self.name, dataset_id=ref.dataset_id, title=title,
            source_url=f"https://example.invalid/{self.name}/{ref.dataset_id}",
            api_url=f"https://example.invalid/api/{ref.dataset_id}",
            license={"id": "eurostat-reuse", "name": "Test source terms",
                     "url": "https://example.invalid/terms", "attribution": "Test source", "notes": ""},
            frequencies=["Q"], n_series=1, n_obs=frame.height,
            dimensions=[{"id": "freq", "name": "Frequency", "codes": {"Q": "Quarterly"}},
                        {"id": "geo", "name": "Geography", "codes": {"AT": "Austria"}}],
            last_updated=self.tokens[ref.dataset_id],
            retrieved_at=f"2026-09-22T00:00:{len(self.fetch_calls):02d}+00:00",
            raw_files=[raw.relative_to(config.data_dir()).as_posix()] if keep_raw else [],
        )
        storage.write_json(directory / "dataset.json", meta.to_dict())
        if not keep_raw:
            raw.unlink()
        return DatasetResult(ref.dataset_id, 1, frame.height, ["Q"])


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("TERRASTAT_DATA_DIR", str(tmp_path))
    adapters = {name: FakeSource(name) for name in ("eurostat", "oecd")}
    monkeypatch.setattr(refresh, "get_source", lambda name: adapters[name])
    monkeypatch.setattr(series, "get_source", lambda name: adapters[name])

    def seed(name="eurostat", did="alpha", *, record_state=True, build=True):
        source = adapters[name]
        ref = DatasetRef(name, did, f"Dataset {did}", ["Q"])
        with source.make_client() as client:
            source.fetch_and_tidy(ref, client)
        if record_state:
            storage.State(name).mark(did, "done", n_series=1, n_obs=4)
        if build:
            series.build_dataset_series(name, did, min_obs=1, force=True)
        source.reset_calls()
        frozen = tmp_path / "snapshot" / "starter" / "shard-00000.parquet"
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(b"immutable existing snapshot")

    return tmp_path, adapters, seed


def _protected(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for folder in ("datasets", "raw", "series", "state", "snapshot")
            for path in (root / folder).rglob("*") if path.is_file() and not path.name.endswith(".lock")}


def _one(report):
    assert report["checked"] == 1
    assert len(report["results"]) == 1
    return report["results"][0]


def _saved_report(report):
    path = Path(report["report"])
    assert path.is_file()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["updated"] == report["updated"] and saved["failed"] == report["failed"]
    return saved


def test_first_refresh_downloads_before_a_stable_catalog_token_can_skip(world):
    root, adapters, seed = world
    seed()
    source = adapters["eurostat"]
    before = _protected(root)
    first = refresh.run_refresh(sources=["eurostat"], ids=["alpha"])
    assert _one(first)["method"] == "download_compare"
    assert first["unchanged"] == 1 and first["updated"] == first["failed"] == 0
    assert len(source.downloads) == 1 and source.cache_hits == 0
    assert _protected(root) == before
    assert (root / "refresh" / "checks" / "eurostat" / "alpha.json").is_file()
    second = refresh.run_refresh(sources=["eurostat"], ids=["alpha"])
    assert _one(second)["method"] == "catalog_token"
    assert second["unchanged"] == 1 and len(source.fetch_calls) == 1
    assert source.catalog_calls == 2
    assert _protected(root) == before
    _saved_report(first)
    _saved_report(second)


def test_force_downloads_fresh_despite_matching_token_and_cached_raw(world):
    root, adapters, seed = world
    seed()
    source = adapters["eurostat"]
    refresh.run_refresh(sources=["eurostat"], ids=["alpha"])
    source.update(observations=_observations(values=(10.0, None, 3.0, 4.0)), token="v1")
    source.reset_calls()
    report = refresh.run_refresh(sources=["eurostat"], ids=["alpha"], force=True)
    result = _one(report)
    assert result["status"] == "updated" and result["method"] == "forced"
    assert result["comparison"]["value_revisions"] == 1
    assert len(source.downloads) == 1 and source.cache_hits == 0
    assert all(workspace != root for _, workspace in source.fetch_calls)
    assert config.data_dir() == root
    assert pl.read_parquet(root / "datasets/eurostat/alpha/observations.parquet")["value"][0] == 10.0
    metadata = storage.read_json(root / "datasets/eurostat/alpha/dataset.json")
    assert metadata["raw_files"]
    assert all((root / path).is_file() and (root / path).is_relative_to(root / "raw") for path in metadata["raw_files"])
    assert (root / "snapshot/starter/shard-00000.parquet").read_bytes() == b"immutable existing snapshot"


def test_sources_without_a_reliable_token_download_compare_every_time(world):
    root, adapters, seed = world
    source = adapters["eurostat"]
    source.tokens["alpha"] = None
    seed()
    before = _protected(root)
    for _ in range(2):
        report = refresh.run_refresh(sources=["eurostat"], ids=["alpha"])
        assert _one(report)["method"] == "download_compare" and report["unchanged"] == 1
    assert len(source.downloads) == 2 and source.cache_hits == 0
    assert _protected(root) == before


def test_refresh_reports_added_removed_revised_flags_and_preserves_missing_in_derived_data(world):
    root, adapters, seed = world
    seed()
    source = adapters["eurostat"]
    new = pl.DataFrame([
        ("Q.AT", "Q", "AT", "2020-Q1", dt.date(2020, 1, 1), 2.0, "p"),
        ("Q.AT", "Q", "AT", "2020-Q2", dt.date(2020, 4, 1), None, None),
        ("Q.AT", "Q", "AT", "2020-Q3", dt.date(2020, 7, 1), None, "c"),
        ("Q.AT", "Q", "AT", "2021-Q1", dt.date(2021, 1, 1), 5.0, None),
    ], schema=_OBS_SCHEMA, orient="row")
    source.update(observations=new)
    result = _one(refresh.run_refresh(sources=["eurostat"], ids=["alpha"]))
    counts = result["comparison"]
    assert {k: counts[k] for k in ("added", "removed", "revised", "value_revisions", "flag_revisions")} == {
        "added": 1, "removed": 1, "revised": 3, "value_revisions": 2, "flag_revisions": 3,
    }
    derived = pl.read_parquet(root / "series/eurostat/freq=Q/alpha.parquet")
    assert derived["values"][0].to_list() == [2.0, None, None, 5.0]
    assert derived["dates"][0].to_list() == new["date"].to_list()
    assert storage.State("eurostat").status("alpha") == "done"
    assert storage.State("eurostat-series").status("alpha") == "done"


def test_metadata_only_change_rebuilds_the_series_description(world):
    root, adapters, seed = world
    seed()
    adapters["eurostat"].update(title="A clearer revised dataset title")
    report = refresh.run_refresh(sources=["eurostat"], ids=["alpha"])
    result = _one(report)
    assert report["updated"] == 1 and result["metadata_changed"] is True
    assert all(result["comparison"][k] == 0 for k in ("added", "removed", "revised"))
    derived = pl.read_parquet(root / "series/eurostat/freq=Q/alpha.parquet")
    assert derived["dataset_title"][0] == "A clearer revised dataset title"
    assert "A clearer revised dataset title" in derived["description"][0]


@pytest.mark.parametrize("component", ["datasets/eurostat/alpha/dataset.json", "raw/eurostat/alpha/payload.json",
                                       "series/eurostat/freq=Q/alpha.parquet"])
def test_local_file_fingerprint_changes_invalidate_a_token_skip(world, component):
    root, adapters, seed = world
    seed()
    source = adapters["eurostat"]
    refresh.run_refresh(sources=["eurostat"], ids=["alpha"])
    source.reset_calls()
    changed = root / component
    info = changed.stat()
    os.utime(changed, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
    result = _one(refresh.run_refresh(sources=["eurostat"], ids=["alpha"]))
    assert result["method"] == "download_compare" and len(source.downloads) == 1


@pytest.mark.parametrize("failure", ["fetch", "build", "compare"])
def test_failure_retains_all_previous_data_state_snapshot_and_receipt(world, monkeypatch, failure):
    root, adapters, seed = world
    seed()
    source = adapters["eurostat"]
    refresh.run_refresh(sources=["eurostat"], ids=["alpha"])
    before = _protected(root)
    receipt_path = root / "refresh/checks/eurostat/alpha.json"
    receipt = receipt_path.read_bytes()
    source.update(observations=_observations(values=(9.0, None, 3.0, 4.0)))
    if failure == "fetch":
        source.failure = RuntimeError("simulated fetch failure after a partial staged write")
    elif failure == "compare":
        source.update(observations=pl.concat([_observations(), _observations().head(1)]))
    else:
        def failed_build(*args, **kwargs):
            raise RuntimeError("simulated derived build failure")
        monkeypatch.setattr(series, "build_dataset_series", failed_build)
        if hasattr(refresh, "build_dataset_series"):
            monkeypatch.setattr(refresh, "build_dataset_series", failed_build)
    report = refresh.run_refresh(sources=["eurostat"], ids=["alpha"])
    assert report["failed"] == 1 and _one(report)["status"] == "failed"
    assert _protected(root) == before
    assert receipt_path.read_bytes() == receipt
    assert config.data_dir() == root and all(client.closed for client in source.clients)
    _saved_report(report)


def test_manually_populated_normalized_data_does_not_require_a_fetch_state_event(world):
    root, adapters, seed = world
    seed(record_state=False, build=False)
    assert not (root / "state/eurostat.jsonl").exists()
    adapters["eurostat"].update(observations=_observations(values=(2.0, None, 3.0, 4.0)))
    report = refresh.run_refresh(sources=["eurostat"], ids=["alpha"])
    assert report["updated"] == 1
    assert (root / "series/eurostat/freq=Q/alpha.parquet").is_file()
    assert storage.State("eurostat").status("alpha") == "done"


def test_unknown_requested_dataset_is_rejected_without_changing_existing_data(world):
    root, adapters, seed = world
    seed()
    before = _protected(root)
    with pytest.raises(ValueError, match="unknown|Unknown|not found|not downloaded|not local"):
        refresh.run_refresh(sources=["eurostat"], ids=["not_a_dataset"])
    assert not adapters["eurostat"].fetch_calls and _protected(root) == before


def test_source_selection_and_dataset_limit_bound_the_work(world):
    _, adapters, seed = world
    seed("eurostat", "alpha")
    seed("eurostat", "beta")
    seed("oecd", "alpha")
    report = refresh.run_refresh(sources=["eurostat"], limit=1, min_interval=0.25, max_per_minute=7)
    assert report["checked"] == 1 and len(adapters["eurostat"].fetch_calls) == 1
    assert not adapters["oecd"].fetch_calls and adapters["oecd"].catalog_calls == 0
    assert (0.25, 7) in adapters["eurostat"].client_options


@pytest.mark.parametrize("exception,interrupted", [(StopRequested("time budget exhausted"), False),
                                                  (KeyboardInterrupt(), True)])
def test_stop_and_interrupt_save_a_report_and_restore_context_without_partial_publication(world, exception, interrupted):
    root, adapters, seed = world
    seed()
    before = _protected(root)
    adapters["eurostat"].failure = exception
    report = refresh.run_refresh(sources=["eurostat"], ids=["alpha"])
    assert report["stopped_early"] is True and report["interrupted"] is interrupted
    assert _protected(root) == before and config.data_dir() == root
    assert all(client.closed for client in adapters["eurostat"].clients)
    _saved_report(report)


def test_changed_publication_establishes_a_baseline_that_can_skip_next_time(world):
    root, adapters, seed = world
    seed()
    adapters["eurostat"].update(observations=_observations(values=(8.0, None, 3.0, 4.0)))
    first = refresh.run_refresh(sources=["eurostat"])
    assert first["updated"] == 1
    second = refresh.run_refresh(sources=["eurostat"])
    assert _one(second)["method"] == "catalog_token"
    assert len(adapters["eurostat"].downloads) == 1


@pytest.mark.parametrize("mode", ["missing", "error"])
def test_fresh_catalog_failure_retains_data_and_does_not_expose_credentials(world, monkeypatch, mode):
    root, adapters, seed = world
    seed()
    before = _protected(root)
    source = adapters["eurostat"]
    if mode == "missing":
        del source.current["alpha"]
    else:
        def unavailable(client):
            raise RuntimeError("https://example.invalid?api_key=private-secret")
        monkeypatch.setattr(source, "catalog", unavailable)
    report = refresh.run_refresh(sources=["eurostat"])
    assert report["failed"] == 1 and not source.fetch_calls
    assert "private-secret" not in json.dumps(report)
    assert _protected(root) == before


@pytest.mark.parametrize("checkpoint", [False, True])
def test_compact_deployment_refresh_does_not_retain_raw_downloads(world, checkpoint):
    import shutil
    root, adapters, seed = world
    seed(record_state=False, build=False)
    raw = (root / "raw").resolve()
    assert raw.is_relative_to(root.resolve())
    shutil.rmtree(raw)
    if checkpoint:
        storage.write_json(root / "raw/eurostat/alpha/_pages.json", {"complete": True})
    path = root / "datasets/eurostat/alpha/dataset.json"
    meta = storage.read_json(path)
    meta["raw_files"] = []
    storage.write_json(path, meta)
    adapters["eurostat"].update(observations=_observations(values=(2.0, None, 3.0, 4.0)))
    report = refresh.run_refresh(sources=["eurostat"])
    assert report["updated"] == 1
    assert not list((root / "raw/eurostat/alpha").glob("payload*"))
    assert storage.read_json(path)["raw_files"] == []
    assert _one(refresh.run_refresh(sources=["eurostat"]))["method"] == "catalog_token"


def test_retrieval_dates_in_citations_do_not_create_metadata_revisions():
    old = {"source": "oecd", "retrieved_at": "2025-12-31", "license": {
        "attribution": "OECD (2025), Data (accessed on 2025-12-31).", "notes": "Unchanged terms"}}
    new = {"source": "oecd", "retrieved_at": "2026-01-01", "license": {
        "attribution": "OECD (2026), Data (accessed on 2026-01-01).", "notes": "Unchanged terms"}}
    assert refresh._metadata(old) == refresh._metadata(new)
    new["license"]["notes"] = "Revised terms"
    assert refresh._metadata(old) != refresh._metadata(new)


def test_data_workspace_restores_data_and_reference_cache_on_failure(tmp_path):
    original_root, original_cache = config.data_dir(), config.reference_cache_dir()
    with pytest.raises(RuntimeError):
        with config.data_workspace(tmp_path / "stage", cache=tmp_path / "fresh-cache"):
            assert config.data_dir() == tmp_path / "stage"
            assert storage.cache_dir("eurostat") == tmp_path / "fresh-cache/eurostat"
            raise RuntimeError("interrupted adapter")
    assert config.data_dir() == original_root and config.reference_cache_dir() == original_cache


def test_unknown_local_normalized_files_are_not_deleted_by_refresh(world):
    root, adapters, seed = world
    seed()
    (root / "datasets/eurostat/alpha/research-notes.txt").write_text("Keep my notes")
    before = _protected(root)
    adapters["eurostat"].update(title="Updated title")
    result = _one(refresh.run_refresh(sources=["eurostat"]))
    assert result["status"] == "failed" and "unmanaged files" in result["error"]
    assert _protected(root) == before


def test_catalog_checks_use_a_fresh_cache_without_overwriting_the_live_one(world, monkeypatch):
    root, adapters, seed = world
    seed()
    path = root / "cache/eurostat/toc.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("old cached catalogue")
    source = adapters["eurostat"]
    original = source.catalog
    def fresh_catalog(client):
        staged = storage.cache_dir("eurostat") / "toc.xml"
        assert staged != path and not staged.exists()
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text("fresh catalogue")
        return original(client)
    monkeypatch.setattr(source, "catalog", fresh_catalog)
    assert refresh.run_refresh(sources=["eurostat"])["unchanged"] == 1
    assert path.read_text() == "old cached catalogue"
