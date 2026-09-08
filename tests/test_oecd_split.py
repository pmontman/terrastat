"""Splitting an OECD dataflow that is too large for one request, without touching the network."""
import gzip
import json

import httpx
import polars as pl
import pytest

from terrastat.http import TooLarge
from terrastat.sources.base import DatasetRef
from terrastat.sources.oecd import OecdSource, _tidy_csv

HEADER = "STRUCTURE,STRUCTURE_ID,STRUCTURE_NAME,ACTION,REF_AREA,Reference area,MEASURE,Measure,TIME_PERIOD,Time period,OBS_VALUE,Observation value\n"


def _rows(area, n=2):
    return "".join(f"DATAFLOW,X:Y(1.0),D,I,{area},Area {area},EXP,Expenditure,{2000+i},,{i + 1}.5,\n" for i in range(n))


class FakeClient:
    """Serves the split requests from a fixed set of areas; /all is always too large."""

    def __init__(self, areas, oversized=()):
        self.areas = set(areas)
        self.oversized = set(oversized)
        self.requests = []
        self.stop_check = None
        self.on_progress = None

    def download(self, url, dest, params=None, headers=None, max_bytes=None):
        self.requests.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith("/all?format=csvfilewithlabels"):
            raise TooLarge("whole dataflow exceeded the cap")
        key = url.rsplit("/", 1)[1].split("?")[0]
        codes = [c for c in key.split(".")[0].split("+")]
        present = [c for c in codes if c in self.areas]
        if not present:
            raise httpx.HTTPStatusError("no results", request=None, response=httpx.Response(404))
        if any(c in self.oversized for c in present) and len(present) > 0:
            raise TooLarge(f"{dest.name} exceeded the part cap")
        dest.write_text(HEADER + "".join(_rows(a) for a in present), encoding="utf-8")
        return dest


@pytest.fixture()
def ref():
    return DatasetRef(source="oecd", dataset_id="X__Y__1.0", title="t", extra={"agency": "X", "flow": "Y", "version": "1.0"})


def _seed_structure(tmp_path, codes):
    d = tmp_path / "cache" / "oecd" / "structure"
    d.mkdir(parents=True)
    (d / "X__Y__1.0.json").write_text(json.dumps({"dims": ["REF_AREA", "MEASURE", "TIME_PERIOD"], "split_dim": "REF_AREA", "split_codes": codes}), encoding="utf-8")


def test_split_fetches_only_areas_with_data(tmp_path, monkeypatch, ref):
    monkeypatch.setenv("TERRASTAT_DATA_DIR", str(tmp_path))
    codes = [f"C{i:02d}" for i in range(20)]
    _seed_structure(tmp_path, codes)
    have = {"C00", "C01", "C11"}
    client = FakeClient(have)
    parts = OecdSource().fetch_raw(ref, client)
    assert parts, "the split must produce parts"
    # /all once, then one request per batch of 8: 20 codes -> 3 batches
    assert client.requests[0].endswith("/all?format=csvfilewithlabels")
    assert len(client.requests) == 4
    manifest = json.loads((tmp_path / "raw" / "oecd" / "X__Y__1.0" / "_parts.json").read_text(encoding="utf-8"))
    assert manifest["complete"] and sorted(manifest["done_codes"]) == codes
    out = tmp_path / "obs.parquet"
    res = _tidy_csv(parts, out)
    got = set(pl.read_parquet(out).get_column("REF_AREA").unique().to_list())
    assert got == have
    assert res["n_series"] == len(have)


def test_split_halves_a_batch_that_is_still_too_large(tmp_path, monkeypatch, ref):
    monkeypatch.setenv("TERRASTAT_DATA_DIR", str(tmp_path))
    codes = [f"C{i:02d}" for i in range(8)]
    _seed_structure(tmp_path, codes)
    # C03 is huge: the batch of 8 fails, then 4, then 2, until C03 is alone and skipped
    client = FakeClient(set(codes), oversized={"C03"})
    parts = OecdSource().fetch_raw(ref, client)
    manifest = json.loads((tmp_path / "raw" / "oecd" / "X__Y__1.0" / "_parts.json").read_text(encoding="utf-8"))
    assert manifest["skipped"] == ["C03"]
    assert sorted(manifest["done_codes"]) == codes
    out = tmp_path / "obs.parquet"
    _tidy_csv(parts, out)
    got = set(pl.read_parquet(out).get_column("REF_AREA").unique().to_list())
    assert got == set(codes) - {"C03"}, "every area except the oversized one is kept"


def test_split_resumes_from_the_manifest(tmp_path, monkeypatch, ref):
    monkeypatch.setenv("TERRASTAT_DATA_DIR", str(tmp_path))
    codes = [f"C{i:02d}" for i in range(16)]
    _seed_structure(tmp_path, codes)
    client = FakeClient(set(codes))
    OecdSource().fetch_raw(ref, client)
    first = len(client.requests)
    client2 = FakeClient(set(codes))
    OecdSource().fetch_raw(ref, client2)
    assert client2.requests == [], "a completed split re-uses the parts on disk"
    assert first > 1


def test_an_external_dataflow_is_refused_before_any_request(tmp_path, monkeypatch):
    """27 dataflows (TiVA, the DAC/CRS aid statistics, three archived ones) are registered in the
    public catalogue but defined in another OECD space. Their data endpoint answers HTTP 500 with
    a .NET null-reference message for every key and format, so the request must not be made."""
    from terrastat.sources.base import NoData

    monkeypatch.setenv("TERRASTAT_DATA_DIR", str(tmp_path))

    class NoNetwork:
        def download(self, *a, **k):
            raise AssertionError("an external dataflow must not reach the network")

    ref = DatasetRef(
        source="oecd",
        dataset_id="OECD.DCD.FSD__DSD_CRS_DF_CRS__1.6",
        title="Creditor Reporting System",
        extra={
            "agency": "OECD.DCD.FSD", "flow": "DSD_CRS@DF_CRS", "version": "1.6",
            "external": True,
            "structure_url": "https://sdmx.oecd.org/dcd-public/rest/dataflow/OECD.DCD.FSD/DSD_CRS@DF_CRS/1.6",
        },
    )
    with pytest.raises(NoData) as exc:
        OecdSource().fetch_raw(ref, NoNetwork())
    assert "dcd-public" in str(exc.value)


def test_a_known_oversized_dataflow_skips_straight_to_slices(tmp_path, monkeypatch):
    """Re-running a too_large dataflow must not spend the whole cap (20 GB) rediscovering that it
    is too large: the marker left by the first attempt sends it straight to the split path."""
    monkeypatch.setenv("TERRASTAT_DATA_DIR", str(tmp_path))
    from terrastat.storage import raw_dir, write_json

    did = "X__Y__1.0"
    write_json(raw_dir("oecd", did) / "_oversized.json", {"reason": "abandoned at the cap"})

    src = OecdSource()
    client = FakeClient(areas=["AUS", "BEL"])
    ref = DatasetRef(source="oecd", dataset_id=did, title="T",
                     extra={"agency": "X", "flow": "Y", "version": "1.0"})
    monkeypatch.setattr(src, "_structure", lambda r, c: {"dims": ["REF_AREA", "MEASURE"], "split_dim": "REF_AREA", "split_codes": ["AUS", "BEL"]})
    parts = src.fetch_raw(ref, client)

    assert parts, "the split should have produced parts"
    assert not any("/all?" in u for u in client.requests), f"the whole-dataflow request was made anyway: {client.requests}"
