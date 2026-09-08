"""OECD Data Explorer, through the SDMX REST API at sdmx.oecd.org (no key needed).

Catalogue : ``/public/rest/dataflow/all`` lists every dataflow with agency, id, version and name.
            Dataflows owned by other agencies (mirrors of Eurostat data, for instance) are skipped.
Data      : ``/public/rest/data/<agency>,<id>,<version>/all?format=csvfilewithlabels`` gives one
            CSV with, for every dimension and attribute, a code column followed by a label column.

Each dataflow is pulled in one request and tidied by streaming, so memory stays flat whatever the
size: the CSV is read in batches, written to Parquet, then sorted by series with a streaming pass
that spills to disk.

Some dataflows are too big for one request even so (the education-finance one runs past 20 GB
because it holds roughly 460 MB per country). Past ``max_download_bytes`` the transfer is
abandoned and the dataflow is re-fetched a slice at a time, keyed on the first dimension, which
for OECD is almost always the reference area. Slices start as batches of codes and halve whenever
one is still too big, so a bad guess costs one part rather than the whole dataflow; codes with no
data answer 404 and are skipped; progress is checkpointed after every part. The parts share a
schema and are concatenated by the tidy step, so the result is indistinguishable from a
single-request fetch.

Licence: OECD Terms and Conditions (data section): extraction, adaptation and distribution for any
purpose including commercial, with the citation given in the metadata; third-party data excepted.
"""
from __future__ import annotations

import csv
import gzip
import json
import logging
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import httpx
import polars as pl
import pyarrow as pa
from pyarrow import csv as pa_csv

from terrastat.http import PoliteClient, TooLarge
from terrastat.sources.base import (
    CATALOG_SCHEMA,
    DatasetMeta,
    DatasetRef,
    DatasetResult,
    Dimension,
    License,
    NoData,
    NotTimeSeries,
    Source,
    safe_dim_name,
)
from terrastat.storage import (
    ParquetBatchWriter,
    cache_dir,
    dataset_dir,
    is_fresh,
    now_iso,
    raw_dir,
    read_json,
    safe_id,
    write_json,
)
from terrastat.timeparse import canonical_frequency, frequency_of_period, period_to_date

log = logging.getLogger(__name__)

API = "https://sdmx.oecd.org/public/rest"
DATAFLOWS_URL = f"{API}/dataflow/all"
DATA_URL = f"{API}/data/{{agency}},{{flow}},{{version}}/all?format=csvfilewithlabels"
LANDING = "https://data-explorer.oecd.org/vis?df[ds]=dsDisseminateFinalDMZ&df[id]={flow}&df[ag]={agency}&df[vs]={version}"

NS_STR = "{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure}"
NS_COM = "{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common}"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

LICENSE = License(
    id="oecd-terms",
    name="OECD Terms and Conditions, Data: use, adaptation and distribution for any purpose with citation",
    url="https://www.oecd.org/en/about/terms-conditions.html",
    attribution="OECD ({year}), {title} ({dataset_id}), {url} (accessed on {date}).",
    notes=(
        "Data may be fully or partially owned by third parties or carry additional restrictions; the "
        "OECD asks users to check the dataset's metadata ('source' tab) before incorporating it."
    ),
)

LEADING = ("STRUCTURE", "STRUCTURE_ID", "STRUCTURE_NAME", "ACTION")
ATTR_KEEP = ("OBS_STATUS", "CONF_STATUS", "UNIT_MULT", "DECIMALS")
SPLIT_BATCH = 8  # codes of the first dimension per request when a dataflow has to be split


class OecdSource(Source):
    name = "oecd"
    default_min_interval = 3.0
    default_max_per_minute = 12
    # A runaway guard, not a quality filter. The "Local areas" drilldowns are legitimately huge
    # (Population by age and sex streams 6.6 GB of CSV for 1.25M series, and compresses to 270 MB
    # on disk), so the cap sits well above them; tidying is streamed, so size costs time, not
    # memory. Past the cap a transfer is abandoned, nothing is left behind, and the dataflow is
    # recorded as too_large so a later run can split it by dimension.
    max_download_bytes = 20_000_000_000
    # when splitting, each part is capped much lower so a mis-sized batch wastes little
    split_part_bytes = 4_000_000_000

    def catalog(self, client: PoliteClient) -> pl.DataFrame:
        path = cache_dir(self.name) / "dataflows.xml"
        if not is_fresh(path):
            client.download(DATAFLOWS_URL, path)
        root = ET.parse(path).getroot()
        rows = []
        for f in root.iter(f"{NS_STR}Dataflow"):
            agency, flow, version = f.get("agencyID") or "", f.get("id") or "", f.get("version") or ""
            if not agency.upper().startswith("OECD"):
                continue  # dataflows owned by other agencies are mirrors; fetch them at the source
            names = {n.get(XML_LANG, "en"): (n.text or "").strip() for n in f.findall(f"{NS_COM}Name")}
            descs = {n.get(XML_LANG, "en"): (n.text or "").strip() for n in f.findall(f"{NS_COM}Description")}
            ann = {}
            for a in f.iter(f"{NS_COM}Annotation"):
                t = a.find(f"{NS_COM}AnnotationType")
                if t is not None and t.text:
                    txt = a.find(f"{NS_COM}AnnotationText")
                    title = a.find(f"{NS_COM}AnnotationTitle")
                    ann[t.text] = (txt.text if txt is not None else None) or (title.text if title is not None else "")
            rows.append(
                {
                    "dataset_id": safe_id(f"{agency}__{flow}__{version}"),
                    "title": names.get("en") or next(iter(names.values()), ""),
                    "frequencies": [],
                    "n_values": None,
                    "last_updated": None,
                    "extra": json.dumps(
                        {
                            "agency": agency,
                            "flow": flow,
                            "version": version,
                            "description": descs.get("en", ""),
                            "annotations": {k: str(v)[:500] for k, v in ann.items() if v},
                            # a flow whose definition lives in another OECD space (sti-public,
                            # dcd-public, archive). The public data endpoint cannot serve these:
                            # see fetch_raw
                            "external": f.get("isExternalReference") == "true",
                            "structure_url": f.get("structureURL") or "",
                        },
                        ensure_ascii=False,
                    ),
                }
            )
        df = pl.DataFrame(rows, schema=CATALOG_SCHEMA)
        log.info("oecd catalogue: %d OECD-owned dataflows", df.height)
        return df

    def fetch_raw(self, ref: DatasetRef, client: PoliteClient) -> list:
        """The whole dataflow in one request, or, if that is too big, one part per key slice."""
        did = ref.dataset_id
        if ref.extra.get("external"):
            # The catalogue registers 27 dataflows (TiVA, the DAC/CRS aid statistics, a few
            # archived ones) whose definition lives in another OECD space. Their data is not
            # served by the public endpoint: asking for it returns HTTP 500 with a .NET
            # "Object reference not set to an instance of an object", every time, for any key or
            # format. Retrying is pure load on their server, so refuse before the first request
            # and say where the data actually lives.
            where = ref.extra.get("structure_url") or "another OECD space"
            raise NoData(f"{did}: registered here but defined in {where}; the public data endpoint cannot serve it")
        raw = raw_dir(self.name, did)
        csv_gz = raw / f"{did}.csv.gz"
        if csv_gz.exists():
            return [csv_gz]
        manifest = raw / "_parts.json"
        if manifest.exists() and read_json(manifest).get("complete"):
            return sorted(raw.glob("part_*.csv.gz"))
        oversized = raw / "_oversized.json"
        if oversized.exists() or manifest.exists():
            # We already know the whole-dataflow request does not fit, either because it was
            # abandoned at the cap once or because a split is half done. Asking again would spend
            # the entire cap (20 GB by default) to relearn it, so go straight to the slices.
            log.info("%s: known to exceed the cap; fetching it in slices", did)
            return self._fetch_split(ref, client)
        url = DATA_URL.format(agency=ref.extra["agency"], flow=ref.extra["flow"], version=ref.extra["version"])
        try:
            self._download_gz(url, csv_gz, client, self.max_download_bytes)
            return [csv_gz]
        except TooLarge as exc:
            log.warning("%s: %s; splitting by dimension instead", did, exc)
            write_json(oversized, {"reason": str(exc)[:300], "recorded_at": now_iso()})
            return self._fetch_split(ref, client)
        except httpx.HTTPStatusError as exc:
            # the dataflow is registered but empty: SDMX answers 404 to a query that matches no
            # observation. Terminal, so record it as such rather than retrying it every run.
            if exc.response.status_code == 404:
                raise NoData(f"{did}: the dataflow is in the catalogue but holds no observations") from exc
            raise

    def _download_gz(self, url: str, dest_gz: Path, client: PoliteClient, cap: int) -> Path:
        """Download a CSV and store it gzipped, without keeping the plain file around.

        HTTP 413 becomes ``TooLarge``: it is the server declining to generate a response this big,
        which means exactly what our own byte cap means, and every caller already knows how to
        answer that by asking for a smaller slice. Defensive only — OECD has not returned 413 for
        any request this crawler makes; it does return it for ``lastNObservations``, which we
        never send.
        """
        tmp_csv = dest_gz.with_name(dest_gz.name + ".plain")  # dataflow ids contain dots
        try:
            try:
                client.download(url, tmp_csv, max_bytes=cap)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 413:
                    raise TooLarge(f"{url}: the server refused to generate a response this large (HTTP 413)") from exc
                raise
            with open(tmp_csv, "rb") as src, gzip.open(dest_gz, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst)
        finally:
            tmp_csv.unlink(missing_ok=True)
        return dest_gz

    # -- splitting a dataflow that is too large for one request --------------------------------

    def _structure(self, ref: DatasetRef, client: PoliteClient) -> dict:
        """Dimension ids in key order, plus the codes of the first dimension. Cached."""
        did = ref.dataset_id
        path = cache_dir(self.name) / "structure" / f"{did}.json"
        if path.exists():
            return read_json(path)
        agency, flow, version = ref.extra["agency"], ref.extra["flow"], ref.extra["version"]
        xml_path = cache_dir(self.name) / "structure" / f"{did}.xml"
        client.download(f"{API}/dataflow/{agency}/{flow}/{version}?references=all", xml_path)
        root = ET.parse(xml_path).getroot()
        codelists = {cl.get("id"): [c.get("id") for c in cl.findall(f"{NS_STR}Code")] for cl in root.iter(f"{NS_STR}Codelist")}
        dsd = next(root.iter(f"{NS_STR}DataStructure"), None)
        if dsd is None:
            # ``references=all`` returned the dataflow but no structure, which is what an
            # external reference looks like; a bare StopIteration here said nothing useful
            xml_path.unlink(missing_ok=True)
            raise NoData(f"{did}: the public API has no data structure for it, only a reference to another space")
        dim_list = dsd.find(f".//{NS_STR}DimensionList")
        dims = []
        for d in dim_list.findall(f"{NS_STR}Dimension"):
            enum = d.find(f".//{NS_STR}Enumeration/Ref")
            dims.append({"id": d.get("id"), "codelist": enum.get("id") if enum is not None else None})
        out = {"dims": [d["id"] for d in dims], "split_dim": dims[0]["id"] if dims else None, "split_codes": codelists.get(dims[0]["codelist"], []) if dims else []}
        write_json(path, out)
        xml_path.unlink(missing_ok=True)
        return out

    def _fetch_split(self, ref: DatasetRef, client: PoliteClient) -> list:
        """Fetch the dataflow one slice of the first dimension at a time.

        Slices start as batches of codes and are halved whenever one still exceeds the (smaller)
        per-part cap, so a bad guess costs at most one part rather than one whole dataflow. Codes
        with no data answer 404 and are skipped. Progress is written to ``_parts.json`` after every
        part, so an interrupted split resumes.
        """
        did = ref.dataset_id
        raw = raw_dir(self.name, did)
        raw.mkdir(parents=True, exist_ok=True)
        manifest_path = raw / "_parts.json"
        state = read_json(manifest_path) if manifest_path.exists() else {"parts": [], "skipped": [], "done_codes": [], "complete": False}
        struct = self._structure(ref, client)
        if not struct["split_codes"]:
            raise RuntimeError(f"{did}: cannot split, the first dimension has no codelist")
        base = DATA_URL.format(agency=ref.extra["agency"], flow=ref.extra["flow"], version=ref.extra["version"]).split("/all?")[0]
        suffix = "." * (len(struct["dims"]) - 1)
        done = set(state["done_codes"])
        todo = [c for c in struct["split_codes"] if c not in done]
        log.info("%s: splitting on %s, %d codes (%d already fetched)", did, struct["split_dim"], len(todo), len(done))

        queue = [todo[i : i + SPLIT_BATCH] for i in range(0, len(todo), SPLIT_BATCH)]
        n_part = len(state["parts"])
        while queue:
            batch = queue.pop(0)
            key = "+".join(batch) + suffix
            part = raw / f"part_{n_part:04d}.csv.gz"
            try:
                self._download_gz(f"{base}/{key}?format=csvfilewithlabels", part, client, self.split_part_bytes)
            except TooLarge:
                # too big by our own cap, or by theirs (HTTP 413): halve the batch either way
                part.unlink(missing_ok=True)
                if len(batch) == 1:
                    log.warning("%s: %s=%s alone is too large to fetch; skipped", did, struct["split_dim"], batch[0])
                    state["skipped"].append(batch[0])
                    state["done_codes"].append(batch[0])
                    write_json(manifest_path, state)
                    continue
                mid = len(batch) // 2
                queue[0:0] = [batch[:mid], batch[mid:]]  # retry the halves first
                log.info("%s: part too large, splitting %d codes into %d + %d", did, len(batch), mid, len(batch) - mid)
                continue
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:  # no data for this slice
                    state["done_codes"].extend(batch)
                    write_json(manifest_path, state)
                    continue
                raise
            n_part += 1
            state["parts"].append({"file": part.name, "codes": batch})
            state["done_codes"].extend(batch)
            write_json(manifest_path, state)
        state["complete"] = True
        write_json(manifest_path, state)
        parts = sorted(raw.glob("part_*.csv.gz"))
        log.info("%s: split into %d parts (%d codes had no data, %d skipped as too large)", did, len(parts), len(struct["split_codes"]) - len(state["parts"]) - len(state["skipped"]), len(state["skipped"]))
        if not parts:
            raise RuntimeError(f"{did}: split produced no data at all")
        return parts

    def fetch_and_tidy(self, ref: DatasetRef, client: PoliteClient, keep_raw: bool = True) -> DatasetResult:
        did = ref.dataset_id
        agency, flow, version = ref.extra["agency"], ref.extra["flow"], ref.extra["version"]
        raw = raw_dir(self.name, did)
        out = dataset_dir(self.name, did)
        url = DATA_URL.format(agency=agency, flow=flow, version=version)
        parts = self.fetch_raw(ref, client)

        result = _tidy_csv(parts, out / "observations.parquet")
        if result.get("bad_rows"):
            log.warning("%s: %d malformed CSV rows skipped", did, result["bad_rows"])
        dims = [Dimension(id=d, name=result["dim_names"][d], codes=result["labels"].get(d, {})).__dict__ for d in result["dims"]]
        attrs = [Dimension(id="flag", name=result["dim_names"].get("OBS_STATUS", "Observation status"), codes=result["labels"].get("OBS_STATUS", {})).__dict__]
        lic = LICENSE.__dict__.copy()
        lic["attribution"] = LICENSE.attribution.format(
            year=now_iso()[:4], title=ref.title, dataset_id=f"{agency}:{flow}({version})", url=LANDING.format(flow=flow, agency=agency, version=version), date=now_iso()[:10]
        )
        meta = DatasetMeta(
            source=self.name,
            dataset_id=did,
            title=ref.title,
            description=ref.extra.get("description", ""),
            source_url=LANDING.format(flow=flow, agency=agency, version=version),
            api_url=url,
            metadata_urls={"structure": f"{API}/dataflow/{agency}/{flow}/{version}?references=all"},
            license=lic,
            origin={"agencies": ["OECD", agency], "us_federal": None},
            frequencies=result["frequencies"],
            dimensions=dims,
            attributes=attrs,
            time_start=result["period_min"],
            time_end=result["period_max"],
            n_series=result["n_series"],
            n_obs=result["n_obs"],
            last_updated=ref.last_updated,
            retrieved_at=now_iso(),
            raw_files=[str(p.relative_to(raw.parent.parent.parent)) for p in parts] if keep_raw else [],
            extra={"catalogue": {k: v for k, v in ref.extra.items() if k != "description"}, "n_raw_parts": len(parts)},
        )
        write_json(out / "dataset.json", meta.to_dict())
        if not keep_raw:
            for p in parts:
                p.unlink(missing_ok=True)
        return DatasetResult(did, result["n_series"], result["n_obs"], result["frequencies"])


LABEL_SUFFIX = "__label"  # lowercase, so it can never collide with an SDMX id


def _split_columns(cols: list[str]) -> tuple[list[str], list[str], dict[str, str]]:
    """Split the header into (dimension codes, attribute codes, code -> label text).

    After the four leading columns the layout is strictly ``CODE, Label`` pairs, so pair by
    position. Guessing from the shape of the string does not work: three dataflows label a
    dimension with its own id (``ISO`` -> "ISO", ``ISIN`` -> "ISIN"), which is indistinguishable
    from a code column, and mistaking a label for a code produced a duplicate column that polars
    refused. Pairing by position was checked against every OECD dataflow fetched so far: 1,102 of
    them, all strictly paired, no unpaired code anywhere.
    """
    body = cols[len(LEADING) :]
    if len(body) % 2:
        raise RuntimeError(f"unexpected OECD CSV layout: {len(body)} columns after {LEADING[-1]}, expected code/label pairs")
    code_cols, texts = body[0::2], body[1::2]
    label_text = dict(zip(code_cols, texts))
    it = code_cols.index("TIME_PERIOD") if "TIME_PERIOD" in code_cols else len(code_cols)
    iv = code_cols.index("OBS_VALUE") if "OBS_VALUE" in code_cols else it
    dims = code_cols[:it]
    attrs = [c for c in code_cols[iv + 1 :] if c in ATTR_KEEP]
    return dims, attrs, label_text


def _read_names(cols: list[str]) -> list[str]:
    """The names to read the body under, replacing the header's own.

    A label is free text, so it can repeat, be empty, or equal its own code; naming each label
    column ``<CODE>__label`` instead makes every column unique and predictable whatever the
    source writes.
    """
    body = cols[len(LEADING) :]
    names = list(LEADING)
    for code in body[0::2]:
        names += [code, code + LABEL_SUFFIX]
    return names


CSV_BLOCK_BYTES = 64 << 20


def _csv_header(csv_gz: Path) -> list[str]:
    """Column names only, without reading the body.

    The header must go through a real CSV parser rather than splitting on commas: OECD label
    columns are free text and several of them contain commas, notably the national-accounts
    ``STO`` dimension, labelled ``"Stocks, Transactions, Other Flows"``. Splitting naively tore
    that into three fields, so ``_split_columns`` invented a dimension called ``Transactions``
    and the tidy step then failed looking for a column no CSV parser would ever produce.
    """
    with gzip.open(csv_gz, "rt", encoding="utf-8", newline="") as fh:
        return [c.strip() for c in next(csv.reader(fh))]


def _tidy_csv(sources: Path | list, out_path: Path) -> dict:
    """Stream one or more labelled CSVs into the long layout, then sort by series.

    Nothing is ever fully in memory: each CSV is read in blocks, every block is written straight
    out, and the final ordering is a streaming sort that spills to disk. A dataflow too large for
    one request arrives as several parts (see ``_fetch_split``); they share a schema and are
    concatenated here.
    """
    parts = [sources] if isinstance(sources, Path) else list(sources)
    if not parts:
        raise RuntimeError("no CSV parts to tidy")
    cols = _csv_header(parts[0])
    if "TIME_PERIOD" not in cols:
        # a handful of OECD dataflows are questionnaires, not series ("Is there a CbCR law in
        # place?" -> "Yes"): no time dimension and a text OBS_VALUE. Nothing to store here.
        raise NotTimeSeries(f"{parts[0].stem} has no TIME_PERIOD column; it is not a time series")
    for p in parts[1:]:
        # the parts of a split fetch are read under one set of names, so they must agree
        if _csv_header(p) != cols:
            raise RuntimeError(f"{p.name} has a different header from {parts[0].name}; refusing to concatenate them")
    dims, attrs, label_text = _split_columns(cols)
    names = _read_names(cols)
    label_of = {c: c + LABEL_SUFFIX for c in dims + attrs}
    safe = {c: safe_dim_name(c) for c in dims}  # a dimension may collide with a reserved column
    dim_names = {safe.get(c, c): label_text.get(c) or c for c in dims + attrs}
    extra = {a: a.lower() for a in attrs if a != "OBS_STATUS"}
    keep = dims + ["TIME_PERIOD", "OBS_VALUE"] + attrs + [label_of[c] for c in dims + attrs]

    labels: dict[str, dict] = defaultdict(dict)
    freqs: set[str] = set()
    n_obs = 0
    pmin = pmax = None
    unsorted_path = out_path.with_name(out_path.name + ".unsorted")
    writer = ParquetBatchWriter(unsorted_path)

    read_opts = pa_csv.ReadOptions(block_size=CSV_BLOCK_BYTES, column_names=names, skip_rows=1)
    # strings_can_be_null: an empty CSV field means "no code / no flag", which must be null, not ""
    convert_opts = pa_csv.ConvertOptions(column_types={c: pa.string() for c in names}, strings_can_be_null=True)
    bad_rows = 0

    def _on_bad_row(row):
        """Drop a row the parser cannot line up, instead of losing the whole dataflow for it.

        OECD occasionally emits a row whose field count does not match the header. Skipping costs
        those observations; raising cost every observation in the dataflow.
        """
        nonlocal bad_rows
        bad_rows += 1
        if bad_rows <= 3:
            # row.number is None when the reader cannot place the row, which is the normal case
            # for a blocked read; saying "row None" made a handled skip look like a defect
            where = f"row {row.number}" if row.number is not None else "a row (position unknown in a blocked read)"
            log.warning("%s: skipping malformed CSV %s (%s columns, expected %s)",
                        out_path.parent.name, where, row.actual_columns, row.expected_columns)
        return "skip"

    # newlines_in_values: some free-text labels and comments carry a line break, and without this
    # the parser reads the fragment before it as a short row and rejects it
    parse_opts = pa_csv.ParseOptions(newlines_in_values=True, invalid_row_handler=_on_bad_row)

    def _blocks():
        """Every block of every part, as one sequence."""
        for part in parts:
            with gzip.open(part, "rb") as fh:
                yield from pa_csv.open_csv(fh, read_options=read_opts, parse_options=parse_opts, convert_options=convert_opts)

    try:
        for batch in _blocks():
            df = pl.from_arrow(pa.Table.from_batches([batch]))
            # the header is the only description of the layout, so say plainly when it disagrees
            # with the parsed frame rather than letting a select fail on a name further down
            missing = [c for c in dims + ["TIME_PERIOD", "OBS_VALUE"] if c not in df.columns]
            if missing:
                raise RuntimeError(f"header and body disagree; columns absent from the parsed CSV: {missing}")
            df = df.select([c for c in keep if c in df.columns])
            # code -> label maps, accumulated across blocks (small: one entry per distinct code)
            for c in dims + attrs:
                lc = label_of.get(c)
                if lc is None or lc not in df.columns or c not in df.columns:
                    continue
                pairs = df.select(pl.col(c), pl.col(lc)).unique().drop_nulls(c)
                labels[safe.get(c, c)].update(zip(pairs.get_column(c).to_list(), pairs.get_column(lc).fill_null("").to_list()))
            periods = df.get_column("TIME_PERIOD").unique().to_list()
            pframe = pl.DataFrame(
                {"period": periods, "date": [period_to_date(p) for p in periods], "_pfreq": [frequency_of_period(p) for p in periods]},
                schema={"period": pl.String, "date": pl.Date, "_pfreq": pl.String},
            )
            # rename the dimensions to their safe names first, so that creating "value" or
            # "flag" below cannot collide with a dimension that happens to carry that name
            sdims = [safe.get(d, d) for d in dims]
            long = (
                df.select(dims + ["TIME_PERIOD", "OBS_VALUE"] + [a for a in attrs if a in df.columns])
                .rename({c: s for c, s in safe.items() if c != s})
                .rename({"TIME_PERIOD": "period", "OBS_VALUE": "_v"})
                .with_columns(pl.col("_v").cast(pl.Float64, strict=False).alias("value"))
                .join(pframe, on="period", how="left")
                .with_columns(pl.concat_str([pl.col(d).fill_null("") for d in sdims], separator=".").alias("series_key"))
            )
            long = long.with_columns((pl.col("OBS_STATUS") if "OBS_STATUS" in long.columns else pl.lit(None, pl.String)).alias("flag"))
            long = long.filter(pl.col("value").is_not_null() | pl.col("flag").is_not_null())
            if long.is_empty():
                continue
            if "FREQ" in sdims:
                freqs.update(canonical_frequency(f, "oecd") for f in long.get_column("FREQ").unique().to_list() if f)
            else:
                freqs.update(f for f in long.get_column("_pfreq").unique().to_list() if f and f != "OTHER")
            long = long.rename({a: n for a, n in extra.items() if a in long.columns})
            out = long.select(["series_key", *sdims, "period", "date", "value", "flag", *[n for a, n in extra.items() if n in long.columns]])
            writer.write(out)
            n_obs += out.height
            bmin, bmax = out.get_column("date").min(), out.get_column("date").max()
            if bmin is not None and (pmin is None or bmin < pmin):
                pmin = bmin
            if bmax is not None and (pmax is None or bmax > pmax):
                pmax = bmax
        written = writer.close()
    except Exception:
        writer.abort()
        unsorted_path.unlink(missing_ok=True)
        raise

    n_series = 0
    if written is not None:
        # streaming sort so the file ends up grouped by series, which layer 2 relies on
        lf = pl.scan_parquet(unsorted_path).sort(["series_key", "date"])
        lf.sink_parquet(out_path, compression="zstd", row_group_size=200_000)
        n_series = int(pl.scan_parquet(out_path).select(pl.col("series_key").n_unique()).collect().item())
        unsorted_path.unlink(missing_ok=True)
        # keep only the codes that actually occur; one streaming pass per dimension
        for c in [safe.get(d, d) for d in dims]:
            used = set(pl.scan_parquet(out_path).select(pl.col(c).unique()).collect().get_column(c).to_list())
            labels[c] = {k: v for k, v in sorted(labels.get(c, {}).items()) if k in used}

    return {
        "dims": [safe.get(d, d) for d in dims],
        "dim_names": dim_names,
        "labels": {k: dict(v) for k, v in labels.items()},
        "frequencies": sorted(freqs),
        "n_series": n_series,
        "n_obs": n_obs,
        "period_min": str(pmin) if pmin else None,
        "period_max": str(pmax) if pmax else None,
        "bad_rows": bad_rows,
    }
