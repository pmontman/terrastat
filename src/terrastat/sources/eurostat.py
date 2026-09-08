"""Eurostat, through the SDMX 2.1 dissemination API (no key needed).

Catalogue  : table of contents (XML) + metabase (dimension codes per dataset, which also tells the
             frequency before anything is downloaded).
Data       : one bulk TSV per dataset, ``.../sdmx/2.1/data/<CODE>?format=TSV&compressed=true``.
             The server generates the file on request, which takes one or two minutes for a large
             dataset; that is why the catalogue is walked sequentially and slowly.
Structure  : the data structure definition (small) names the dimensions and points at versioned
             codelists (code -> label), which are shared across datasets and cached once.
Metadata   : each dataset links to a reference-metadata (ESMS) page; the URL is stored, the page
             is not downloaded in this version.

Licence: Eurostat's copyright notice authorises re-use of statistical data for commercial and
non-commercial purposes with source acknowledgement; a few exceptions exist (see LICENSE below).
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import polars as pl

from terrastat.http import PoliteClient
from terrastat.sources.base import (
    CATALOG_SCHEMA,
    DatasetMeta,
    DatasetRef,
    DatasetResult,
    Dimension,
    License,
    Source,
    safe_dim_name,
)
from terrastat.storage import (
    ParquetBatchWriter,
    cache_dir,
    dataset_dir,
    ensure_gzip,
    is_fresh,
    now_iso,
    raw_dir,
    read_json,
    write_json,
)
from terrastat.timeparse import canonical_frequency, frequency_of_period, period_to_date

log = logging.getLogger(__name__)

API = "https://ec.europa.eu/eurostat/api/dissemination"
TOC_XML = f"{API}/catalogue/toc/xml"
METABASE = f"{API}/catalogue/metabase.txt.gz"
SDMX = f"{API}/sdmx/2.1"
LANDING = "https://ec.europa.eu/eurostat/databrowser/view/{code}/default/table"

NS_STR = "{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure}"
NS_COM = "{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common}"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

LICENSE = License(
    id="eurostat-reuse",
    name="Eurostat copyright notice: free re-use with acknowledgement (Commission Decision 2011/833/EU; CC BY 4.0 for editorial content)",
    url="https://ec.europa.eu/eurostat/web/main/help/copyright-notice",
    attribution="Source: Eurostat, dataset {dataset_id} ({title}), {url}, accessed {date}",
    notes=(
        "Re-use for commercial and non-commercial purposes is authorised provided the source is "
        "acknowledged and modifications are stated. The notice lists exceptions: certain data for "
        "non-EU countries and certain trade data are restricted to non-commercial use, and third-party "
        "material needs separate permission. Check the dataset's ESMS metadata page when in doubt."
    ),
)

TIME_DIMS = {"time", "time_period"}


class EurostatSource(Source):
    name = "eurostat"
    default_min_interval = 3.0  # Eurostat asks users not to parallelise; one slow stream is fine
    default_max_per_minute = 12

    # -- catalogue ----------------------------------------------------------------------------

    def catalog(self, client: PoliteClient) -> pl.DataFrame:
        cache = cache_dir(self.name)
        toc_path, meta_path = cache / "toc.xml", cache / "metabase.txt.gz"
        if not is_fresh(toc_path):
            client.download(TOC_XML, toc_path)
        if not is_fresh(meta_path):
            client.download(METABASE, meta_path)
        leaves = _parse_toc(toc_path)
        dims, freqs = _parse_metabase(meta_path)
        # a dataset can sit in several folders of the tree: keep one row, remember every path
        merged: dict[str, dict] = {}
        for leaf in leaves:
            code = leaf["code"]
            if code in merged:
                merged[code].setdefault("paths", []).append(" > ".join(leaf.get("path", [])))
            else:
                leaf["paths"] = [" > ".join(leaf.get("path", []))]
                merged[code] = leaf
        rows = []
        for leaf in merged.values():
            code = leaf["code"]
            rows.append(
                {
                    "dataset_id": code,
                    "title": leaf.get("title", ""),
                    "frequencies": sorted(freqs.get(code, set())),
                    "n_values": leaf.get("values"),
                    "last_updated": leaf.get("lastUpdate"),
                    "extra": json.dumps(
                        {
                            "type": leaf.get("type"),
                            "data_start": leaf.get("dataStart"),
                            "data_end": leaf.get("dataEnd"),
                            "last_structure_change": leaf.get("lastModified"),
                            "source": leaf.get("source", ""),
                            "unit": leaf.get("unit", ""),
                            "short_description": leaf.get("shortDescription", ""),
                            "metadata": leaf.get("metadata", {}),
                            "download": leaf.get("downloadLink", {}),
                            "dimensions": dims.get(code, []),
                            "paths": leaf.get("paths", []),
                        },
                        ensure_ascii=False,
                    ),
                }
            )
        df = pl.DataFrame(rows, schema=CATALOG_SCHEMA)
        log.info("eurostat catalogue: %d datasets/tables", df.height)
        return df

    # -- one dataset --------------------------------------------------------------------------

    def fetch_and_tidy(self, ref: DatasetRef, client: PoliteClient, keep_raw: bool = True) -> DatasetResult:
        code = ref.dataset_id
        raw = raw_dir(self.name, code)
        out = dataset_dir(self.name, code)
        data_url = f"{SDMX}/data/{code.upper()}?format=TSV&compressed=true"
        tsv_gz = self.fetch_raw(ref, client)[0]

        result = _tidy_tsv(tsv_gz, out / "observations.parquet")
        if result is None:
            result = {"n_series": 0, "n_obs": 0, "frequencies": [], "used": {}, "flags": set(), "period_min": None, "period_max": None}

        # structure and labels: best effort, the observations are already safe on disk
        try:
            dsd = self._dsd(code, client)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: no data structure definition (%s); dimensions kept without labels", code, exc)
            dsd = {"dimensions": [{"id": d, "codelist": None} for d in result["used"]], "attributes": []}
        codelists: dict[str, dict] = {}
        for d in dsd["dimensions"] + dsd["attributes"]:
            key = d.get("codelist")
            if key and key not in codelists:
                try:
                    codelists[key] = self._codelist(key, client)
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s: codelist %s unavailable (%s)", code, key, exc)
                    codelists[key] = {"codes": {}}
        concepts = dsd.get("concepts", {})

        dimensions = []
        # the structure calls the dimension by its original name; the column may have been renamed
        # to avoid colliding with a reserved column, so match on the safe name
        known = {safe_dim_name(d["id"]) for d in dsd["dimensions"]}
        for d in dsd["dimensions"] + [{"id": u, "codelist": None} for u in result["used"] if u not in known]:
            col = safe_dim_name(d["id"])
            cl = codelists.get(d.get("codelist")) or {"codes": {}}
            used = sorted(c for c in result["used"].get(col, set()) if c is not None)
            name = concepts.get(d["id"]) or concepts.get(d["id"].upper()) or d["id"]
            dimensions.append(Dimension(id=col, name=name, codes={c: cl["codes"].get(c, "") for c in used}).__dict__)
        # Eurostat packs two different things into the flag cell: the observation status ("p" for
        # provisional) and, prefixed with "@", the confidentiality status ("@C" for confidential).
        # Label both, from their own codelists, so no flag is left without a meaning.
        flag_codes: dict[str, str] = {}
        for a in dsd["attributes"]:
            if not a.get("codelist"):
                continue
            aid = a["id"].upper()
            if aid not in ("OBS_FLAG", "CONF_STATUS"):
                continue
            cl = codelists[a["codelist"]]
            prefix = "@" if aid == "CONF_STATUS" else ""
            for flag_code, label in cl["codes"].items():  # never shadow `code`, the dataset id
                flag_codes[f"{prefix}{flag_code}"] = label
        attributes = []
        if flag_codes or result["flags"]:
            attributes.append(
                Dimension(
                    id="flag",
                    name="Observation status, and confidentiality status where prefixed with @",
                    codes={c: flag_codes.get(c, "") for c in sorted(result["flags"])},
                ).__dict__
            )

        lic = LICENSE.__dict__.copy()
        lic["attribution"] = LICENSE.attribution.format(dataset_id=code, title=ref.title, url=LANDING.format(code=code), date=now_iso()[:10])
        meta = DatasetMeta(
            source=self.name,
            dataset_id=code,
            title=ref.title,
            description=ref.extra.get("short_description", ""),
            source_url=LANDING.format(code=code),
            api_url=data_url,
            metadata_urls=ref.extra.get("metadata", {}),
            license=lic,
            origin={"agencies": [ref.extra.get("source") or "Eurostat"], "us_federal": None},
            frequencies=result["frequencies"],
            dimensions=dimensions,
            attributes=attributes,
            time_start=result["period_min"],
            time_end=result["period_max"],
            n_series=result["n_series"],
            n_obs=result["n_obs"],
            last_updated=ref.last_updated,
            retrieved_at=now_iso(),
            raw_files=[str(tsv_gz.relative_to(raw.parent.parent.parent))] if keep_raw else [],
            extra={"catalogue": {k: v for k, v in ref.extra.items() if k not in ("metadata", "download", "dimensions")}},
        )
        write_json(out / "dataset.json", meta.to_dict())
        if not keep_raw and tsv_gz.exists():
            tsv_gz.unlink()
        return DatasetResult(code, result["n_series"], result["n_obs"], result["frequencies"])

    # -- structure and codelists (cached) -----------------------------------------------------

    def fetch_raw(self, ref: DatasetRef, client: PoliteClient) -> list:
        code = ref.dataset_id
        tsv_gz = raw_dir(self.name, code) / f"{code}.tsv.gz"
        if not tsv_gz.exists():
            client.download(f"{SDMX}/data/{code.upper()}?format=TSV&compressed=true", tsv_gz)
            ensure_gzip(tsv_gz)
        return [tsv_gz]

    def _dsd(self, code: str, client: PoliteClient) -> dict:
        """Dimensions, their codelist references, and the human names of the concepts (two small XMLs)."""
        path = cache_dir(self.name) / "dsd" / f"{code}.json"
        if path.exists():
            return read_json(path)
        xml_path = cache_dir(self.name) / "dsd" / f"{code}.xml"
        client.download(f"{SDMX}/datastructure/ESTAT/{code.upper()}", xml_path)
        dsd = _parse_dsd(xml_path)
        xml_path.unlink(missing_ok=True)
        try:
            cs_path = cache_dir(self.name) / "dsd" / f"{code}.concepts.xml"
            client.download(f"{SDMX}/conceptscheme/ESTAT/{code.upper()}", cs_path)
            dsd["concepts"] = _parse_conceptscheme(cs_path)
            cs_path.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: concept scheme unavailable (%s); dimension ids used as names", code, exc)
            dsd["concepts"] = {}
        write_json(path, dsd)
        return dsd

    def _codelist(self, key: str, client: PoliteClient) -> dict:
        """key = 'AGENCY/ID/VERSION' -> {'id', 'version', 'codes': {code: label}}.

        The TSV rendering of a codelist arrives in a few seconds; the SDMX-ML one, with all
        languages, takes the server two minutes for a large list such as GEO.
        """
        agency, cid, version = key.split("/")
        path = cache_dir(self.name) / "codelist" / f"{cid}_{version}.json"
        if path.exists():
            return read_json(path)
        tsv_path = cache_dir(self.name) / "codelist" / f"{cid}_{version}.tsv.gz"
        client.download(f"{SDMX}/codelist/{agency}/{cid}/{version}?format=TSV&compressed=true&lang=en", tsv_path)
        ensure_gzip(tsv_path)
        codes: dict[str, str] = {}
        with gzip.open(tsv_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 2 and parts[0]:
                    codes[parts[0]] = parts[1]
        cl = {"id": cid, "version": version, "codes": codes}
        write_json(path, cl)
        tsv_path.unlink(missing_ok=True)
        return cl


# -- parsers ------------------------------------------------------------------------------------


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _parse_toc(path: Path) -> list[dict]:
    """Flatten the TOC tree into one dict per leaf (dataset or table), keeping the folder path."""
    leaves: list[dict] = []
    stack: list[dict] = []  # enclosing branches, each {"title": str | None}
    for event, el in ET.iterparse(path, events=("start", "end")):
        tag = _local(el.tag)
        if event == "start":
            if tag == "branch":
                stack.append({"title": None})
            continue
        if tag == "title" and stack and stack[-1]["title"] is None and el.get("language", "en") == "en":
            # the first English title after a branch opens is the branch's own
            stack[-1]["title"] = (el.text or "").strip()
        elif tag == "leaf":
            leaf: dict = {"type": el.get("type"), "path": [b["title"] for b in stack if b["title"]]}
            for c in el:
                ct = _local(c.tag)
                text = (c.text or "").strip()
                if ct in ("title", "shortDescription", "source", "unit"):
                    if c.get("language", "en") == "en":
                        leaf[ct] = text
                elif ct in ("metadata", "downloadLink"):
                    leaf.setdefault(ct, {})[c.get("format", "")] = text
                elif ct == "values":
                    leaf["values"] = int(text) if text.isdigit() else None
                elif ct != "children":
                    leaf[ct] = text
            if leaf.get("code"):
                leaves.append(leaf)
            el.clear()
        elif tag == "branch":
            stack.pop()
            el.clear()
    return leaves


def _parse_metabase(path: Path) -> tuple[dict, dict]:
    """metabase lines: '<dataset>\\t<dimension>\\t<code>'. Returns (dims per dataset, frequencies per dataset)."""
    dims: dict[str, list[str]] = defaultdict(list)
    freqs: dict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str]] = set()
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            ds, dim, code = parts[0], parts[1], parts[2]
            if (ds, dim) not in seen:
                seen.add((ds, dim))
                dims[ds].append(dim)
            if dim == "time":
                f = frequency_of_period(code)
                if f != "OTHER":
                    freqs[ds].add(f)
            elif dim == "freq":
                freqs[ds].add(canonical_frequency(code, "eurostat"))
    return dims, freqs


def _parse_dsd(path: Path) -> dict:
    root = ET.parse(path).getroot()
    dims, attrs = [], []
    for dim in root.iter(f"{NS_STR}Dimension"):
        ref = dim.find(f".//{NS_STR}Enumeration/Ref")
        dims.append(
            {
                "id": dim.get("id"),
                "position": int(dim.get("position", 0)),
                "codelist": f"{ref.get('agencyID')}/{ref.get('id')}/{ref.get('version')}" if ref is not None else None,
            }
        )
    for a in root.iter(f"{NS_STR}Attribute"):
        ref = a.find(f".//{NS_STR}Enumeration/Ref")
        attrs.append(
            {
                "id": a.get("id"),
                "codelist": f"{ref.get('agencyID')}/{ref.get('id')}/{ref.get('version')}" if ref is not None else None,
            }
        )
    dims.sort(key=lambda d: d["position"])
    time_dim = next((t.get("id") for t in root.iter(f"{NS_STR}TimeDimension")), "TIME_PERIOD")
    return {"dimensions": dims, "attributes": attrs, "time_dimension": time_dim}


def _en_name(el) -> str:
    names = el.findall(f"{NS_COM}Name")
    for n in names:
        if n.get(XML_LANG, "en") == "en":
            return (n.text or "").strip()
    return (names[0].text or "").strip() if names else ""


def _parse_conceptscheme(path: Path) -> dict:
    """{concept id: English name} from a dataset's concept scheme."""
    root = ET.parse(path).getroot()
    return {c.get("id"): _en_name(c) for c in root.iter(f"{NS_STR}Concept")}


# -- TSV -> long parquet --------------------------------------------------------------------------

TARGET_CELLS_PER_CHUNK = 4_000_000


def _period_frame(periods: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "period": periods,
            "date": [period_to_date(p) for p in periods],
            "_pfreq": [frequency_of_period(p) for p in periods],
        },
        schema={"period": pl.String, "date": pl.Date, "_pfreq": pl.String},
    )


def _tidy_tsv(tsv_gz: Path, out_path: Path) -> dict | None:
    """Stream the wide TSV, melt it into the long layout, and write one row group per chunk.

    Each TSV row is one series, so every chunk holds whole series and the output file is
    grouped by ``series_key`` (a property layer 2 relies on).
    """
    with gzip.open(tsv_gz, "rt", encoding="utf-8", newline="") as fh:
        header = fh.readline().rstrip("\r\n")
        if not header:
            return None
        first_col, *col_headers = header.split("\t")
        row_part, col_dim = first_col.split("\\", 1) if "\\" in first_col else (first_col, "TIME_PERIOD")
        # a few datasets (sbs_ins_5d1 and friends) have a dimension literally named "value",
        # which would collide with the observation column
        row_dims = [safe_dim_name(d) for d in row_part.split(",")]
        col_dim = safe_dim_name(col_dim.strip())
        col_headers = [h.strip() for h in col_headers]
        col_is_time = col_dim.upper() in ("TIME_PERIOD", "TIME")
        time_in_rows = None if col_is_time else next((d for d in row_dims if d.lower() in TIME_DIMS), None)

        ncols = len(col_headers)
        rows_per_chunk = max(200, TARGET_CELLS_PER_CHUNK // max(1, ncols))
        writer = ParquetBatchWriter(out_path)
        used: dict[str, set] = defaultdict(set)
        flags: set[str] = set()
        freqs: set[str] = set()
        n_series = 0
        n_obs = 0
        pmin = pmax = None
        col_names = [f"c{i}" for i in range(ncols)]
        colmap = pl.DataFrame({"_col": col_names, "_hdr": col_headers})
        col_periods = _period_frame(col_headers) if col_is_time else None
        header_line = "_key\t" + "\t".join(col_names) + "\n"
        try:
            while True:
                lines = []
                for _ in range(rows_per_chunk):
                    line = fh.readline()
                    if not line:
                        break
                    lines.append(line)
                if not lines:
                    break
                buf = io.BytesIO((header_line + "".join(lines)).encode("utf-8"))
                wide = pl.read_csv(buf, separator="\t", has_header=True, infer_schema=False, quote_char=None, truncate_ragged_lines=True)
                long = (
                    wide.unpivot(index="_key", on=col_names, variable_name="_col", value_name="_raw")
                    .with_columns(pl.col("_raw").str.strip_chars().str.split_exact(" ", 1).alias("_vf"))
                    .unnest("_vf")
                    .rename({"field_0": "_v", "field_1": "flag"})
                    .with_columns(
                        pl.when(pl.col("_v") == ":").then(None).otherwise(pl.col("_v")).cast(pl.Float64, strict=False).alias("value"),
                        pl.when(pl.col("flag") == "").then(None).otherwise(pl.col("flag")).alias("flag"),
                    )
                    .filter(pl.col("value").is_not_null() | pl.col("flag").is_not_null())
                    .drop("_raw", "_v")
                )
                if long.is_empty():
                    continue
                long = long.with_columns(pl.col("_key").str.split(",").list.to_struct(fields=row_dims).alias("_d")).unnest("_d")
                long = long.join(colmap, on="_col", how="left").drop("_col")
                if col_is_time:
                    long = long.rename({"_hdr": "period"}).join(col_periods, on="period", how="left")
                    dim_cols = list(row_dims)
                else:
                    long = long.rename({"_hdr": col_dim})
                    dim_cols = row_dims + [col_dim]
                    if time_in_rows:
                        periods = long.get_column(time_in_rows).unique().to_list()
                        long = long.rename({time_in_rows: "period"}).join(_period_frame(periods), on="period", how="left")
                        dim_cols = [d for d in dim_cols if d != time_in_rows]
                    else:
                        long = long.with_columns(pl.lit(None, pl.String).alias("period"), pl.lit(None, pl.Date).alias("date"), pl.lit(None, pl.String).alias("_pfreq"))
                dim_cols = [d for d in dim_cols if d in long.columns]
                long = long.with_columns(pl.concat_str([pl.col(d).fill_null("") for d in dim_cols], separator=".").alias("series_key"))
                out = long.select(["series_key", *dim_cols, "period", "date", "value", "flag"]).sort(["series_key", "date"])
                writer.write(out)
                n_series += out.get_column("series_key").n_unique()
                n_obs += out.height
                for d in dim_cols:
                    used[d].update(out.get_column(d).unique().to_list())
                flags.update(x for x in out.get_column("flag").unique().to_list() if x)
                if "freq" in out.columns:
                    freqs.update(canonical_frequency(f, "eurostat") for f in out.get_column("freq").unique().to_list() if f)
                else:
                    freqs.update(f for f in long.get_column("_pfreq").unique().to_list() if f)
                pmn, pmx = out.get_column("date").min(), out.get_column("date").max()
                if pmn is not None and (pmin is None or pmn < pmin):
                    pmin = pmn
                if pmx is not None and (pmax is None or pmx > pmax):
                    pmax = pmx
        except Exception:
            writer.abort()
            raise
        written = writer.close()
    if written is None:
        return None
    return {
        "n_series": n_series,
        "n_obs": n_obs,
        "frequencies": sorted(f for f in freqs if f and f != "OTHER") or sorted(f for f in freqs if f),
        "used": used,
        "flags": flags,
        "period_min": str(pmin) if pmin else None,
        "period_max": str(pmax) if pmax else None,
    }
