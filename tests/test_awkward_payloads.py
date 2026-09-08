"""Real shapes from the sources that used to crash the tidy step."""
import gzip
import json

import polars as pl
import pytest

from terrastat.sources.base import NotTimeSeries, safe_dim_name
from terrastat.sources.eurostat import _tidy_tsv
from terrastat.sources.oecd import _csv_header, _split_columns, _tidy_csv


def test_reserved_dimension_names_are_suffixed():
    assert safe_dim_name("value") == "value_dim"
    assert safe_dim_name("date") == "date_dim"
    assert safe_dim_name("flag") == "flag_dim"
    assert safe_dim_name("geo") == "geo"


def test_eurostat_dimension_literally_named_value(tmp_path):
    """sbs_ins_5d1 and three siblings have a dimension called 'value'; it must not collide
    with the observation column."""
    tsv = "freq,indic_sb,value,geo\\TIME_PERIOD\t2015 \t2016 \n" "A,V11110,MIO_EUR,AT\t1.5 \t2.5 p\n" "A,V11110,PC,BE\t3.0 \t: \n"
    src = tmp_path / "x.tsv.gz"
    with gzip.open(src, "wt", encoding="utf-8") as fh:
        fh.write(tsv)
    out = tmp_path / "obs.parquet"
    res = _tidy_tsv(src, out)
    df = pl.read_parquet(out)
    assert "value_dim" in df.columns and "value" in df.columns
    assert df.columns == ["series_key", "freq", "indic_sb", "value_dim", "geo", "period", "date", "value", "flag"]
    at = df.filter(pl.col("geo") == "AT").sort("date")
    assert at.get_column("value").to_list() == [1.5, 2.5]
    assert at.get_column("value_dim").to_list() == ["MIO_EUR", "MIO_EUR"]
    assert res["n_series"] == 2
    assert "value_dim" in res["used"]


def test_oecd_questionnaire_is_not_a_time_series(tmp_path):
    """A few OECD dataflows are surveys: no TIME_PERIOD, and OBS_VALUE is text such as 'Yes'."""
    csv = (
        "STRUCTURE,STRUCTURE_ID,STRUCTURE_NAME,ACTION,REF_AREA,Reference area,MEASURE,Measure,OBS_VALUE,Observation value\n"
        "DATAFLOW,X:Y(1.0),Questionnaire,I,AUS,Australia,CBCR_LAW,Is there a law?,Yes,\n"
    )
    src = tmp_path / "q.csv.gz"
    with gzip.open(src, "wt", encoding="utf-8") as fh:
        fh.write(csv)
    with pytest.raises(NotTimeSeries):
        _tidy_csv(src, tmp_path / "obs.parquet")


def test_oecd_dimension_named_value(tmp_path):
    csv = (
        "STRUCTURE,STRUCTURE_ID,STRUCTURE_NAME,ACTION,REF_AREA,Reference area,VALUE,Value type,TIME_PERIOD,Time period,OBS_VALUE,Observation value\n"
        "DATAFLOW,X:Y(1.0),D,I,AUS,Australia,GROSS,Gross,2020,,1.5,\n"
        "DATAFLOW,X:Y(1.0),D,I,AUS,Australia,GROSS,Gross,2021,,2.5,\n"
    )
    src = tmp_path / "d.csv.gz"
    with gzip.open(src, "wt", encoding="utf-8") as fh:
        fh.write(csv)
    out = tmp_path / "obs.parquet"
    res = _tidy_csv(src, out)
    df = pl.read_parquet(out)
    assert "VALUE_dim" in df.columns and "value" in df.columns
    assert df.get_column("value").to_list() == [1.5, 2.5]
    assert res["n_series"] == 1
    assert res["labels"]["VALUE_dim"] == {"GROSS": "Gross"}


def test_dataset_metadata_keeps_its_own_id(tmp_path, monkeypatch):
    """The dataset id must survive tidying: a loop variable once shadowed it and the citation,
    the landing URL and the dataset_id column all silently became a flag code."""
    import gzip

    from terrastat.http import PoliteClient
    from terrastat.sources.base import DatasetRef
    from terrastat.sources.eurostat import EurostatSource

    monkeypatch.setenv("TERRASTAT_DATA_DIR", str(tmp_path))
    did = "apro_test_m"
    raw = tmp_path / "raw" / "eurostat" / did
    raw.mkdir(parents=True)
    with gzip.open(raw / f"{did}.tsv.gz", "wt", encoding="utf-8") as fh:
        fh.write("freq,geo\\TIME_PERIOD\t2020-01 \t2020-02 \n" "M,FR\t1.5 \t2.5 p\n" "M,BE\t3.0 \t: @C\n")

    class NoNetwork(PoliteClient):
        def __init__(self):
            pass

        def download(self, *a, **k):
            raise RuntimeError("the structure request should be optional")

    ref = DatasetRef(source="eurostat", dataset_id=did, title="Poultry test", extra={})
    res = EurostatSource().fetch_and_tidy(ref, NoNetwork())
    assert res.dataset_id == did

    meta = json.loads((tmp_path / "datasets" / "eurostat" / did / "dataset.json").read_text(encoding="utf-8"))
    assert meta["dataset_id"] == did
    assert f"dataset {did} " in meta["license"]["attribution"]
    assert did in meta["source_url"]


# -- OECD CSV quoting -------------------------------------------------------------------------
#
# The real header of OECD.SDD.NAD DSD_NAMAIN10_IDC@DF_TABLE5_IDC and its ~20 siblings: the STO
# and EXPENDITURE labels both contain commas, so the source quotes them.

_NAMAIN_HEADER = (
    "STRUCTURE,STRUCTURE_ID,STRUCTURE_NAME,ACTION,REF_AREA,Reference area,"
    'STO,"Stocks, Transactions, Other Flows",'
    'EXPENDITURE,"Expenditure (COFOG, COICOP, COPP or COPNI)",'
    "TIME_PERIOD,Time period,OBS_VALUE,Observation value,OBS_STATUS,Observation status\n"
)
_ROW = "DATAFLOW,X:Y(1.0),T,I,AUS,Australia,P51G,Gross fixed capital formation,_T,Total,{p},,{v},,,\n"


def _gz(tmp_path, name, text):
    src = tmp_path / name
    with gzip.open(src, "wt", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return src


def test_quoted_commas_in_a_label_do_not_invent_a_dimension(tmp_path):
    """A label such as "Stocks, Transactions, Other Flows" used to be split on its own commas,
    which conjured a dimension named Transactions and broke ~20 national-accounts dataflows."""
    cols = _csv_header(_gz(tmp_path, "n.csv.gz", _NAMAIN_HEADER))
    assert "Stocks, Transactions, Other Flows" in cols
    assert "Transactions" not in cols
    dims, attrs, label_of = _split_columns(cols)
    assert dims == ["REF_AREA", "STO", "EXPENDITURE"]
    assert label_of["STO"] == "Stocks, Transactions, Other Flows"
    assert attrs == ["OBS_STATUS"]


def test_namain_style_dataflow_tidies(tmp_path):
    body = _ROW.format(p=2020, v=1.5) + _ROW.format(p=2021, v=2.5)
    out = tmp_path / "obs.parquet"
    res = _tidy_csv(_gz(tmp_path, "n.csv.gz", _NAMAIN_HEADER + body), out)
    assert res["n_series"] == 1 and res["n_obs"] == 2
    assert res["dims"] == ["REF_AREA", "STO", "EXPENDITURE"]
    assert res["dim_names"]["STO"] == "Stocks, Transactions, Other Flows"
    assert res["labels"]["STO"] == {"P51G": "Gross fixed capital formation"}
    assert pl.read_parquet(out).get_column("value").to_list() == [1.5, 2.5]


def test_a_malformed_row_is_skipped_not_fatal(tmp_path):
    """OECD sometimes emits a row with fewer fields than the header; losing that row is better
    than losing every observation in the dataflow."""
    body = _ROW.format(p=2020, v=1.5) + "DATAFLOW,X:Y(1.0),T,I,AUS\n" + _ROW.format(p=2021, v=2.5)
    res = _tidy_csv(_gz(tmp_path, "n.csv.gz", _NAMAIN_HEADER + body), tmp_path / "obs.parquet")
    assert res["bad_rows"] == 1
    assert res["n_obs"] == 2


def test_a_line_break_inside_a_quoted_field_is_not_a_row_break(tmp_path):
    body = 'DATAFLOW,X:Y(1.0),"Table 5,\nannual",I,AUS,Australia,P51G,Gross fixed capital formation,_T,Total,2020,,1.5,,,\n'
    res = _tidy_csv(_gz(tmp_path, "n.csv.gz", _NAMAIN_HEADER + body), tmp_path / "obs.parquet")
    assert res["bad_rows"] == 0 and res["n_obs"] == 1
