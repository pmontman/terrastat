import datetime as dt
import gzip

import polars as pl

from terrastat.sources.eurostat import _tidy_tsv

TSV = (
    "freq,unit,geo\\TIME_PERIOD\t2015-01 \t2015-02 \t2015-03 \n"
    "M,PC,AT\t1.5 \t: \t2.5 p\n"
    "M,PC,BE\t: \t: \t: \n"  # no observations at all: must not appear
    "M,PC,DE\t3.0 b\t: c\t4.0 \n"  # ': c' keeps the confidential flag without a value
)


def test_tidy_tsv_monthly(tmp_path):
    src = tmp_path / "x.tsv.gz"
    with gzip.open(src, "wt", encoding="utf-8") as fh:
        fh.write(TSV)
    out = tmp_path / "observations.parquet"
    res = _tidy_tsv(src, out)
    df = pl.read_parquet(out)
    assert df.columns == ["series_key", "freq", "unit", "geo", "period", "date", "value", "flag"]
    assert res["n_series"] == 2 and res["n_obs"] == 5
    assert res["frequencies"] == ["M"]
    assert res["period_min"] == "2015-01-01" and res["period_max"] == "2015-03-01"
    at = df.filter(pl.col("geo") == "AT").sort("date")
    assert at.get_column("value").to_list() == [1.5, 2.5]
    assert at.get_column("flag").to_list() == [None, "p"]
    assert at.get_column("date").to_list() == [dt.date(2015, 1, 1), dt.date(2015, 3, 1)]
    de = df.filter(pl.col("geo") == "DE").sort("date")
    assert de.get_column("value").to_list() == [3.0, None, 4.0]
    assert de.get_column("flag").to_list() == ["b", "c", None]
    assert set(df.get_column("series_key").to_list()) == {"M.PC.AT", "M.PC.DE"}
    assert res["used"]["geo"] == {"AT", "DE"} and res["flags"] == {"p", "b", "c"}


def test_tidy_tsv_column_dimension_is_not_time(tmp_path):
    tsv = "freq,unit,time\\geo\tAT \tDE \n" "A,PC,2019\t1 \t2 \n" "A,PC,2020\t3 \t: \n"
    src = tmp_path / "y.tsv.gz"
    with gzip.open(src, "wt", encoding="utf-8") as fh:
        fh.write(tsv)
    out = tmp_path / "observations.parquet"
    res = _tidy_tsv(src, out)
    df = pl.read_parquet(out).sort(["series_key", "date"])
    assert set(df.get_column("series_key").to_list()) == {"A.PC.AT", "A.PC.DE"}
    assert df.filter(pl.col("geo") == "AT").get_column("value").to_list() == [1.0, 3.0]
    assert df.get_column("date").to_list()[:2] == [dt.date(2019, 1, 1), dt.date(2020, 1, 1)]
    assert res["frequencies"] == ["A"]
