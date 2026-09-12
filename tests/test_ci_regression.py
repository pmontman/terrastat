"""Two things CI caught that local runs never did."""
import datetime as dt

from terrastat import corpus, site
from tests.test_corpus import corpus_root  # noqa: F401


def test_check_still_passes_on_a_later_day(corpus_root, tmp_path, monkeypatch):  # noqa: F811
    """The pages say when they were generated. Taking that from the clock at render time meant
    `--check` re-rendered with today's date and could never match a page built yesterday -- so
    CI failed on the first push made a day after the figures were computed."""
    md, page, tb = tmp_path / "corpus.md", tmp_path / "index.html", tmp_path / "t.json"
    assert corpus.main(["--out", str(md), "--html", str(page), "--tables", str(tb)]) == 0

    class Tomorrow(dt.date):
        @classmethod
        def today(cls):
            return dt.date(2099, 1, 1)

    monkeypatch.setattr(corpus.dt, "date", Tomorrow)
    monkeypatch.setattr(site.dt, "date", Tomorrow)
    assert corpus.check(md, page, tb) == []
    assert "2099" not in page.read_text(encoding="utf-8")


def test_the_generation_stamp_round_trips(corpus_root, tmp_path):  # noqa: F811
    t = corpus.compute(values=False, concentration=False)
    assert t["generated"] == dt.date.today().isoformat()
    p = corpus.save_tables(t, tmp_path / "t.json")
    assert corpus.load_tables(p)["generated"] == t["generated"]


def test_the_shared_fixture_imports_under_bare_pytest():
    """`pytest` and `python -m pytest` differ in whether the repo root is importable. The
    pyproject setting is what makes the fixture import above work in both; if it goes, this file
    fails to collect at all, which is the point."""
    import tomllib
    from pathlib import Path

    cfg = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text("utf-8"))
    assert "." in cfg["tool"]["pytest"]["ini_options"]["pythonpath"]
