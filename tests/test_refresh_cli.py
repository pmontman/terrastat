"""The refresh command selects local inputs and preserves legacy reprocessing semantics."""
import json
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from terrastat import cli


@pytest.fixture
def refresh_call(monkeypatch):
    monkeypatch.setattr(cli, "_setup_logging", lambda *args, **kwargs: None)
    call = Mock(return_value={
        "checked": 3, "unchanged": 2, "updated": 1, "failed": 0,
        "stopped_early": False, "report": "data/refresh/reports/example.json", "results": [],
    })
    module = ModuleType("terrastat.refresh")
    module.run_refresh = call
    monkeypatch.setitem(sys.modules, "terrastat.refresh", module)
    return call


def test_refresh_defaults_to_locally_collected_sources(refresh_call, capsys):
    assert cli.main(["refresh"]) == 0
    refresh_call.assert_called_once_with(
        sources=None, ids=None, force=False, min_interval=None,
        max_per_minute=None, max_hours=None, limit=None,
    )
    out = capsys.readouterr().out
    assert "Checked 3: 2 unchanged, 1 updated, 0 failed." in out
    assert "Report: data/refresh/reports/example.json" in out


def test_refresh_forwards_explicit_selection_and_download_controls(refresh_call):
    assert cli.main(["refresh", "eurostat", "--ids", "une_rt_q", "namq_10_gdp",
                     "--force", "--limit", "2", "--max-hours", "0.5",
                     "--min-interval", "0", "--max-per-minute", "4"]) == 0
    refresh_call.assert_called_once_with(
        sources=["eurostat"], ids=["une_rt_q", "namq_10_gdp"], force=True,
        min_interval=0.0, max_per_minute=4, max_hours=0.5, limit=2,
    )


@pytest.mark.parametrize("arguments, sources, ids", [
    (["--sources", "fred", "oecd"], ["fred", "oecd"], None),
    (["--sources", "eurostat", "--ids", "une_rt_q"], ["eurostat"], ["une_rt_q"]),
])
def test_refresh_accepts_named_source_selection(refresh_call, arguments, sources, ids):
    assert cli.main(["refresh", *arguments]) == 0
    assert refresh_call.call_args.kwargs["sources"] == sources
    assert refresh_call.call_args.kwargs["ids"] == ids


@pytest.mark.parametrize("arguments", [
    ["eurostat", "--sources", "fred"],
    ["--ids", "une_rt_q"],
    ["--sources", "eurostat", "fred", "--ids", "une_rt_q"],
    ["--sources"],
    ["eurostat", "--ids"],
    ["--limit", "0"],
    ["--limit", "-1"],
    ["--limit", "1.5"],
    ["--max-hours", "0"],
    ["--max-hours", "nan"],
    ["--max-hours", "inf"],
    ["--min-interval", "-1"],
    ["--min-interval", "nan"],
    ["--max-per-minute", "0"],
    ["--no-series"],
])
def test_invalid_refresh_arguments_do_not_start_work(refresh_call, arguments):
    with pytest.raises(SystemExit) as exc:
        cli.main(["refresh", *arguments])
    assert exc.value.code == 2
    refresh_call.assert_not_called()


def test_refresh_json_output_and_failure_exit_code(refresh_call, capsys):
    refresh_call.return_value.update(failed=1, updated=0)
    assert cli.main(["refresh", "--json"]) == 1
    assert json.loads(capsys.readouterr().out) == refresh_call.return_value


def test_refresh_budget_stop_is_reported_without_claiming_failure(refresh_call, capsys):
    refresh_call.return_value["stopped_early"] = True
    assert cli.main(["refresh"]) == 0
    assert "Stopped early" in capsys.readouterr().out


def test_refresh_keyboard_interrupt_returns_130(refresh_call, capsys):
    refresh_call.side_effect = KeyboardInterrupt
    assert cli.main(["refresh"]) == 130
    assert "interrupted" in capsys.readouterr().err


def test_refresh_saved_interruption_report_returns_130(refresh_call, capsys):
    refresh_call.return_value.update(interrupted=True, stopped_early=True)
    assert cli.main(["refresh"]) == 130
    out = capsys.readouterr().out
    assert "Interrupted" in out and "Report:" in out


def test_refresh_local_failure_is_reported(refresh_call, capsys):
    refresh_call.side_effect = RuntimeError("another operation may be running")
    assert cli.main(["refresh"]) == 1
    assert "another operation" in capsys.readouterr().err


@pytest.mark.parametrize("flag, deprecated", [("--reprocess", False), ("--force", True)])
def test_fetch_reprocess_keeps_legacy_force_behavior_with_targeted_warning(
        refresh_call, monkeypatch, capsys, flag, deprecated):
    from terrastat import pipeline

    monkeypatch.setattr(pipeline, "load_catalog", Mock(return_value=object()))
    monkeypatch.setattr(pipeline, "select_refs", Mock(return_value=[]))
    fetch = Mock(return_value={"failed": 0})
    monkeypatch.setattr(pipeline, "run_fetch", fetch)
    assert cli.main(["fetch", "eurostat", flag]) == 0
    assert fetch.call_args.kwargs["force"] is True
    refresh_call.assert_not_called()
    err = capsys.readouterr().err
    assert ("deprecated" in err) is deprecated
    if deprecated:
        assert "cached raw data" in err and "terrastat refresh" in err


def test_refresh_force_is_not_deprecated(refresh_call, capsys):
    assert cli.main(["refresh", "--force"]) == 0
    assert refresh_call.call_args.kwargs["force"] is True
    assert "deprecated" not in capsys.readouterr().err


def test_committed_refresh_with_pending_state_requires_recovery(refresh_call, capsys):
    refresh_call.return_value.update(results=[{"status": "updated", "state_pending": True}])
    assert cli.main(["refresh"]) == 1
    assert "bookkeeping needs recovery" in capsys.readouterr().out
