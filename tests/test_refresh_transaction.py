"""Publication failures and process interruption cannot silently mix releases."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from terrastat import refresh_transaction as tx
from terrastat.config import data_workspace
from terrastat.storage import State


def _put(root, path, content):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def _fixture(root):
    old = {"datasets/eurostat/demo/data": b"old observations", "datasets/eurostat/demo/meta": b"old meta",
           "raw/eurostat/demo/data": b"old raw", "series/eurostat/freq=M/demo.parquet": b"old monthly",
           "cache/eurostat/codes.json": b"old labels"}
    new = {"datasets/eurostat/demo/data": b"new observations", "datasets/eurostat/demo/meta": b"new meta",
           "raw/eurostat/demo/data": b"new raw", "series/eurostat/freq=Q/demo.parquet": b"new quarterly",
           "cache/eurostat/codes.json": b"new labels"}
    for path, value in old.items():
        _put(root, path, value)
    txn = tx.begin(root, "eurostat", "demo")
    for path, value in new.items():
        _put(txn / "stage", path, value)
    paths = ["datasets/eurostat/demo", "raw/eurostat/demo", "series/eurostat/freq=M/demo.parquet",
             "series/eurostat/freq=Q/demo.parquet", "cache/eurostat/codes.json"]
    updates = [{"source": "eurostat", "dataset_id": "demo", "status": "done", "extra": {"n_obs": 2}}]
    return txn, paths, updates, old, new


def _assert_files(root, wanted):
    actual = {p.relative_to(root).as_posix(): p.read_bytes() for layer in ("datasets", "raw", "series", "cache")
              for p in (root / layer).rglob("*") if p.is_file()}
    assert actual == wanted


def test_publish_and_state_commit(tmp_path):
    txn, paths, updates, old, new = _fixture(tmp_path)
    tx.publish(txn, paths, updates)
    _assert_files(tmp_path, new)
    assert not txn.exists()
    with data_workspace(tmp_path):
        assert State("eurostat").events["demo"]["n_obs"] == 2
    tx.abort(txn)  # Finally blocks may call abort after completed publication.
    tx.recover(tmp_path)


@pytest.mark.parametrize("failure_at", range(1, 9))
def test_each_rename_failure_restores_all_layers(tmp_path, monkeypatch, failure_at):
    txn, paths, updates, old, new = _fixture(tmp_path)
    original = tx._rename
    calls = 0
    def fail_once(source, target):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise OSError("injected rename failure")
        original(source, target)
    monkeypatch.setattr(tx, "_rename", fail_once)
    with pytest.raises(OSError, match="injected"):
        tx.publish(txn, paths, updates)
    _assert_files(tmp_path, old)
    assert not (tmp_path / "state/eurostat.jsonl").exists()
    tx.abort(txn)
    tx.recover(tmp_path)


@pytest.mark.parametrize("stop", ["building", "rename1", "rename2", "rename5", "committed"])
def test_killed_process_is_recovered(tmp_path, stop):
    txn, paths, updates, old, new = _fixture(tmp_path)
    script = '''
import json, os, sys
from pathlib import Path
from terrastat import refresh_transaction as tx
txn, paths, updates, stop = Path(sys.argv[1]), json.loads(sys.argv[2]), json.loads(sys.argv[3]), sys.argv[4]
if stop == 'building': os._exit(73)
if stop == 'committed':
    tx._finish = lambda *args: os._exit(73)
else:
    original = tx._rename
    calls = 0
    def interrupted(source, target):
        global calls
        original(source, target)
        calls += 1
        if stop == 'rename' + str(calls): os._exit(73)
    tx._rename = interrupted
tx.publish(txn, paths, updates)
'''
    import json
    child = subprocess.run([sys.executable, "-c", script, str(txn), json.dumps(paths), json.dumps(updates), stop],
                           capture_output=True, text=True, timeout=40)
    assert child.returncode == 73, child.stderr
    tx.recover(tmp_path)
    _assert_files(tmp_path, new if stop == "committed" else old)
    assert not txn.exists()
    tx.recover(tmp_path)


def test_committed_state_failure_is_replayed_without_reverting_data(tmp_path, monkeypatch):
    txn, paths, updates, old, new = _fixture(tmp_path)
    original = State.mark
    monkeypatch.setattr(State, "mark", lambda *a, **k: (_ for _ in ()).throw(OSError("state unavailable")))
    with pytest.raises(tx.CommittedRefreshError):
        tx.publish(txn, paths, updates)
    _assert_files(tmp_path, new)
    tx.abort(txn)
    assert txn.exists()
    monkeypatch.setattr(State, "mark", original)
    tx.recover(tmp_path)
    with data_workspace(tmp_path):
        assert State("eurostat").status("demo") == "done"
    before = (tmp_path / "state/eurostat.jsonl").read_bytes()
    tx.recover(tmp_path)
    assert (tmp_path / "state/eurostat.jsonl").read_bytes() == before


@pytest.mark.parametrize("path", ["../outside", "snapshot/frozen", "datasets/oecd/demo", "raw/eurostat/other",
                                 "datasets/eurostat/demo/../other", "cache/eurostat/CON", "C:/outside"])
def test_unsafe_publication_paths_are_rejected(tmp_path, path):
    txn, paths, updates, old, new = _fixture(tmp_path)
    with pytest.raises(ValueError):
        tx.publish(txn, [path], [])
    _assert_files(tmp_path, old)
    tx.abort(txn)


def test_interrupted_rollback_removal_can_be_recovered(tmp_path, monkeypatch):
    txn, paths, updates, old, new = _fixture(tmp_path)
    original_rename, original_remove = tx._rename, tx._remove
    calls = 0
    def fail_later(source, target):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("publish failed")
        original_rename(source, target)
    def partial_remove(path, container):
        if path == tmp_path / "datasets/eurostat/demo":
            (path / "data").unlink()
            raise OSError("rollback interrupted")
        original_remove(path, container)
    monkeypatch.setattr(tx, "_rename", fail_later)
    monkeypatch.setattr(tx, "_remove", partial_remove)
    with pytest.raises(tx.RecoveryRequiredError):
        tx.publish(txn, paths, updates)
    monkeypatch.setattr(tx, "_rename", original_rename)
    monkeypatch.setattr(tx, "_remove", original_remove)
    tx.recover(tmp_path)
    _assert_files(tmp_path, old)
