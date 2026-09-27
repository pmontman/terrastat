"""Crash recovery must retain completed events and never conceal log corruption."""
import json
import os
from pathlib import Path

import pytest

from terrastat import storage
from terrastat.locking import exclusive_lock


def _event(dataset_id="old", status="done", **info):
    return json.dumps({"dataset_id": dataset_id, "status": status, **info},
                      ensure_ascii=False).encode("utf-8")


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    monkeypatch.setenv("TERRASTAT_DATA_DIR", str(tmp_path))
    path = tmp_path / "state" / "eurostat.jsonl"
    path.parent.mkdir()
    return path


def test_events_survive_reload_and_last_event_wins(state_file):
    state = storage.State("eurostat")
    state.mark("one", "failed", error="temporary issue")
    state.mark("two", "done", n_obs=40)
    state.mark("one", "done", n_obs=30)
    loaded = storage.State("eurostat")
    assert loaded.events == state.events
    assert loaded.ids_with("done") == {"one", "two"}
    assert loaded.ids_with("failed") == set()
    assert loaded.status("missing") is None
    assert len(state_file.read_bytes().splitlines()) == 3


@pytest.mark.parametrize("tail", [
    b"{", b'{"dataset_id":', b'{"dataset_id":"unfinished',
    b'{"dataset_id":"unfinished","status":"done"',
    b'{"dataset_id":"unfinished","status":"failed","error":"caf\xc3',
    b'{"dataset_id":"unfinished","status":"failed","error":"x\xe2\x82',
    b'{"dataset_id":"unfinished","status":"failed","error":"\\u12',
    b'{"dataset_id":"unfinished","status":"failed","error":"slash\\',
    b'{"dataset_id":"unfinished","extra":tru',
    b'{"dataset_id":"unfinished","extra":nul',
    b'{"dataset_id":"unfinished","extra":-',
    b'{"dataset_id":"unfinished","extra":1.',
    b'{"dataset_id":"unfinished","extra":1.25e+',
    b'{"dataset_id":"unfinished","extra":[1,',
])
def test_only_incomplete_unterminated_tail_is_repaired(state_file, caplog, tail):
    prefix = _event("one") + b"\r\n\n" + _event("two", "failed") + b"\n"
    state_file.write_bytes(prefix + tail)
    with caplog.at_level("WARNING", logger="terrastat.storage"):
        state = storage.State("eurostat")
    assert state_file.read_bytes() == prefix
    assert state.status("one") == "done" and state.status("two") == "failed"
    assert "Recovered incomplete final state record" in caplog.text
    assert "discarded" in caplog.text
    state.mark("three", "done")
    assert storage.State("eurostat").ids_with("done") == {"one", "three"}
    assert len([r for r in caplog.records if "Recovered" in r.message]) == 1


@pytest.mark.parametrize("ascii_only", [False, True])
def test_recovery_handles_every_byte_boundary_in_a_normal_serialized_event(state_file, ascii_only):
    completed = _event("completed") + b"\n"
    record = json.dumps({"dataset_id": "next", "status": "failed",
                         "error": 'café € 😀; quoted "text" and a backslash \\',
                         "extra": [None, True, False, 1.25e-120, -0.5, {"x": 1}]},
                        ensure_ascii=ascii_only).encode("utf-8")
    for cut in range(1, len(record)):
        state_file.write_bytes(completed + record[:cut])
        recovered = storage.State("eurostat")
        assert recovered.ids_with("done") == {"completed"}, cut
        assert state_file.read_bytes() == completed, cut


@pytest.mark.parametrize("bad", [
    b'{"dataset_id":"x","status":"done",}',
    b'{"dataset_id":"x" "status":"done"}',
    b'{"dataset_id":"x","status":"done"}junk',
    b'{"dataset_id":"x","status":"bad\\q',
    b'{"dataset_id":"x","status":"bad\\uGG',
    b'{"dataset_id":"x","status":"bad\xff',
    b'{"dataset_id":"x","status":\xe2',
    b'{"dataset_id":"x","status":tru ',
    b'{"dataset_id":"x","status":1e2e',
    b'{"dataset_id":"x","status":1.2.',
    b'{"dataset_id":"x","status":"word".',
    b"garbage", b"[]", b"null", b"{}",
    b'{"dataset_id":"x"}', b'{"dataset_id":1,"status":"done"}',
])
def test_malformed_unterminated_final_record_is_not_silently_removed(state_file, bad):
    original = _event() + b"\n" + bad
    state_file.write_bytes(original)
    with pytest.raises(ValueError, match="state (record|event)"):
        storage.State("eurostat")
    assert state_file.read_bytes() == original


@pytest.mark.parametrize("bad", [b'{"dataset_id":', b'{"dataset_id":"bad\xc3', b"not-json"])
@pytest.mark.parametrize("later", [b"", _event("later") + b"\n"])
def test_corruption_in_a_terminated_line_is_never_repaired(state_file, bad, later):
    original = _event() + b"\n" + bad + b"\n" + later
    state_file.write_bytes(original)
    with pytest.raises(ValueError, match="corrupt state record"):
        storage.State("eurostat")
    assert state_file.read_bytes() == original


def test_valid_unterminated_last_record_is_preserved_and_separated_before_append(state_file, caplog):
    original = _event("before", description="café — quarterly data")
    state_file.write_bytes(original)
    state = storage.State("eurostat")
    assert state_file.read_bytes() == original
    assert state.status("before") == "done"
    state.mark("after", "done")
    assert state_file.read_bytes().startswith(original + b"\n")
    assert storage.State("eurostat").ids_with("done") == {"before", "after"}
    assert "Recovered" not in caplog.text


def test_multiple_live_instances_reload_new_records_before_appending(state_file):
    first = storage.State("eurostat")
    second = storage.State("eurostat")
    first.mark("one", "done")
    second.mark("two", "done")
    assert second.ids_with("done") == {"one", "two"}
    first.mark("three", "done")
    assert first.ids_with("done") == {"one", "two", "three"}
    assert storage.State("eurostat").events == first.events


def test_stale_instance_recovers_new_partial_tail_before_its_append(state_file):
    state = storage.State("eurostat")
    state.mark("one", "done")
    with state_file.open("ab") as stream:
        stream.write(_event("two") + b"\n" + b'{"dataset_id":"interrupted')
    state.mark("three", "done")
    assert state.ids_with("done") == {"one", "two", "three"}
    assert storage.State("eurostat").events == state.events


@pytest.mark.parametrize("change", ["replace", "truncate", "same_size_edit"])
def test_changed_logs_are_reloaded_instead_of_reusing_stale_events(state_file, change):
    state = storage.State("eurostat")
    state.mark("old", "done")
    if change == "replace":
        replacement = state_file.with_suffix(".replacement")
        replacement.write_bytes(_event("new") + b"\n")
        os.replace(replacement, state_file)
    elif change == "truncate":
        state_file.write_bytes(b"")
    else:
        original = state_file.read_bytes()
        state_file.write_bytes(original.replace(b'"old"', b'"new"'))
        # Force a changed timestamp even on a coarse-timestamp filesystem.
        info = state_file.stat()
        os.utime(state_file, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
    state.mark("latest", "done")
    assert "old" not in state.events
    assert state.ids_with("done") == ({"latest"} if change == "truncate" else {"new", "latest"})
    assert storage.State("eurostat").events == state.events


def test_append_flushes_to_disk_before_advancing_in_memory_state(state_file, monkeypatch):
    state = storage.State("eurostat")
    original_fsync = os.fsync
    called = []

    def checked_fsync(fd):
        assert state.events == {}
        assert json.loads(state_file.read_bytes())["dataset_id"] == "new"
        called.append(fd)
        original_fsync(fd)

    monkeypatch.setattr(storage.os, "fsync", checked_fsync)
    state.mark("new", "done")
    assert called and state.status("new") == "done"


def test_fsync_failure_does_not_report_success_in_memory(state_file, monkeypatch):
    state = storage.State("eurostat")
    state.mark("old", "done")
    before = state.events.copy()

    def failed_fsync(fd):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(storage.os, "fsync", failed_fsync)
    with pytest.raises(OSError, match="simulated storage failure"):
        state.mark("new", "done")
    assert state.events == before and state.status("new") is None


def test_partial_write_failure_preserves_memory_and_is_recoverable(state_file, monkeypatch):
    state = storage.State("eurostat")
    state.mark("old", "done")
    before = state.events.copy()
    original_open = Path.open

    class InterruptedAppend:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, data):
            self.stream.write(data[:10])
            self.stream.flush()
            raise OSError("simulated interrupted append")

    def interrupted_open(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return InterruptedAppend(stream) if path == state_file and mode == "ab" else stream

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", interrupted_open)
        with pytest.raises(OSError, match="simulated interrupted append"):
            state.mark("new", "done")
    assert state.events == before
    recovered = storage.State("eurostat")
    assert recovered.events == before
    recovered.mark("retry", "done")
    assert storage.State("eurostat").ids_with("done") == {"old", "retry"}


def test_writer_exclusion_covers_both_repair_and_append(state_file):
    state = storage.State("eurostat")
    state.mark("old", "done")
    original = state_file.read_bytes() + b'{"dataset_id":"partial'
    state_file.write_bytes(original)
    before = state.events.copy()
    with exclusive_lock(state.lock_path):
        with pytest.raises(RuntimeError, match="writer lock"):
            storage.State("eurostat")
        with pytest.raises(RuntimeError, match="writer lock"):
            state.mark("new", "done")
        assert state_file.read_bytes() == original and state.events == before
    assert storage.State("eurostat").events == before
    state.mark("new", "done")
    assert storage.State("eurostat").ids_with("done") == {"old", "new"}
