"""The classification log and the snapshot it builds.

No Qt here — the whole point of moving classification into a log is that the
decision model is plain data, replayable and testable without a session, a
window, or a single file being moved.
"""

from __future__ import annotations

import json

import pytest

from vidtriage.core.errors import FileOperationError
from vidtriage.plugins.builtin.triage.ledger import (
    Decision,
    Ledger,
    NullLedger,
    default_log_dir,
    discover_logs,
    identity_for,
    new_log_path,
    now_stamp,
    read_log,
    replay,
    summarise,
)
from vidtriage.plugins.builtin.triage.models import ClassEntry, VideoItem
from vidtriage.plugins.builtin.triage.session import Session
from vidtriage.plugins.builtin.triage.snapshot import plan_snapshot, write_snapshot


def write_log(path, *pairs):
    """A log holding ``(file, class)`` decisions, in order."""
    ledger = Ledger(path)
    for file, class_name in pairs:
        ledger.append(Decision(file, class_name, now_stamp()))
    return path


@pytest.fixture
def corpus(tmp_path):
    """An input directory of three tiny files standing in for videos."""
    source = tmp_path / "in"
    source.mkdir()
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        (source / name).write_bytes(b"video-" + name.encode())
    return source


class TestIdentity:
    def test_relative_to_the_input_dir(self, corpus):
        """Relative keys survive the corpus being moved or remounted."""
        assert identity_for(corpus / "a.mp4", corpus) == "a.mp4"

    def test_nested_files_keep_their_subpath(self, corpus):
        nested = corpus / "day1" / "a.mp4"
        nested.parent.mkdir()
        nested.touch()
        assert identity_for(nested, corpus) == "day1/a.mp4"

    def test_files_outside_the_corpus_are_absolute(self, corpus, tmp_path):
        stray = tmp_path / "elsewhere.mp4"
        stray.touch()
        assert identity_for(stray, corpus) == str(stray.resolve())


class TestLedger:
    def test_appends_one_line_per_decision_plus_a_header(self, tmp_path):
        path = write_log(tmp_path / "l.jsonl", ("a.mp4", "cat"), ("b.mp4", "dog"))
        lines = [json.loads(line) for line in path.read_text().splitlines()]

        assert lines[0]["kind"] == "vidtriage-triage-log"
        assert [(r["file"], r["class"]) for r in lines[1:]] == [("a.mp4", "cat"), ("b.mp4", "dog")]

    def test_never_rewrites_an_earlier_record(self, tmp_path):
        path = write_log(tmp_path / "l.jsonl", ("a.mp4", "cat"), ("a.mp4", None))
        assert [d.class_name for d in read_log(path)] == ["cat", None]

    def test_a_write_failure_does_not_raise(self, tmp_path):
        """Losing a line costs one re-key; a dialog costs the session."""
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("")
        Ledger(blocker / "nested" / "l.jsonl").append(Decision("a.mp4", "cat"))

    def test_null_ledger_writes_nothing(self, tmp_path):
        ledger = NullLedger()
        ledger.append(Decision("a.mp4", "cat"))
        assert not ledger.records

    def test_a_torn_line_does_not_lose_the_rest(self, tmp_path):
        path = tmp_path / "l.jsonl"
        write_log(path, ("a.mp4", "cat"))
        with path.open("a") as handle:
            handle.write('{"file": "b.mp4", "cla\n')          # truncated by a crash
            handle.write('{"file": "c.mp4", "class": "dog"}\n')

        assert [(d.file, d.class_name) for d in read_log(path)] == [
            ("a.mp4", "cat"), ("c.mp4", "dog"),
        ]

    def test_a_record_with_no_file_is_skipped(self, tmp_path):
        path = tmp_path / "l.jsonl"
        path.write_text('{"class": "cat"}\n{"file": "b.mp4", "class": "dog"}\n')
        assert [d.file for d in read_log(path)] == ["b.mp4"]

    def test_summarise_reports_count_and_span(self, tmp_path):
        path = write_log(tmp_path / "l.jsonl", ("a.mp4", "cat"), ("b.mp4", "dog"))
        count, first, last = summarise(path)
        assert count == 2
        assert first and last


class TestReplay:
    def test_last_record_in_a_log_wins(self, tmp_path):
        path = write_log(tmp_path / "l.jsonl", ("a.mp4", "cat"), ("a.mp4", "dog"))
        assert replay([path]) == {"a.mp4": "dog"}

    def test_a_later_log_overrides_an_earlier_one(self, tmp_path):
        """The overlay rule — the reason a rerun is a separate log."""
        first = write_log(tmp_path / "1.jsonl", ("a.mp4", "cat"), ("b.mp4", "cat"))
        second = write_log(tmp_path / "2.jsonl", ("a.mp4", "dog"))

        assert replay([first, second]) == {"a.mp4": "dog", "b.mp4": "cat"}

    def test_order_is_the_argument_order_not_the_filename(self, tmp_path):
        first = write_log(tmp_path / "1.jsonl", ("a.mp4", "cat"))
        second = write_log(tmp_path / "2.jsonl", ("a.mp4", "dog"))

        assert replay([second, first])["a.mp4"] == "cat"

    def test_dropping_a_log_un_applies_it(self, tmp_path):
        first = write_log(tmp_path / "1.jsonl", ("a.mp4", "cat"))
        second = write_log(tmp_path / "2.jsonl", ("a.mp4", "dog"))

        assert replay([first, second])["a.mp4"] == "dog"
        assert replay([first])["a.mp4"] == "cat"

    def test_a_null_record_returns_a_file_to_pending(self, tmp_path):
        first = write_log(tmp_path / "1.jsonl", ("a.mp4", "cat"))
        second = write_log(tmp_path / "2.jsonl", ("a.mp4", None))
        assert replay([first, second]) == {"a.mp4": None}

    def test_a_missing_log_is_skipped_not_fatal(self, tmp_path):
        path = write_log(tmp_path / "1.jsonl", ("a.mp4", "cat"))
        assert replay([tmp_path / "gone.jsonl", path]) == {"a.mp4": "cat"}


class TestLogDiscovery:
    def test_the_directory_is_stable_across_processes(self, corpus):
        """A salted hash would hand every launch a fresh, empty log directory.

        Run under two different ``PYTHONHASHSEED`` values, which is exactly what
        would make the built-in ``hash()`` disagree between launches.
        """
        import os
        import subprocess
        import sys

        script = (
            "from vidtriage.plugins.builtin.triage.ledger import default_log_dir;"
            f"print(default_log_dir({str(corpus)!r}).name)"
        )
        names = {
            subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, check=True,
                env={**os.environ, "PYTHONHASHSEED": seed},
            ).stdout.strip()
            for seed in ("1", "2")
        }
        assert names == {default_log_dir(corpus).name}

    def test_discovery_is_oldest_first(self, tmp_path):
        for name in ("20260101T000000Z", "20260301T000000Z", "20260201T000000Z"):
            (tmp_path / f"{name}.triage.jsonl").touch()
        (tmp_path / "notes.txt").touch()

        found = [p.name for p in discover_logs(tmp_path)]
        assert found == [
            "20260101T000000Z.triage.jsonl",
            "20260201T000000Z.triage.jsonl",
            "20260301T000000Z.triage.jsonl",
        ]

    def test_a_new_log_never_collides(self, tmp_path):
        first = new_log_path(tmp_path)
        first.touch()
        assert new_log_path(tmp_path) != first


class TestSessionRecording:
    def test_classify_writes_a_record_and_moves_nothing(self, corpus, tmp_path):
        session = Session(corpus, tmp_path / "out", [ClassEntry("1", "cat")], logs=[])
        session.load()
        item = session.pending[0]

        session.classify(item, ClassEntry("1", "cat"))

        assert item.original_path.exists()
        assert sorted(p.name for p in corpus.iterdir()) == ["a.mp4", "b.mp4", "c.mp4"]
        assert [(d.file, d.class_name) for d in read_log(session.log_path)] == [("a.mp4", "cat")]

    def test_a_second_run_overlays_the_first(self, corpus, tmp_path):
        """Two passes disagreeing is the workflow, not an error."""
        first = Session(corpus, tmp_path / "out", [ClassEntry("1", "cat")], logs=[])
        first.load()
        first.classify(first.pending[0], ClassEntry("1", "cat"))

        second = Session(
            corpus, tmp_path / "out", [ClassEntry("2", "dog")], logs=[first.log_path],
        )
        second.load()
        assert second.classified[0].class_name == "cat"

        second.classify(second.classified[0], ClassEntry("2", "dog"))

        third = Session(
            corpus, tmp_path / "out", [], logs=[first.log_path, second.log_path],
        )
        third.load()
        assert third.classified[0].class_name == "dog"

    def test_undo_records_the_previous_state(self, corpus, tmp_path):
        session = Session(corpus, tmp_path / "out", [ClassEntry("1", "cat")], logs=[])
        session.load()
        item = session.pending[0]

        session.classify(item, ClassEntry("1", "cat"))
        assert session.can_undo
        session.undo_last()

        assert item.is_pending
        assert [d.class_name for d in read_log(session.log_path)] == ["cat", None]

    def test_undo_with_nothing_to_undo_is_harmless(self, corpus, tmp_path):
        session = Session(corpus, tmp_path / "out", [], logs=[])
        session.load()
        assert session.undo_last() is None

    def test_a_replayed_class_is_added_to_the_class_list(self, corpus, tmp_path):
        """Otherwise the key bindings would not cover what the log describes."""
        log = write_log(tmp_path / "1.jsonl", ("a.mp4", "hedgehog"))
        session = Session(corpus, tmp_path / "out", [], logs=[log])
        session.load()

        assert session.classified[0].class_name == "hedgehog"
        assert any(c.name == "hedgehog" for c in session.classes)

    def test_a_decision_for_a_vanished_file_is_ignored(self, corpus, tmp_path):
        log = write_log(tmp_path / "1.jsonl", ("gone.mp4", "cat"))
        session = Session(corpus, tmp_path / "out", [], logs=[log])
        session.load()
        assert len(session.pending) == 3
        assert not session.classified

    def test_a_read_only_session_writes_no_log(self, corpus, tmp_path):
        """--snapshot must not turn every build into another overlay layer."""
        log = write_log(tmp_path / "1.jsonl", ("a.mp4", "cat"))
        before = set(default_log_dir(corpus).glob("*")) if default_log_dir(corpus).is_dir() else set()

        session = Session(corpus, tmp_path / "out", [], logs=[log], record=False)
        session.load()
        session.classify(session.pending[0], ClassEntry("2", "dog"))

        after = set(default_log_dir(corpus).glob("*")) if default_log_dir(corpus).is_dir() else set()
        assert after == before


class TestLegacyImport:
    @pytest.fixture
    def moved_corpus(self, tmp_path):
        """What an old, move-based run left behind: files inside class folders."""
        source = tmp_path / "in"
        output = tmp_path / "out"
        source.mkdir()
        (source / "still-pending.mp4").write_bytes(b"x")
        for folder, name in (("cat", "one.mp4"), ("dog", "two.mp4"), ("_errors", "bad.mp4")):
            (output / folder).mkdir(parents=True)
            (output / folder / name).write_bytes(b"x")
        return source, output

    def test_adopts_already_classified_videos(self, moved_corpus):
        source, output = moved_corpus
        session = Session(source, output, [], logs=[])
        session.load()

        assert {i.name: i.class_name for i in session.classified} == {
            "one.mp4": "cat", "two.mp4": "dog", "bad.mp4": "_errors",
        }
        assert [i.name for i in session.pending] == ["still-pending.mp4"]

    def test_leaves_the_files_exactly_where_they_are(self, moved_corpus):
        source, output = moved_corpus
        Session(source, output, [], logs=[]).load()

        assert (output / "cat" / "one.mp4").exists()
        assert not (source / "one.mp4").exists()

    def test_writes_the_decisions_into_a_log_so_it_only_happens_once(self, moved_corpus):
        source, output = moved_corpus
        first = Session(source, output, [], logs=[])
        first.load()
        assert len(read_log(first.log_path)) == 3

        second = Session(source, output, [], logs=[first.log_path])
        second.load()
        assert len(second.classified) == 3
        assert not second.log_path.exists(), "nothing new to import, so nothing written"

    def test_class_names_come_from_the_folder_names(self, moved_corpus):
        source, output = moved_corpus
        session = Session(source, output, [], logs=[])
        session.load()
        assert {c.name for c in session.classes} == {"cat", "dog"}


class TestSnapshot:
    @pytest.fixture
    def classified(self, corpus):
        items = []
        for name, class_name in (("a.mp4", "cat"), ("b.mp4", "dog"), ("c.mp4", "cat")):
            item = VideoItem(original_path=corpus / name)
            item.history.append(class_name)
            items.append(item)
        return items

    def test_copies_into_class_folders(self, classified, tmp_path):
        result = write_snapshot(classified, tmp_path / "snap")

        assert result.written == 3
        assert (tmp_path / "snap" / "cat" / "a.mp4").read_bytes() == b"video-a.mp4"
        assert (tmp_path / "snap" / "dog" / "b.mp4").exists()
        assert (tmp_path / "snap" / "cat" / "c.mp4").exists()

    def test_the_sources_are_untouched(self, classified, corpus, tmp_path):
        write_snapshot(classified, tmp_path / "snap")
        assert sorted(p.name for p in corpus.iterdir()) == ["a.mp4", "b.mp4", "c.mp4"]

    def test_pending_videos_are_left_out(self, classified, corpus, tmp_path):
        classified.append(VideoItem(original_path=corpus / "a.mp4"))
        plan = plan_snapshot(classified, tmp_path / "snap")
        assert plan.skipped_pending == 1

    def test_refuses_a_non_empty_target(self, classified, tmp_path):
        """Writing into one would leave stale copies in old class folders."""
        target = tmp_path / "snap"
        target.mkdir()
        (target / "leftover.txt").write_text("from a previous run")

        with pytest.raises(FileOperationError, match="not empty"):
            write_snapshot(classified, target)

    def test_an_existing_empty_target_is_fine(self, classified, tmp_path):
        target = tmp_path / "snap"
        target.mkdir()
        assert write_snapshot(classified, target).written == 3

    def test_a_collision_aborts_before_writing_anything(self, corpus, tmp_path):
        nested = corpus / "day2"
        nested.mkdir()
        (nested / "a.mp4").write_bytes(b"a different a.mp4")

        items = []
        for path in (corpus / "a.mp4", nested / "a.mp4"):
            item = VideoItem(original_path=path)
            item.history.append("cat")
            items.append(item)

        target = tmp_path / "snap"
        with pytest.raises(FileOperationError, match="same place"):
            write_snapshot(items, target)
        assert not target.exists(), "a refused snapshot must write nothing at all"

    def test_nothing_classified_is_an_error_not_an_empty_directory(self, tmp_path):
        with pytest.raises(FileOperationError, match="Nothing to snapshot"):
            write_snapshot([], tmp_path / "snap")

    def test_a_vanished_source_is_a_warning_not_a_failure(self, classified, tmp_path):
        classified[0].original_path.unlink()
        result = write_snapshot(classified, tmp_path / "snap")

        assert result.written == 2
        assert any("a.mp4" in warning for warning in result.warnings)

    def test_hardlinks_share_storage_with_the_source(self, classified, tmp_path):
        result = write_snapshot(classified, tmp_path / "snap", link=True)
        copied = tmp_path / "snap" / "cat" / "a.mp4"

        assert result.written == 3
        assert copied.stat().st_ino == classified[0].original_path.stat().st_ino

    def test_link_falls_back_to_copy_and_still_produces_the_file(
        self, classified, tmp_path, monkeypatch,
    ):
        """Cross-filesystem is the common case; it must not fail the snapshot."""
        import os

        def refuse(*_args, **_kwargs):
            raise OSError("Invalid cross-device link")

        monkeypatch.setattr(os, "link", refuse)
        result = write_snapshot(classified, tmp_path / "snap", link=True)

        assert result.written == 3
        assert (tmp_path / "snap" / "cat" / "a.mp4").read_bytes() == b"video-a.mp4"
