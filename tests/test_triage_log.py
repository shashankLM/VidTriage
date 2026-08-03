"""The classification log and the snapshot it builds.

No Qt here — the whole point of moving classification into a log is that the
decision model is plain data, replayable and testable without a session, a
window, or a single file being moved.
"""

from __future__ import annotations

import json

import pytest

from label_kit.core.errors import FileOperationError
from label_kit.plugins.builtin.triage.ledger import (
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
from label_kit.plugins.builtin.triage.models import ClassEntry, MediaItem
from label_kit.plugins.builtin.triage.session import Session
from label_kit.plugins.builtin.triage.snapshot import plan_snapshot, write_snapshot


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

        assert lines[0]["kind"] == "labelkit-triage-log"
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
            "from label_kit.plugins.builtin.triage.ledger import default_log_dir;"
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

    def test_two_runs_in_one_second_still_replay_in_order(self, tmp_path):
        """The collision suffix must not reorder the stack.

        ``new_log_path`` disambiguates a same-second collision with ``-1``, and
        ``-`` sorts before ``Z`` — so plain name order replays the *newer* pass
        first and lets the older one win. That silently inverts the one rule the
        overlay model rests on, and nothing downstream can detect it.
        """
        first = new_log_path(tmp_path)
        write_log(first, ("a.mp4", "early"))
        second = new_log_path(tmp_path)
        write_log(second, ("a.mp4", "late"))

        assert second.name.startswith(first.name[: -len(".triage.jsonl")]), (
            "this test is meaningless unless both logs landed in the same second"
        )
        assert [p.name for p in discover_logs(tmp_path)] == [first.name, second.name]
        assert replay(discover_logs(tmp_path)) == {"a.mp4": "late"}

    def test_a_hand_placed_log_sorts_after_the_generated_ones(self, tmp_path):
        """An unparseable name has no place in the history, so it goes last."""
        for name in ("20260101T000000Z", "20260301T000000Z"):
            (tmp_path / f"{name}.triage.jsonl").touch()
        (tmp_path / "from-a-colleague.triage.jsonl").touch()

        assert [p.name for p in discover_logs(tmp_path)] == [
            "20260101T000000Z.triage.jsonl",
            "20260301T000000Z.triage.jsonl",
            "from-a-colleague.triage.jsonl",
        ]

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

    def test_emptying_the_log_stack_does_not_import_again(self, moved_corpus):
        """Dropping every log means "ignore those passes", not "never triaged".

        Reading it the second way rescans the output directory — which by then
        may hold a snapshot this tool wrote — and fabricates a full set of
        classifications the user never made, writing them into a fresh log.
        """
        source, output = moved_corpus
        Session(source, output, [], logs=[]).load()  # the one real import

        cleared = Session(source, output, [], logs=[])
        cleared.load()

        assert cleared.classified == [], "the import ran a second time"
        assert not cleared.log_path.exists(), "and wrote invented decisions to a log"

    def test_a_snapshot_in_the_output_dir_is_never_adopted(self, tmp_path):
        """The likeliest shape of the bug: output holds this tool's own snapshot."""
        source = tmp_path / "in"
        source.mkdir()
        (source / "a.mp4").write_bytes(b"x")

        session = Session(source, tmp_path / "out", [ClassEntry("1", "cat")], logs=[])
        session.load()
        session.classify(session.pending[0], ClassEntry("1", "cat"))
        write_snapshot(session.classified, tmp_path / "out" / "snap")

        # Reopened with the snapshot as the output directory and the real log
        # stack, which is what the app does on the next launch.
        reopened = Session(source, tmp_path / "out" / "snap", [])
        reopened.load()
        assert [i.original_path for i in reopened.classified] == [source / "a.mp4"], (
            "the snapshot's own copies were adopted as extra classified files"
        )


class TestSnapshot:
    @pytest.fixture
    def classified(self, corpus):
        items = []
        for name, class_name in (("a.mp4", "cat"), ("b.mp4", "dog"), ("c.mp4", "cat")):
            item = MediaItem(original_path=corpus / name)
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
        classified.append(MediaItem(original_path=corpus / "a.mp4"))
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
            item = MediaItem(original_path=path)
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


class TestReservedClassNames:
    def test_errors_cannot_be_used_as_a_class(self):
        """It is the error bucket. A class sharing the name vanishes into it."""
        from label_kit.plugins.builtin.triage.config import parse_classes

        entries, errors = parse_classes("cat\n_errors\ndog")
        assert [e.name for e in entries] == ["cat", "dog"]
        assert any("_errors" in message for message in errors)


class TestSnapshotCarriesAnnotations:
    """A snapshot without its sidecars is a deliverable with the labels missing."""

    def test_a_legacy_sidecar_travels_with_its_media(self, tmp_path):
        """Corpora annotated before the rename still have ``.vidtriage.json``.

        The write path looked only for the current name, so those snapshotted
        as bare media with every annotation left behind — the one direction the
        rename shim did not cover.
        """
        from label_kit.core.annotations import Annotation
        from label_kit.core.frames import FrameRef, source_id_for
        from label_kit.core.geometry import Rect
        from label_kit.persistence.sidecar import (
            LEGACY_SIDECAR_SUFFIX,
            SIDECAR_SUFFIX,
            load_annotations,
            save_annotations,
            sidecar_path_for,
        )
        from label_kit.plugins.builtin.triage.io_ops import copy_media

        source = tmp_path / "clip.mp4"
        source.write_bytes(b"x")
        # Written through the real serialiser and renamed, so this is what
        # VidTriage actually left on disk rather than an approximation of it.
        save_annotations(source, [
            Annotation(FrameRef(source_id_for(source), 0), Rect(1, 2, 3, 4), label="sign"),
        ])
        sidecar_path_for(source).rename(
            source.with_name(source.name + LEGACY_SIDECAR_SUFFIX),
        )

        destination = tmp_path / "snap" / "cat" / "clip.mp4"
        copy_media(source, destination)

        assert [a.label for a in load_annotations(destination)] == ["sign"]
        assert destination.with_name(destination.name + SIDECAR_SUFFIX).exists(), (
            "the copy should land under the current name, migrating as it goes"
        )


class TestImagesAreJustOneFrameMedia:
    """Stills need no special case anywhere — that is the design claim.

    An image is a one-frame source at frame index 0, not a sentinel like -1.
    Everything downstream — the sidecar key, the log record, the snapshot, the
    exporters — therefore works on it unchanged, and a session may freely mix
    videos and images.
    """

    @pytest.fixture
    def mixed(self, tmp_path):
        import cv2
        import numpy as np

        source = tmp_path / "mixed"
        source.mkdir()
        for name in ("photo.png", "shot.jpg"):
            cv2.imwrite(str(source / name), np.full((48, 64, 3), 90, np.uint8))
        writer = cv2.VideoWriter(
            str(source / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48),
        )
        for i in range(5):
            writer.write(np.full((48, 64, 3), i * 40, np.uint8))
        writer.release()
        return source

    def test_a_session_picks_up_images_alongside_videos(self, mixed, tmp_path):
        session = Session(mixed, tmp_path / "out", [], logs=[])
        session.load()
        assert sorted(i.name for i in session.pending) == ["clip.mp4", "photo.png", "shot.jpg"]

    def test_an_image_is_frame_zero_not_a_sentinel(self, mixed):
        """-1 would need a special case in every bounds check and exporter."""
        from label_kit.media.source import open_source

        source = open_source(mixed / "photo.png")
        try:
            assert source.info.frame_count == 1
            frame = source.read()
            assert frame.ref.index == 0
            assert source.read() is None, "a still is exactly one frame"
        finally:
            source.close()

    def test_classifying_an_image_logs_like_any_other_decision(self, mixed, tmp_path):
        session = Session(mixed, tmp_path / "out", [ClassEntry("1", "keep")], logs=[])
        session.load()
        photo = next(i for i in session.pending if i.name == "photo.png")

        session.classify(photo, ClassEntry("1", "keep"))

        assert [(d.file, d.class_name) for d in read_log(session.log_path)] == [
            ("photo.png", "keep"),
        ]

    def test_a_snapshot_carries_images_and_videos_together(self, mixed, tmp_path):
        session = Session(mixed, tmp_path / "out", [ClassEntry("1", "keep")], logs=[])
        session.load()
        for item in list(session.pending):
            session.classify(item, ClassEntry("1", "keep"))

        result = write_snapshot(session.classified, tmp_path / "snap")

        assert result.written == 3
        assert (tmp_path / "snap" / "keep" / "photo.png").exists()
        assert (tmp_path / "snap" / "keep" / "clip.mp4").exists()

    def test_an_images_annotations_ride_along_in_a_snapshot(self, mixed, tmp_path):
        from label_kit.core.annotations import Annotation
        from label_kit.core.frames import FrameRef, source_id_for
        from label_kit.core.geometry import Rect
        from label_kit.persistence.sidecar import load_annotations, save_annotations

        photo = mixed / "photo.png"
        save_annotations(photo, [
            Annotation(FrameRef(source_id_for(photo), 0), Rect(1, 1, 9, 9), label="sign"),
        ])

        session = Session(mixed, tmp_path / "out", [ClassEntry("1", "keep")], logs=[])
        session.load()
        session.classify(
            next(i for i in session.pending if i.name == "photo.png"), ClassEntry("1", "keep"),
        )
        write_snapshot(session.classified, tmp_path / "snap")

        copied = tmp_path / "snap" / "keep" / "photo.png"
        assert load_annotations(copied)[0].label == "sign"

    def test_an_image_only_directory_works(self, tmp_path):
        """No requirement that a session contain any video at all."""
        import cv2
        import numpy as np

        source = tmp_path / "stills"
        source.mkdir()
        cv2.imwrite(str(source / "only.png"), np.full((10, 10, 3), 5, np.uint8))

        session = Session(source, tmp_path / "out", [], logs=[])
        session.load()
        assert [i.name for i in session.pending] == ["only.png"]
