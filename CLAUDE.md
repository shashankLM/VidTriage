# label-kit — notes for future sessions

A PySide6 desktop tool: triage whole media files into classes, annotate
individual frames, and run models on regions you point at. Six layers —
`core` (Qt-free data model), `media` (decode), `view` (canvas/tools/layers),
`persistence` (sidecars/exporters/settings), `plugins` (contract + built-ins),
`app` (shell). Everything user-facing is a plugin, triage included.

---

## Naming

Three spellings, deliberately:

| Form | Used for |
|------|----------|
| `label_kit` | the import package (`from label_kit.core import ...`) |
| `label-kit` | the distribution, the CLI, and prose |
| `labelkit` | everything on disk — `~/.labelkit`, `*.labelkit.json`, the logger root, the `labelkit.plugins` entry-point group |

The third exists because dotfiles and file suffixes read badly with separators.
Do not "unify" them. `core/logging.get_logger` translates `label_kit.x.y` into
`labelkit.x.y`; without that step every record would read `labelkit.label_kit.x.y`.

### The project was renamed from VidTriage, and the compatibility path is live

Users have VidTriage-era data on disk. Two shims carry it forward, both tested
in `tests/test_isolation.py::TestLegacyMigration`:

* `settings.migrate_legacy_config()` — **moves** `~/.vidtriage` to `~/.labelkit`
  once, at startup, before anything reads the config directory. Moves rather
  than copies because `weights/` runs to hundreds of megabytes; `Path.rename` is
  atomic, so there is no half-migrated state. Skipped entirely when the new
  directory already exists — never merge two divergent trees.
* `sidecar.existing_sidecar_for()` — reads `*.vidtriage.json` when no
  `*.labelkit.json` exists. Writes always use the new name, so a file migrates
  the first time its annotations are saved.

`delete_sidecar` removes **both** names. Load-bearing, not tidiness: clearing
the last annotation deletes the sidecar, and if the legacy file survived, the
read fallback would resurrect what the user just deleted.

`ledger._HEADER_KINDS` accepts the old log header. Triage decision logs are the
only record of what a user classified — nothing can reconstruct them.

Drop all of this a release or two out, once nobody has VidTriage data left.
Delete the shims and their tests together.

---

## Invariants — break these and something silently corrupts

### `core/` must not import Qt
Enforced by `tests/test_core_infra.py::test_core_never_imports_qt`. The data
model has to be constructible and testable without a `QApplication`.

### Never identify media by list position
This has caused two real bugs. `current_item` indexed `_order` with the
library's index; the library is shared, so anything that replaced the playlist
made a keystroke file a decision against a *different* file than the one on
screen. The file panel had the same shape of bug latent in `file_selected(which,
row)` the moment filtering existed.

**Resolve by path or by object identity, never by row.** If the lookup fails,
the right answer is `None` and a disabled command — not a plausible neighbour.

### Triage never moves the user's files
Classification appends one line to a JSONL log (`triage/ledger.py`). The log is
the database; the folder layout is a build artifact produced on demand by
`triage/snapshot.py`. Rules that follow:

- The log is **append-only**. Undo writes a correcting record; it never rewrites.
- Replay order is the whole contract: logs apply in the order given, records in
  file order, last write wins. That single rule gives both within-log history
  and cross-log overlay.
- `discover_logs` therefore may not sort on the raw filename. Two runs starting
  in one second produce `…Z` and `…Z-1`, and `-` sorts before `Z`, so plain name
  order replays the newer pass first and lets the older one win. It parses the
  stamp and counter out of the name instead — which also survives the log
  directory being copied, where sorting on mtime would not.
- `import_legacy_output` is gated on the corpus's **log directory**, never on
  the replay stack. Emptying the stack in the setup dialog means "ignore those
  passes"; reading it as "never triaged" rescans the output directory — often a
  snapshot this tool wrote — and invents a full set of classifications.
- A snapshot **refuses a non-empty target**. Writing into one leaves copies from
  a previous run in class folders they no longer belong to, and nothing can
  detect that afterwards.
- `--snapshot` uses `NullLedger` so building the deliverable does not itself
  become an overlay layer.

### An image is a one-frame source, not a special case
`ImageFileSource` reports `frame_count=1`, `fps=0`, and emits
`FrameRef(source_id, 0)`. **Frame index 0, never a `-1` sentinel** — a sentinel
would need handling in every bounds check, the slider, the exporters and the
sidecar schema, whereas index 0 means playback, annotation, inference,
classification, logging and export all work unchanged. Sessions may mix videos
and images freely; there is no code that distinguishes them, which is why there
is nothing to configure.

### `rich` is optional at every use site
Declared in `dependencies` so the default install gets readable tracebacks, but
`core/console.py` returns `None` when the import fails and every caller has a
plain-text path. `RichHandler` runs with `markup=False`: log messages
interpolate arbitrary paths, and a filename containing `[` would otherwise be
eaten as a style tag. Table cells are `rich.text.Text` for the same reason.

### An export names files, and those names are load-bearing
Two things here look cosmetic and are not:

* `ExportRequest.stems()` disambiguates duplicates. `ExportItem.stem` is
  `<filename>_<frame>`, so `a/clip.mp4` and `b/clip.mp4` want one label file and
  one extracted frame between them. Never write a file named from `item.stem`
  directly.
* `YoloExporter` **preserves the class ids already in `classes.txt`**. A label
  file stores an id; only that file says what the id means. Rewriting the list
  from one export's labels renumbers every file already in the directory —
  export cats, then dogs, and the cats come back labelled dog with nothing
  anywhere to indicate it.

Export scope defaults to the whole playlist, reading each file's sidecar; the
open file comes from the live store so unsaved edits are included.

### Menus, shortcuts and help are generated
From the command registry. Registering a `Command` with a `shortcut` is the
entire cost of adding a keybinding *and* documenting it — `Help ▸ Keyboard
Shortcuts` is built from the same registry and cannot go stale. Never hand-write
a shortcut table.

---

## Tests

```bash
pytest                    # default suite; no weights, no network
pytest -m models          # additionally exercise real YOLO / SAM backends
ruff check label_kit tests
```

### `HOME` is redirected at conftest *import* time, not in a fixture
`CONFIG_DIR` is computed from `Path.home()` once, at module scope. Test modules
import it during collection, which is before any fixture body runs — so a
session-scoped fixture setting `HOME` is too late, and the suite writes to the
developer's real `~/.labelkit`. It did exactly that for a while, filling the
real session list with pytest temp directories; the app then restored one of
them on the next real launch. `tests/test_isolation.py` guards this. If it
fails, every other test is touching your home directory.

Creating it that early also means no fixture teardown owns it, so conftest
cleans up in both directions: `atexit` for a run that ends, and a pid-stamped
sweep at import for one that did not. `atexit` does not survive SIGKILL, the OOM
killer or an IDE stop button, and a home leaked that way is invisible to
everything afterwards — eighteen had accumulated in `/tmp` before anyone looked.
The sweep must never get it backwards: an unstamped directory is spared for
`_STARTUP_GRACE_SECONDS`, because deleting a live run's `HOME` mid-test costs
far more than leaving an empty directory around.

### Qt tests run offscreen
Docks never report `isVisible()` in that mode. Assert on the persisted flag and
on `focusWidget()` instead of on real visibility or focus.

### Model-backed tests are opt-in
`addopts = "-m 'not models'"` in `pyproject.toml`. SAM additionally needs a
checkpoint the project cannot ship; it reports itself unavailable with a `curl`
command until one is downloaded to `~/.labelkit/weights/`.
