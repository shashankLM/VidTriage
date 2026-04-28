# VidTriage

Rapidly classify videos into user-defined categories using keyboard shortcuts.

Built with PySide6 and OpenCV — no native dependencies beyond conda/pip.

---

## Quick Start

```bash
# Prerequisites: conda environment with Python 3.11
conda activate py311

# Install dependencies
pip install -r requirements.txt

# Launch
python run.py

# Or with pre-filled directories
python run.py -i /path/to/videos -o /path/to/output
```

## How It Works

```
  Input directory              Output directory
  ┌──────────────┐             ┌──────────────────────┐
  │ video_01.mp4 │  ──[1]──>   │ cat/video_01.mp4     │
  │ video_02.mp4 │  ──[2]──>   │ dog/video_02.mp4     │
  │ video_03.mp4 │  ──[x]──>   │ _errors/video_03.mp4 │
  │ video_04.mp4 │             │                      │
  └──────────────┘             └──────────────────────┘
```

1. **Setup** — Pick input/output directories and define class names
2. **Classify** — Watch each video and press a number key to classify
3. **Files move** — Videos are moved (not copied) into class subdirectories
4. **Resume** — Relaunch anytime; previously classified files are detected from output folders

## Setup Dialog

On launch, a setup dialog lets you configure:

| Field | Description |
|---|---|
| **Session** | Dropdown of saved sessions (defaults to most recent), or "+ New Session" |
| **Input directory** | Folder containing videos to classify (must be readable) |
| **Output directory** | Where classified videos are moved (must be writable) |
| **Classes** | One name per line — keys auto-assigned `1`-`9` |

The session dropdown remembers all previous input/output/class configurations. Selecting a session populates all fields. Sessions are matched by output directory — launching with the same output dir updates the existing session entry.

Classes are shown in a table view with their key assignments. Click the table to edit as text.

Example class input:
```
cat
dog
bird
skip
```
Maps to: `[1] cat` `[2] dog` `[3] bird` `[4] skip`

> **Directory rules:** Input and output must be separate, non-overlapping directories.
> Neither can be inside the other. Both must exist and be accessible.
> Sessions are persisted in `~/.vidtriage/config.json`.

## Menu Bar

| Menu | Items |
|---|---|
| **File** | Reopen Setup, Export Annotations, Quit |
| **Edit** | Undo, Change Classes, Skip |
| **View** | Frame Number Overlay, File Explorer, Fullscreen, Summary |
| **Playback** | Speed, Frame Step, End Behavior |
| **Help** | Keyboard Shortcuts |

### Playback Options

| Setting | Values | Default |
|---|---|---|
| **Speed** | 0.25x, 0.5x, 0.75x, 1x, 1.25x, 1.5x, 1.75x, 2x | 1x |
| **Frame Step** | 1, 2, 5, 10 frames per arrow key press | 1 |
| **End Behavior** | Next Video (auto-advance), Loop, Stop | Next Video |

## Keybindings

| Key | Action |
|---|---|
| `1` – `9` | Classify with mapped class and auto-advance |
| `Space` | Play / pause |
| `→` | Step forward (configurable frame count) |
| `←` | Step backward (configurable frame count) |
| `↓` | Next file |
| `↑` | Previous file |
| `Tab` | Toggle focus between pending and classified lists |
| `s` | Skip to next pending video |
| `e` | Toggle file explorer panel |
| `x` | Move current file to `_errors/` |
| `h` / `?` | Open help |
| `F11` | Toggle fullscreen |
| `Ctrl+Z` | Undo last classification |
| `Ctrl+Q` | Quit |

## File Explorer

The left pane has two lists in a vertical split:

- **Pending** — unclassified videos from the input directory
- **Classified** — videos that have been classified (shows `[class] filename`)

The active list has a blue border. Use `Tab` to switch between them.
Selecting a classified video lets you reclassify it — press a number key to move it to a different class.
Toggle the panel with `e` or via **View → File Explorer**.

## Logs

An activity log is written to the output directory:

| File | Purpose |
|---|---|
| `vidtriage_activity.log` | Human-readable activity log with timestamps |

## Export

Use **File → Export Annotations** to save a CSV of all videos and their classifications.

| Column | Description |
|---|---|
| `video` | Original filename |
| `class` | Assigned class name, or `unclassified` for pending/error videos |
| `path` | Relative path within the output directory |

Defaults to `<outdir>/annotations.csv`. A save dialog lets you pick a different location.

## Supported Formats

`.mp4` `.mkv` `.avi` `.mov` `.wmv` `.flv` `.webm` `.m4v` `.mpg` `.mpeg` `.3gp` `.ts` `.mts`

## Project Structure

```
run.py                   Entry point
vidtriage/
  __main__.py            CLI args + app bootstrap
  wizard.py              Setup dialog (dirs + classes)
  main_window.py         Main GUI, menubar, keybindings, classification logic
  player.py              OpenCV video player with speed/step/overlay controls
  file_explorer.py       Two-list file explorer with status colors
  session.py             Session state — single source of truth for all data
  io_ops.py              File move/undo operations
  config.py              Config persistence + class parsing
  models.py              Dataclasses (VideoItem, ClassEntry, AppConfig)
```
