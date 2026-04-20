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
  └──────────────┘             │ vidtriage_log.csv    │
                               └──────────────────────┘
```

1. **Setup** — Pick input/output directories and define class names
2. **Classify** — Watch each video and press a number key to classify
3. **Files move** — Videos are moved (not copied) into class subdirectories
4. **Resume** — Relaunch anytime; previously classified files are restored from the log

## Setup Dialog

On launch, a setup dialog lets you configure:

| Field | Description |
|---|---|
| **Input directory** | Folder containing videos to classify (must be readable) |
| **Output directory** | Where classified videos are moved (must be writable) |
| **Classes** | One name per line — keys auto-assigned `1`-`9` |

Classes are shown in a table view with their key assignments. Click the table to edit as text.

Example class input:
```
cat
dog
bird
skip
```
Maps to: `[1] cat` `[2] dog` `[3] bird` `[4] skip` `[5] 5` ... `[9] 9`

> **Directory rules:** Input and output must be separate, non-overlapping directories.
> Neither can be inside the other. Both must exist and be accessible.
> Last-used values are remembered across sessions (`~/.vidtriage/config.json`).

## Keybindings

| Key | Action |
|---|---|
| `1` – `9` | Classify with mapped class and auto-advance |
| `Space` | Play / pause |
| `Right` | Next frame |
| `Left` | Previous frame |
| `Down` | Next file |
| `Up` | Previous file |
| `Tab` | Toggle focus between pending and classified lists |
| `x` | Move current file to `_errors/` |
| `h` / `?` | Open help overlay |
| `Ctrl+Z` | Undo last classification |
| `Ctrl+Q` | Quit |

## File Explorer

The left pane has two lists in a vertical split:

- **Pending** — unclassified videos from the input directory
- **Classified** — videos that have been classified (shows `[class] filename`)

The active list has a blue border. Use `Tab` to switch between them.
Selecting a classified video lets you reclassify it — press a number key to move it to a different class.

## Logs

Two log files are written to the output directory:

| File | Purpose |
|---|---|
| `vidtriage_log.csv` | Structured action log — used for resume and auditing |
| `vidtriage_activity.log` | Human-readable activity log with timestamps |

**CSV columns:** `timestamp`, `source_path`, `destination_path`, `class_key`, `class_name`, `action`

**Actions:** `classify`, `undo`, `error`, `skip`

## Export

Click **Export** in the toolbar to save a CSV of all videos and their classifications.

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
  main_window.py         Main GUI, keybindings, classification logic
  player.py              OpenCV video player (frame-by-frame via QTimer)
  file_explorer.py       Two-list file explorer with status colors
  io_ops.py              File move/undo/log operations
  config.py              Config persistence + class parsing
  models.py              Dataclasses (VideoItem, ClassEntry, AppConfig)
```
