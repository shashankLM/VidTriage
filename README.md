# VidTriage

Triage videos into folders, annotate individual frames, and point a model at a
region to have it do the annotating for you.

Built on PySide6 and OpenCV. Model backends are optional — the app runs fine
without them and tells you what to install if you want them.

---

## Quick start

```bash
conda activate py311
pip install -r requirements.txt

python run.py                      # or: python -m vidtriage
python run.py /path/to/videos      # open a folder straight away
python run.py -i in/ -o out/       # pre-fill a triage session
```

Optional model backends:

```bash
pip install -r requirements-models.txt

# YOLO weights download themselves on first use.
# SAM needs a checkpoint:
mkdir -p ~/.vidtriage/weights && cd ~/.vidtriage/weights
curl -LO https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

## The three things it does

### 1. Triage — file whole videos into folders

Press a number key; the video moves into that class's folder and the next one
loads. `U` undoes, `X` files to `_errors/`, `S` skips.

```
  Input directory              Output directory
  ┌──────────────┐             ┌──────────────────────┐
  │ video_01.mp4 │  ──[1]──>   │ cat/video_01.mp4     │
  │ video_02.mp4 │  ──[2]──>   │ dog/video_02.mp4     │
  │ video_03.mp4 │  ──[x]──>   │ _errors/video_03.mp4 │
  └──────────────┘             └──────────────────────┘
```

Files are **moved**, not copied. Relaunch and previously classified videos are
picked back up from the output folders. If a move would overwrite an existing
file it is refused and reported — nothing is ever silently replaced.

### 2. Annotate — draw on frames

Pick a tool (`B` box, `P` point, `G` polygon), draw, and it is saved against
that exact frame. Select with `V`, drag to move, grab a handle to resize,
`Del` to remove, `Ctrl+Z` to undo.

Annotations are stored in a `<video>.vidtriage.json` sidecar next to the media,
written atomically. The sidecar travels with the video when triage moves it.

### 3. Model-assisted — point at a thing, get an annotation

Choose a model under **Annotate ▸ Prompt With**, then:

| Gesture | What happens |
|---|---|
| Drag a box | The model runs inside that region |
| Click a point | SAM-family models segment the object under the cursor |
| Right-click | Adds a *negative* point — "not this" |
| Shift-click | Accumulates points into one prompt, to refine a mask |
| `Ctrl+R` | Re-runs the last prompt |

Results arrive as normal annotations, drawn dashed to mark them as predictions.
**Annotate ▸ Accept Model Predictions** (`Ctrl+Shift+P`) confirms them, keeping
the originating model id in the annotation's attributes.

Inference runs on a worker thread, so the UI never blocks. A newer request
supersedes an older one, so three quick clicks give you the third answer rather
than three stale ones.

---

## Extending it

Everything the user can do is a plugin contribution — including the built-in
triage workflow. A plugin is one class:

```python
from vidtriage.plugins.api import Plugin

class MyPlugin(Plugin):
    id = "myplugin"
    name = "My Feature"

    def activate(self, ctx):
        ctx.add_layer(MyOverlay())          # draws over the frame
        ctx.add_tool(MyTool())              # a new mouse gesture
        ctx.add_model(MyModel())            # a new inference backend
        ctx.add_panel(id="myplugin.panel", title="Mine", factory=MyPanel)
        ctx.add_exporter(MyExporter())
        ctx.add_command(id="myplugin.go", title="Go", shortcut="Ctrl+Shift+G",
                        menu="Tools", handler=self.go)

PLUGIN = MyPlugin
```

Drop that in `~/.vidtriage/plugins/` and it loads on next launch. Menus,
keyboard shortcuts and the overlay/tool/model lists are all *generated from the
registries*, so there is no menu file to edit and no key-handling chain to add
a branch to. Disabling the plugin removes everything it contributed.

Three discovery sources, all equal: built-ins, anything advertising the
`vidtriage.plugins` entry-point group (so `pip install vidtriage-sam3` is
enough), and drop-ins in `~/.vidtriage/plugins/`.

**View ▸ Plugins** shows what loaded, what did not, and why. Launched from a
terminal, startup prints the same thing as two tables — plugins, then models
with their prompt capabilities and, for anything that cannot run, the command
that fixes it.

### Adding a model

```python
from vidtriage.plugins.models import Availability, Capability, InferenceModel, ParamSpec

class Sam3Model(InferenceModel):
    id = "sam3.predict"
    display_name = "SAM 3"
    capabilities = Capability.POINT_PROMPT | Capability.BOX_PROMPT
    parameters = (ParamSpec("threshold", "Threshold", "float", 0.5, 0.0, 1.0),)

    def availability(self):                 # cheap — no heavy imports here
        return Availability.missing_package("sam3") if ... else Availability.available()

    def load(self):                         # slow; runs off the GUI thread
        ...

    def infer(self, request):               # returns annotations in image pixels
        return [request.annotation(mask, label="thing", score=0.9, source=self.id)]
```

`capabilities` is what makes it work everywhere without further wiring: the
model appears under **Prompt With** for the gestures it accepts, and gets a
**Run** entry only if it declares `WHOLE_FRAME`. If `availability()` says no,
it shows disabled with your remedy text instead of failing at click time.

---

## Architecture

```
vidtriage/
  core/         Qt-free data model: geometry, frames, annotations,
                events, registries, commands
  media/        MediaSource, threaded decoder, playback clock, controller
  view/         ImageCanvas, overlay layers, annotation items, tools, theme
  persistence/  Sidecars, COCO/YOLO/CSV exporters, settings
  plugins/      Plugin contract, inference API, threaded runner
    builtin/    triage · annotate · yolo · sam · guides
  app/          Context, thin window, generated menus, transport bar
```

Each layer may import the ones above it, never below. `core` has no Qt import at
all — enforced by a test — so the data model is constructible without a
`QApplication`.

Key invariants, each backed by tests:

- **Frames are never mutated.** Overlays paint on top. What a model receives is
  the true frame, not one with a counter burned into the corner.
- **Coordinates round-trip exactly.** `widget_to_image` is the exact inverse of
  `image_to_widget` at any zoom, which is what lets a click become a prompt.
- **Decoding is off the GUI thread**, with request coalescing and backpressure.
- **Inference is off the GUI thread**, serialised per model, newest-wins.
- **Writes are atomic.** A crash mid-save cannot truncate your annotations.

---

## Keyboard

**Help ▸ Keyboard Shortcuts** (`F1`) is generated from the command registry, so
it is always current. The main ones:

| Key | Action |
|---|---|
| `1`–`9` | Classify with that class (triage) |
| `U` / `X` / `S` | Undo classification · move to `_errors` · skip |
| `Space` · `←` `→` · `↑` `↓` | Play/pause · step frame · previous/next file |
| `V` `H` `B` `P` `G` | Select · pan · box · point · polygon |
| `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / redo an annotation edit |
| `Ctrl+L` | Set the working label |
| `Ctrl+R` | Re-run the last model prompt |
| `Ctrl+E` / `Ctrl+S` | Export · save annotations now |
| `Ctrl+=` `Ctrl+-` `Ctrl+0` `Ctrl+1` | Zoom in · out · fit · actual size |
| `Tab` · `E` | Switch pending/classified · toggle the file panel |
| `Ctrl+G` | Ratio guides |
| `F11` · `F1` · `Ctrl+Q` | Fullscreen · help · quit |

Mouse: wheel zooms at the cursor, middle-drag pans, right-click opens a context
menu.

> **Changed from v1:** `Ctrl+Z` now undoes an *annotation* edit. Undo of a
> *classification* moved to `U`. There are two independent histories now, and
> annotation edits are by far the more frequent.

---

## Overlays

Non-destructive, toggled under **View ▸ Overlays**:

| Overlay | Purpose |
|---|---|
| Frame Counter | Frame index and timestamp, as a HUD |
| Crosshair | Cursor crosshair with pixel coordinates and RGB readout |
| Ratio Guides | Horizontal/vertical lines at fractions of the frame (`Ctrl+G`) |

Ratio guides replace the habit of re-encoding a directory of clips with
`cv2.line` just to see where `h/2` falls. Set them to any fractions you like via
**View ▸ Overlays ▸ Set Horizontal Guides…**; they are adjustable while the video
plays and never touch the pixels.

---

## Export

**File ▸ Export Annotations…** (`Ctrl+E`)

| Format | Notes |
|---|---|
| COCO JSON | Boxes, polygon segmentation, RLE for masks |
| YOLO labels | `labels/*.txt` plus `classes.txt`; non-box shapes reduce to their bounding box, and that is reported |
| CSV | One row per annotation |

Tick **extract the annotated frames** to write the referenced images alongside,
producing a directory a training run can consume directly.

Triage classifications export separately via **File ▸ Export Classifications…**.

---

## Development

```bash
pip install -r requirements.txt
pip install pytest ruff

pytest                  # default suite, no model weights needed
pytest -m models        # additionally exercise real YOLO / SAM backends
ruff check vidtriage tests
```

`python run.py --no-plugins` starts the bare shell, and `--safe-mode` skips
drop-in plugins — both useful for isolating a misbehaving extension.

Plugin, decode and layer-paint faults are contained rather than fatal, so the
report that replaces the crash is the only thing you get: run with `-v` to add
local variables to those tracebacks. Console output uses `rich` when it is
installed and falls back to plain text when it is not; the rotating log in the
session output directory is always plain.

Config lives in `~/.vidtriage/`: `settings.json`, `sessions.json`,
`plugins.json`, `weights/`, `plugins/`.
