# 14 — Screen Recorder

> **Example file:** `examples/14_screen_recorder.py`

Your neural rendering result looks amazing on screen — but how do you
show it in your paper, poster, or tweet?  Screenshots only capture a
single frame.  What you really need is an animation.

This chapter introduces Vultorch's built-in recording API.  Any canvas
can record its contents to an animated **GIF** (via Pillow) — no
screen-capture software required.

## What we're building

A psychedelic kaleidoscope animation with a control sidebar:

| Area | Content |
|------|---------|
| Left | **Controls** — Record/Stop toggle, quality slider, speed slider |
| Right | **Animation** — GPU-generated kaleidoscope pattern |

Press the Record button.  Interact with the speed slider while recording.
Press Stop — the GIF appears in your working directory.

## New friends

| New thing | What it does | Why it matters |
|-----------|-------------|----------------|
| `canvas.start_recording(path)` | Begin capturing frames to a `.gif` | Requires only Pillow |
| `canvas.stop_recording()` | Finalize and write the output file | Returns the absolute path of the saved file |
| `canvas.is_recording` | Check whether recording is active | Show a red "● REC" indicator |
| `panel.record_button(canvas, path)` | One-line toggle widget | Combines a styled button with start/stop logic |
| `quality` (0–1) | Controls colours per frame | Lower quality → fewer colours → smaller file |

## How recording works

Under the hood, every frame while recording is active:

1. The bound tensor is moved to CPU (if on CUDA)
2. Converted to uint8 RGB
3. Colour-quantized according to `quality` (2–256 colours)
4. Appended to a list of PIL Images, written to GIF on stop

This means recording does add some CPU overhead (one GPU→CPU copy per
frame).  For a 256×256 canvas at 30 fps this is negligible.  For
4K tensors you may want to record at a lower resolution.

## Quality

The `quality` parameter goes from 0 to 1:

| quality | Colours | Use case |
|---------|---------|----------|
| 1.0 | 256 | Full quality, largest file |
| 0.8 | ~205 | Good default |
| 0.5 | ~129 | Compact, slight banding |
| 0.1 | ~27 | Very small, visible dithering |

```python
canvas.start_recording("out.gif", fps=15, quality=0.5)
```

## Dependency

GIF recording requires Pillow:

```bash
pip install Pillow
```

## The record button

The easiest way to add recording to any demo is `panel.record_button`:

```python
@ctrl.on_frame
def draw():
    ctrl.record_button(my_canvas, "output.gif", fps=30, quality=0.8)
```

That's it — one line.  It shows "Record" when idle (normal button) and
"Stop Recording" when active (red button).

## Programmatic recording

For headless or scripted use, call the methods directly:

```python
canvas.start_recording("sweep.gif", fps=15, quality=0.6)
for lr in learning_rates:
    train(lr)
    canvas.bind(result)
    view.step()
    view.end_step()
canvas.stop_recording()
```

This works in both windowed and headless mode — great for generating
paper figures or supplementary animations from a parameter sweep.

## Full code

```python
import math
import torch
import vultorch

if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"
H, W = 256, 256

# Pre-compute coordinate grids
ys = torch.linspace(-1.0, 1.0, H, device=device)
xs = torch.linspace(-1.0, 1.0, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")
rr = (xx ** 2 + yy ** 2).sqrt()
angle = torch.atan2(yy, xx)

# View + panels
view = vultorch.View("14 - Screen Recorder", 1024, 700)
ctrl = view.panel("Controls", side="left", width=0.30)
anim_panel = view.panel("Animation")

# Shared display tensor
display = vultorch.create_tensor(H, W, channels=4, device=device,
                                 name="anim", window=view.window)
display[:, :, 3] = 1.0
canvas = anim_panel.canvas("anim")
canvas.bind(display)

state = {"speed": 2.0, "quality": 0.8}


@view.on_frame
def animate():
    t = view.time * state["speed"]
    k = 6
    r = rr * 4.0 + t
    a = (angle * k).remainder(math.pi * 2) + t * 0.5
    display[:, :, 0] = (r.sin() * a.cos() * 0.5 + 0.5).clamp(0, 1)
    display[:, :, 1] = ((r + 2.094).sin() * (a + 1.047).cos() * 0.5 + 0.5).clamp(0, 1)
    display[:, :, 2] = ((r + 4.189).sin() * (a + 2.094).cos() * 0.5 + 0.5).clamp(0, 1)


@ctrl.on_frame
def draw_ctrl():
    ctrl.text(f"FPS: {view.fps:.0f}")
    ctrl.separator()
    state["quality"] = ctrl.slider("Quality", 0.0, 1.0, default=0.8)
    state["speed"] = ctrl.slider("Speed", 0.5, 10.0, default=2.0)
    ctrl.separator()

    ctrl.record_button(canvas, "recording.gif", fps=30,
                       quality=state["quality"])

    ctrl.separator()
    if canvas.is_recording:
        ctrl.text_colored(1, 0.3, 0.3, 1, "● RECORDING")
    else:
        ctrl.text("● Idle")


view.run()
```

## Try it

```bash
python examples/14_screen_recorder.py
```

Adjust the quality and speed sliders, hit Record, wait a few seconds,
hit Stop.  Check your working directory for `recording.gif`.


