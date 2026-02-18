# 06 — Pixel Canvas

> **Example file:** `examples/06_pixel_canvas.py`

So far every tensor we've displayed was *read-only* from the viewer's
perspective — the GPU computes something, Vultorch shows it, end of story.
But what if the user wants to paint *into* the tensor? With the mouse?
In real time?

That's what this chapter is about: turning a zero-copy GPU tensor into an
interactive drawing surface. Left-click to paint, right-click to erase,
pick your color, resize your brush — done.

The magic is that there is no magic.
`create_tensor` gives you a normal `torch.Tensor` on CUDA.
You write pixels into it with standard indexing (`tensor[y, x, :3] = color`).
Because it's zero-copy, the display updates without any upload call.

## New friends

| New thing | What it does | Why it matters |
|-----------|-------------|----------------|
| `ui.get_mouse_pos()` | Returns `(x, y)` of the mouse cursor in screen pixels | You need this to know *where* on the canvas the user is pointing |
| `ui.is_item_hovered()` | `True` if the mouse is over the last-drawn widget (the canvas image) | So you only paint when the cursor is on the canvas, not the controls |
| `ui.is_mouse_clicked(0)` | `True` on the frame the left button is pressed | Detect a single click |
| `ui.is_mouse_dragging(0)` | `True` while the left button is held down and the mouse moves | Continuous painting — fires every frame while you drag |
| Screen → pixel mapping | Convert screen coordinates to tensor `[y, x]` index | The canvas image is stretched to fill the panel; you need to undo that stretch to find which pixel the mouse is over |

## What we're building

A 128×128 pixel canvas: left-click draws, right-click erases, with a sidebar
for brush size, colors, clear button, and a grid overlay toggle.

## Full code

```python
import torch
import vultorch
from vultorch import ui

if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"

H, W = 128, 128

view = vultorch.View("06 - Pixel Canvas", 900, 700)
ctrl = view.panel("Controls", side="left", width=0.24)
draw_panel = view.panel("Canvas")

# Zero-copy RGBA tensor
canvas_tensor = vultorch.create_tensor(H, W, channels=4, device=device,
                                        name="canvas", window=view.window)
canvas_tensor[:, :, :3] = 0.1
canvas_tensor[:, :, 3] = 1.0
canvas = draw_panel.canvas("canvas", filter="nearest", fit=True)
canvas.bind(canvas_tensor)

# Persistent backing store (so grid overlay doesn't accumulate)
backing = torch.zeros(H, W, 3, device=device)
backing[:] = 0.1

state = {
    "brush_size": 1,
    "brush_color": (1.0, 0.3, 0.1),
    "show_grid": False,
    "bg_color": (0.1, 0.1, 0.1),
}


def draw_brush(cy, cx, size, r, g, b):
    half = size // 2
    y0, y1 = max(0, cy - half), min(H, cy + half + 1)
    x0, x1 = max(0, cx - half), min(W, cx + half + 1)
    backing[y0:y1, x0:x1, 0] = r
    backing[y0:y1, x0:x1, 1] = g
    backing[y0:y1, x0:x1, 2] = b


def clear_canvas():
    r, g, b = state["bg_color"]
    backing[:, :, 0] = r
    backing[:, :, 1] = g
    backing[:, :, 2] = b


def refresh_display():
    canvas_tensor[:, :, :3] = backing
    if state["show_grid"]:
        for i in range(0, H, 8):
            canvas_tensor[i, :, :3] = canvas_tensor[i, :, :3].clamp(0, 0.85) + 0.15
        for j in range(0, W, 8):
            canvas_tensor[:, j, :3] = canvas_tensor[:, j, :3].clamp(0, 0.85) + 0.15


@ctrl.on_frame
def draw_controls():
    ctrl.text(f"Canvas: {W}×{H}")
    ctrl.text(f"FPS: {view.fps:.1f}")
    ctrl.separator()

    state["brush_size"] = ctrl.slider_int("Brush Size", 1, 16, default=1)
    state["brush_color"] = ctrl.color_picker("Brush Color",
                                              default=(1.0, 0.3, 0.1))
    state["bg_color"] = ctrl.color_picker("Background",
                                           default=(0.1, 0.1, 0.1))
    ctrl.separator()

    if ctrl.button("Clear", width=120):
        clear_canvas()

    state["show_grid"] = ctrl.checkbox("Show Grid (8px)", default=False)

    ctrl.separator()
    ctrl.text_wrapped(
        "Left-click to draw. Right-click to erase. "
        "Adjust brush size and color above."
    )


@draw_panel.on_frame
def handle_drawing():
    if not ui.is_item_hovered():
        refresh_display()
        return

    mx, my = ui.get_mouse_pos()

    # Map screen position to tensor pixel coordinates
    wp_x, wp_y = ui.get_window_pos()
    win_w, win_h = ui.get_window_size()
    content_x = wp_x + 8
    content_y = wp_y + 26
    content_w = win_w - 16
    content_h = win_h - 34

    u = max(0.0, min(1.0, (mx - content_x) / max(content_w, 1)))
    v = max(0.0, min(1.0, (my - content_y) / max(content_h, 1)))
    px = int(u * (W - 1))
    py = int(v * (H - 1))

    painting = ui.is_mouse_clicked(0) or ui.is_mouse_dragging(0, 0.0)
    erasing  = ui.is_mouse_clicked(1) or ui.is_mouse_dragging(1, 0.0)

    if painting:
        r, g, b = state["brush_color"]
        draw_brush(py, px, state["brush_size"], r, g, b)
    elif erasing:
        r, g, b = state["bg_color"]
        draw_brush(py, px, state["brush_size"], r, g, b)

    refresh_display()


view.run()
```

## What just happened?

### Screen coordinates → tensor pixels

This is the core trick of the example. The canvas image is stretched to
fill the panel, so a 128×128 tensor might be displayed at 600×500 screen
pixels. When the mouse is at screen position `(mx, my)`, you need to
figure out which tensor pixel that corresponds to:

```python
# Panel window position and size
wp_x, wp_y = ui.get_window_pos()
win_w, win_h = ui.get_window_size()

# Content area (subtract ImGui title bar + padding)
content_x = wp_x + 8
content_y = wp_y + 26

# Normalize to [0, 1]
u = (mx - content_x) / content_w
v = (my - content_y) / content_h

# Scale to pixel indices
px = int(u * (W - 1))
py = int(v * (H - 1))
```

This is the same mental model as converting normalized device coordinates
to pixel coordinates in a rasterizer — just in reverse.

### is_item_hovered + is_mouse_dragging

ImGui tracks which widget the mouse is over. After the canvas image is
drawn (which happens automatically inside the panel), `is_item_hovered()`
tells you whether the cursor is on the image.

`is_mouse_dragging(button, threshold)` returns `True` every frame while
the button is held and the mouse has moved at least `threshold` pixels.
With `threshold=0.0` it fires immediately — effectively "is button held."

Combined, this gives you continuous painting:

```python
if ui.is_item_hovered():
    if ui.is_mouse_dragging(0, 0.0):
        draw_brush(py, px, ...)
```

### Backing store + display refresh

We keep a separate `backing` tensor (RGB, no alpha) that stores the
actual pixel art. Each frame, we copy it to the display tensor and
optionally overlay a grid. This prevents the grid lines from
"baking into" the artwork over time.

```python
canvas_tensor[:, :, :3] = backing          # copy artwork
if state["show_grid"]:
    canvas_tensor[::8, :, :3] += 0.15      # lighten grid rows
    canvas_tensor[:, ::8, :3] += 0.15      # lighten grid columns
```

## Key takeaways

1. **`create_tensor` is a two-way street** — the GPU can write to it
   (simulation) and Python can write to it (user interaction). Vultorch
   displays whatever is in the tensor, no questions asked.

2. **Screen → tensor mapping** — `get_mouse_pos()` gives screen pixels;
   you subtract the panel origin, divide by the panel size, and multiply
   by tensor dimensions. Same math as UV coordinates in graphics.

3. **`is_item_hovered` + `is_mouse_dragging`** — the standard pattern
   for interactive widgets. Check hover first, then check button state.

4. **Backing store pattern** — if you overlay decorations (grids, markers,
   crosshairs) on top of user data, keep the raw data in a separate tensor
   and composite every frame. Otherwise decorations accumulate.

5. **`filter="nearest"`** — essential for pixel art. Without it the
   128×128 grid would look like a blurry watercolor instead of crisp squares.

!!! tip
    This screen-to-pixel mapping technique is the same one you'd use to
    build annotation tools — segmentation masks, bounding boxes, keypoint
    labeling. The tensor *is* the label map.
