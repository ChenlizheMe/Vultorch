"""
06 - Pixel Canvas
=================
Draw on a GPU tensor with the mouse — no CPU round-trip.

Key concepts
------------
- create_tensor      : Zero-copy shared GPU tensor
- ui.get_mouse_pos   : Read mouse cursor in screen coordinates
- ui.is_item_hovered : Check if the canvas is under the cursor
- ui.is_mouse_clicked: Detect mouse button press
- ui.is_mouse_dragging: Continuous paint while button held
- filter="nearest"   : Pixel-perfect display for grid-like data
- Panel ↔ tensor     : translate screen coords → tensor pixel coords

Layout
------
Left  : Controls — brush size, brush color, clear button, grid toggle
Right : Canvas — the drawing surface
"""

import torch
import vultorch
from vultorch import ui

if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"

# ── Canvas parameters ─────────────────────────────────────────────
H, W = 128, 128

# ── View + panels ─────────────────────────────────────────────────
view = vultorch.View("06 - Pixel Canvas", 900, 700)
ctrl = view.panel("Controls", side="left", width=0.24)
draw_panel = view.panel("Canvas")

# Zero-copy RGBA tensor — writes are instantly visible to the GPU
canvas_tensor = vultorch.create_tensor(H, W, channels=4, device=device,
                                        name="canvas", window=view.window)
canvas_tensor[:, :, :3] = 0.1          # dark grey background
canvas_tensor[:, :, 3] = 1.0           # fully opaque
canvas = draw_panel.canvas("canvas", filter="nearest", fit=True)
canvas.bind(canvas_tensor)

# Keep a persistent copy so the grid overlay doesn't accumulate
backing = torch.zeros(H, W, 3, device=device)
backing[:] = 0.1

# ── State ─────────────────────────────────────────────────────────
state = {
    "brush_size": 1,
    "brush_color": (1.0, 0.3, 0.1),
    "show_grid": False,
    "bg_color": (0.1, 0.1, 0.1),
}


def draw_brush(cy: int, cx: int, size: int, r: float, g: float, b: float):
    """Paint a square brush centered at (cy, cx)."""
    half = size // 2
    y0 = max(0, cy - half)
    y1 = min(H, cy + half + 1)
    x0 = max(0, cx - half)
    x1 = min(W, cx + half + 1)
    backing[y0:y1, x0:x1, 0] = r
    backing[y0:y1, x0:x1, 1] = g
    backing[y0:y1, x0:x1, 2] = b


def clear_canvas():
    """Fill the canvas with the background color."""
    r, g, b = state["bg_color"]
    backing[:, :, 0] = r
    backing[:, :, 1] = g
    backing[:, :, 2] = b


def refresh_display():
    """Copy backing store to RGBA display, optionally add grid overlay."""
    canvas_tensor[:, :, :3] = backing
    if state["show_grid"]:
        for i in range(0, H, 8):
            canvas_tensor[i, :, :3] = canvas_tensor[i, :, :3].clamp(0, 0.85) + 0.15
        for j in range(0, W, 8):
            canvas_tensor[:, j, :3] = canvas_tensor[:, j, :3].clamp(0, 0.85) + 0.15


# ── Controls ──────────────────────────────────────────────────────
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
        "Left-click on the canvas to draw. "
        "Right-click to erase (paints background color). "
        "Adjust brush size and color with the controls above."
    )


# ── Drawing interaction ──────────────────────────────────────────
@draw_panel.on_frame
def handle_drawing():
    """Run after the canvas image is drawn — check mouse state."""
    if not ui.is_item_hovered():
        refresh_display()
        return

    # Mouse position in screen space
    mx, my = ui.get_mouse_pos()

    # Image rect: the fit canvas fills the panel content area.
    # Window position + small offsets for ImGui title bar / padding.
    wp_x, wp_y = ui.get_window_pos()
    win_w, win_h = ui.get_window_size()
    content_x = wp_x + 8     # default window padding
    content_y = wp_y + 26    # title bar height
    content_w = win_w - 16
    content_h = win_h - 34

    # Normalize to [0, 1] within the image
    u = (mx - content_x) / max(content_w, 1)
    v = (my - content_y) / max(content_h, 1)
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))

    # Convert to pixel coordinates
    px = int(u * (W - 1))
    py = int(v * (H - 1))

    # Left-click / drag → draw; right-click / drag → erase
    painting = ui.is_mouse_clicked(0) or ui.is_mouse_dragging(0, 0.0)
    erasing = ui.is_mouse_clicked(1) or ui.is_mouse_dragging(1, 0.0)

    if painting:
        r, g, b = state["brush_color"]
        draw_brush(py, px, state["brush_size"], r, g, b)
    elif erasing:
        r, g, b = state["bg_color"]
        draw_brush(py, px, state["brush_size"], r, g, b)

    refresh_display()


view.run()
