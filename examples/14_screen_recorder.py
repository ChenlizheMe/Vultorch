"""
14 - Screen Recorder
====================
Record any canvas to GIF with a single button press.

This demo shows a psychedelic kaleidoscope animation and provides
Record / Stop buttons to capture the output.

Layout:
- Left panel : Controls (record button, quality slider, speed slider)
- Right panel: Animated output

Key concepts
------------
- canvas.start_recording   : Begin recording frames
- canvas.stop_recording    : Finalize and save the file
- panel.record_button      : One-line toggle widget
- quality (0~1)            : Controls GIF colour count (smaller = smaller file)
- Recording works in both windowed and headless mode
"""

import math

import torch
import vultorch

if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"
H, W = 256, 256

# ── Pre-compute coordinate grids ───────────────────────────────────
ys = torch.linspace(-1.0, 1.0, H, device=device)
xs = torch.linspace(-1.0, 1.0, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")
rr = (xx ** 2 + yy ** 2).sqrt()
angle = torch.atan2(yy, xx)

# ── View + panels ──────────────────────────────────────────────────
view = vultorch.View("14 - Screen Recorder", 1024, 700)
ctrl = view.panel("Controls", side="left", width=0.30)
anim_panel = view.panel("Animation")

# Shared display tensor
display = vultorch.create_tensor(H, W, channels=4, device=device,
                                 name="anim", window=view.window)
display[:, :, 3] = 1.0
canvas = anim_panel.canvas("anim")
canvas.bind(display)

state = {
    "speed": 2.0,
    "quality": 0.8,
}


@view.on_frame
def animate():
    """Generate a trippy kaleidoscope animation on the GPU."""
    t = view.time * state["speed"]
    k = 6  # kaleidoscope fold count

    r = rr * 4.0 + t
    a = (angle * k).remainder(math.pi * 2) + t * 0.5

    display[:, :, 0] = (r.sin() * a.cos() * 0.5 + 0.5).clamp(0, 1)
    display[:, :, 1] = ((r + 2.094).sin() * (a + 1.047).cos() * 0.5 + 0.5).clamp(0, 1)
    display[:, :, 2] = ((r + 4.189).sin() * (a + 2.094).cos() * 0.5 + 0.5).clamp(0, 1)


@ctrl.on_frame
def draw_ctrl():
    ctrl.text(f"FPS: {view.fps:.0f}")
    ctrl.separator()

    # ── Quality & speed controls ──
    state["quality"] = ctrl.slider("Quality", 0.0, 1.0, default=0.8)
    state["speed"] = ctrl.slider("Speed", 0.5, 10.0, default=2.0)

    ctrl.separator()

    # ── Record / Stop toggle button ──
    ctrl.record_button(canvas, "recording.gif", fps=30,
                       quality=state["quality"])

    ctrl.separator()

    # ── Status ──
    if canvas.is_recording:
        ctrl.text_colored(1, 0.3, 0.3, 1, "● RECORDING")
    else:
        ctrl.text("● Idle")

    ctrl.separator()
    ctrl.text_wrapped(
        "Press Record to capture the animation to a GIF. "
        "Adjust Quality to control file size (lower = smaller). "
        "Requires Pillow (pip install Pillow)."
    )


view.run()
