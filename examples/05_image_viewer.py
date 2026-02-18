"""
05 - Image Viewer
=================
Load, transform, and save images — all without PIL or OpenCV.

Key concepts
------------
- vultorch.imread()  : Load an image file as a torch.Tensor (uses stb_image)
- vultorch.imwrite() : Save a tensor to an image file (PNG/JPEG/BMP)
- Canvas.save()      : Save a canvas's bound tensor to disk
- panel.combo()      : Drop-down menu for selecting options
- panel.input_text() : Text input field
- filter switch      : Compare "nearest" vs "linear" sampling

Layout
------
Left  : Controls panel — transform selector, brightness/contrast sliders,
        filter toggle, save path input, save button
Right : Two stacked canvases — original (top) and transformed (bottom)
"""

from pathlib import Path

import torch
import vultorch

if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"

# ── Load image ────────────────────────────────────────────────────
# vultorch.imread uses stb_image internally — supports PNG, JPEG, BMP,
# TGA, HDR.  No PIL, no OpenCV, no extra dependencies.
img_path = Path(__file__).resolve().parents[1] / "docs" / "images" / "pytorch_logo.png"
original = vultorch.imread(img_path, channels=3, device=device)
H, W, C = original.shape

# Working copy for transforms (we never modify the original)
transformed = original.clone()

# ── View + panels ─────────────────────────────────────────────────
view = vultorch.View("05 - Image Viewer", 1024, 768)
ctrl = view.panel("Controls", side="left", width=0.28)
img_panel = view.panel("Image")

# Two canvases: original on top, transformed on bottom
canvas_orig = img_panel.canvas("Original")
canvas_orig.bind(original)

canvas_xform = img_panel.canvas("Transformed")
canvas_xform.bind(transformed)

# ── State ─────────────────────────────────────────────────────────
TRANSFORMS = [
    "None",
    "Horizontal Flip",
    "Vertical Flip",
    "Grayscale",
    "Invert",
    "Sepia",
]

state = {
    "brightness": 0.0,
    "contrast": 1.0,
    "last_transform": -1,       # force first-frame update
    "last_brightness": None,
    "last_contrast": None,
}


def apply_transform(img: torch.Tensor, idx: int) -> torch.Tensor:
    """Apply a selected transform to the image."""
    if idx == 0:                 # None
        return img.clone()
    elif idx == 1:               # Horizontal Flip
        return img.flip(1)
    elif idx == 2:               # Vertical Flip
        return img.flip(0)
    elif idx == 3:               # Grayscale
        gray = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        return gray.unsqueeze(-1).expand_as(img).contiguous()
    elif idx == 4:               # Invert
        return 1.0 - img
    elif idx == 5:               # Sepia
        r = img[:, :, 0] * 0.393 + img[:, :, 1] * 0.769 + img[:, :, 2] * 0.189
        g = img[:, :, 0] * 0.349 + img[:, :, 1] * 0.686 + img[:, :, 2] * 0.168
        b = img[:, :, 0] * 0.272 + img[:, :, 1] * 0.534 + img[:, :, 2] * 0.131
        return torch.stack([r, g, b], dim=-1).clamp(0, 1)
    return img.clone()


def apply_brightness_contrast(img: torch.Tensor,
                               brightness: float,
                               contrast: float) -> torch.Tensor:
    """Apply brightness and contrast adjustment."""
    # contrast: multiply around 0.5 midpoint
    # brightness: offset
    return ((img - 0.5) * contrast + 0.5 + brightness).clamp(0, 1)


@ctrl.on_frame
def draw_controls():
    ctrl.text(f"Image: {img_path.name}")
    ctrl.text(f"Size: {W} × {H}")
    ctrl.text(f"FPS: {view.fps:.1f}")
    ctrl.separator()

    # ── Transform selector ────────────────────────────────────
    ctrl.text("Transform")
    xform_idx = ctrl.combo("##transform", TRANSFORMS, default=0)

    ctrl.separator()

    # ── Brightness / Contrast ─────────────────────────────────
    ctrl.text("Adjustments")
    brightness = ctrl.slider("Brightness", -1.0, 1.0, default=0.0)
    contrast = ctrl.slider("Contrast", 0.0, 3.0, default=1.0)

    # Only recompute when something changed
    changed = (xform_idx != state["last_transform"]
               or brightness != state["last_brightness"]
               or contrast != state["last_contrast"])

    if changed:
        result = apply_transform(original, xform_idx)
        result = apply_brightness_contrast(result, brightness, contrast)
        transformed[:] = result
        state["last_transform"] = xform_idx
        state["last_brightness"] = brightness
        state["last_contrast"] = contrast

    ctrl.separator()

    # ── Filter toggle ─────────────────────────────────────────
    ctrl.text("Sampling Filter")
    filter_idx = ctrl.combo("##filter", ["Linear", "Nearest"], default=0)
    filt = "nearest" if filter_idx == 1 else "linear"
    canvas_orig.filter = filt
    canvas_xform.filter = filt

    ctrl.separator()

    # ── Save ──────────────────────────────────────────────────
    ctrl.text("Save Output")
    save_path = ctrl.input_text("Path", default="output.png")

    if ctrl.button("Save Image", width=140):
        try:
            canvas_xform.save(save_path)
            state["save_msg"] = f"Saved to {save_path}"
        except Exception as e:
            state["save_msg"] = f"Error: {e}"

    if "save_msg" in state:
        ctrl.text_wrapped(state["save_msg"])

    ctrl.separator()
    ctrl.text_wrapped(
        "Pick a transform from the drop-down, adjust brightness "
        "and contrast with the sliders, then save the result. "
        "Try switching the filter to 'Nearest' when zoomed in — "
        "you'll see individual pixels instead of blurry interpolation."
    )


view.run()
