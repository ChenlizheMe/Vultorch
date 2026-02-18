# 05 — Image Viewer

> **Example file:** `examples/05_image_viewer.py`

So far we've been manufacturing tensors from thin air — gradients,
checkerboards, neural network outputs. Very impressive, but at some point
you probably want to look at an *actual image*. You know, the kind that
lives on your hard drive. In a `.png` file. Like a normal person.

"Just use PIL," you say. Sure — and then `torchvision.transforms`, and
then `numpy`, and then `cv2.cvtColor` because someone mixed up RGB and
BGR again, and then you're three Stack Overflow tabs deep wondering why
everything is upside-down and slightly green.

Vultorch has built-in image I/O. One function in, one function out.
No PIL, no OpenCV, no existential dread.

## New friends

| New thing | What it does | How to use |
|-----------|-------------|------------|
| **imread** | Load a file into a CUDA tensor | `vultorch.imread("photo.png")` |
| **imwrite** | Save a tensor to a file | `vultorch.imwrite("out.png", t)` |
| **Canvas.save()** | Save the canvas's bound tensor | `canvas.save("out.png")` |
| **panel.combo()** | Drop-down selector | `panel.combo("Pick", ["A","B"])` |
| **panel.input_text()** | Text input field | `panel.input_text("Path")` |
| **canvas.filter** | Sampling mode (`"linear"` / `"nearest"`) | `canvas.filter = "nearest"` |

## What we're building

A mini image viewer: load a photo, pick a transform from a drop-down,
tweak brightness / contrast with sliders, and save the result.

| Left | Right (two canvases) |
|------|----------------------|
| **Controls** — transform combo, brightness/contrast sliders, filter toggle, save | **Original** (top) |
| | **Transformed** (bottom) |

## Full code

```python
from pathlib import Path

import torch
import vultorch

if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"

# ── Load image ────────────────────────────────────────────────────
img_path = Path(__file__).resolve().parents[1] / "docs" / "images" / "pytorch_logo.png"
original = vultorch.imread(img_path, channels=3, device=device)
H, W, C = original.shape

# Working copy for transforms
transformed = original.clone()

# ── View + panels ─────────────────────────────────────────────────
view = vultorch.View("05 - Image Viewer", 1024, 768)
ctrl = view.panel("Controls", side="left", width=0.28)
img_panel = view.panel("Image")

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
    "last_transform": -1,
    "last_brightness": None,
    "last_contrast": None,
}


def apply_transform(img, idx):
    if idx == 0:    return img.clone()
    elif idx == 1:  return img.flip(1)               # horizontal flip
    elif idx == 2:  return img.flip(0)               # vertical flip
    elif idx == 3:                                    # grayscale
        gray = img[:,:,0]*0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114
        return gray.unsqueeze(-1).expand_as(img).contiguous()
    elif idx == 4:  return 1.0 - img                 # invert
    elif idx == 5:                                    # sepia
        r = img[:,:,0]*0.393 + img[:,:,1]*0.769 + img[:,:,2]*0.189
        g = img[:,:,0]*0.349 + img[:,:,1]*0.686 + img[:,:,2]*0.168
        b = img[:,:,0]*0.272 + img[:,:,1]*0.534 + img[:,:,2]*0.131
        return torch.stack([r, g, b], dim=-1).clamp(0, 1)
    return img.clone()


def apply_brightness_contrast(img, brightness, contrast):
    return ((img - 0.5) * contrast + 0.5 + brightness).clamp(0, 1)


@ctrl.on_frame
def draw_controls():
    ctrl.text(f"Image: {img_path.name}")
    ctrl.text(f"Size: {W} × {H}")
    ctrl.text(f"FPS: {view.fps:.1f}")
    ctrl.separator()

    # Transform selector
    ctrl.text("Transform")
    xform_idx = ctrl.combo("##transform", TRANSFORMS, default=0)

    ctrl.separator()

    # Brightness / Contrast
    ctrl.text("Adjustments")
    brightness = ctrl.slider("Brightness", -1.0, 1.0, default=0.0)
    contrast   = ctrl.slider("Contrast",    0.0, 3.0, default=1.0)

    changed = (xform_idx != state["last_transform"]
               or brightness != state["last_brightness"]
               or contrast   != state["last_contrast"])

    if changed:
        result = apply_transform(original, xform_idx)
        result = apply_brightness_contrast(result, brightness, contrast)
        transformed[:] = result
        state["last_transform"]  = xform_idx
        state["last_brightness"] = brightness
        state["last_contrast"]   = contrast

    ctrl.separator()

    # Filter toggle
    ctrl.text("Sampling Filter")
    filter_idx = ctrl.combo("##filter", ["Linear", "Nearest"], default=0)
    canvas_orig.filter  = "nearest" if filter_idx == 1 else "linear"
    canvas_xform.filter = "nearest" if filter_idx == 1 else "linear"

    ctrl.separator()

    # Save
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


view.run()
```

## What just happened?

### imread — images without the dependency hell

```python
original = vultorch.imread(img_path, channels=3, device=device)
```

One line. Returns a `(H, W, 3)` float32 CUDA tensor with values in
`[0, 1]`. Supports PNG, JPEG, BMP, TGA, HDR, PSD, and GIF (first
frame). Uses `stb_image` under the hood — no Python image library
needed.

Optional parameters:

- `channels=4` — force RGBA output.
- `size=(256, 256)` — resize after loading (bilinear interpolation).
- `device="cpu"` — keep it on CPU if you prefer.
- `shared=True` — allocate via `create_tensor` for zero-copy display.

### combo — the drop-down menu

```python
xform_idx = ctrl.combo("##transform", TRANSFORMS, default=0)
```

Shows a drop-down with the items in the list. Returns the **index** (int)
of the selected item. The state is managed automatically by the panel —
just pass `default=` for the initial selection.

The `##` prefix hides the label in ImGui (the text after `##` is used as
an internal ID only). Useful when you don't want a label next to your
widget.

### input_text — free text entry

```python
save_path = ctrl.input_text("Path", default="output.png")
```

Returns the current string. Type a filename, hit Enter or click Save.
`max_length=256` by default — plenty for a file path.

### Canvas.save() — one-line export

```python
canvas_xform.save(save_path)
```

Saves whatever tensor is currently bound to the canvas. The file format
is inferred from the extension (`.png`, `.jpg`, `.bmp`, `.tga`, `.hdr`).
Under the hood it calls `vultorch.imwrite()`.

### filter — nearest vs linear

```python
canvas_orig.filter = "nearest"   # pixel-perfect, blocky when zoomed
canvas_orig.filter = "linear"    # bilinear interpolation, smooth
```

Switch the sampling filter at any time. Try toggling it when the image
is stretched — `"nearest"` shows you the raw pixels, `"linear"` blurs
them into smooth gradients. For scientific visualization (segmentation
masks, attention maps) you almost always want `"nearest"`.

## Key takeaways

1. **`imread` / `imwrite`** — zero-dependency image I/O. Reads straight
   into a CUDA tensor, writes straight from one. No PIL, no numpy, no
   `cv2.cvtColor` misadventures.

2. **`combo`** — drop-down selection. Returns an int index. Perfect for
   mode switches, preset selectors, enum-style choices.

3. **`input_text`** — free-form string input. Useful for file paths,
   model names, experiment tags.

4. **`Canvas.save()`** — save the bound tensor to disk in one call.
   Extension determines the format.

5. **Lazy recomputation** — we only re-run the transform when a slider
   or combo value actually changes. Checking `changed` before doing
   tensor ops avoids wasting GPU cycles every frame.

!!! tip
    `imread` supports a `size=(H, W)` argument for resizing at load
    time. Useful when your image is 4K but you only need a 256×256
    preview.

!!! note
    `imwrite` accepts float32 tensors in `[0, 1]` as well as uint8
    tensors in `[0, 255]`. It handles the conversion automatically.
