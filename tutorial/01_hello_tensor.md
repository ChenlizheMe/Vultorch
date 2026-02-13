# 01 — Hello Tensor

> **Example file:** `examples/01_hello_tensor.py`

The simplest way to use Vultorch: create a window, add a panel, put a canvas
on it, bind a CUDA tensor, and run.

## Core hierarchy

```
View          (top-level window)
 └── Panel    (dockable ImGui sub-window)
      └── Canvas   (GPU image slot for a tensor)
           └── bind(tensor)  →  auto-rendered every frame
```

That's the entire mental model.

## Full code

```python
import torch
import vultorch

# -- 1. Prepare data -------------------------------------------------------
# A 256×256 RGB gradient.  Any (H, W, C) float32 CUDA tensor will work.
H, W = 256, 256
x = torch.linspace(0, 1, W, device="cuda")            # horizontal ramp
y = torch.linspace(0, 1, H, device="cuda")            # vertical ramp
t = torch.stack([
    x.unsqueeze(0).expand(H, W),                       # R channel: left→right
    y.unsqueeze(1).expand(H, W),                       # G channel: top→bottom
    torch.full((H, W), 0.3, device="cuda"),            # B channel: constant
], dim=-1)                                             # shape: (256, 256, 3)

# -- 2. Create View → Panel → Canvas → bind tensor -------------------------
view   = vultorch.View("01 - Hello Tensor", 512, 512)  # open a 512×512 window
panel  = view.panel("Viewer")                          # add one panel
canvas = panel.canvas("gradient")                      # add a canvas to it
canvas.bind(t)                                         # bind the tensor

# The four lines above can also be written as a one-liner:
# view.panel("Viewer").canvas("gradient").bind(t)

# -- 3. Run -----------------------------------------------------------------
# run() blocks until the user closes the window.
# The canvas auto-fills the panel and re-uploads from the bound tensor
# every frame.
view.run()
```

## Step-by-step breakdown

1. **Prepare the data** — any `(H, W)`, `(H, W, 1)`, `(H, W, 3)`, or `(H, W, 4)`
   float32 / float16 / uint8 tensor on CUDA will work.  Vultorch handles RGBA
   expansion internally.

2. **Build the object tree** — `View` creates the OS window, `panel()` adds a
   dockable ImGui window, `canvas()` adds a GPU image slot inside that panel,
   and `bind()` connects the tensor.

3. **Run** — `view.run()` enters a blocking event loop.  Every frame, the canvas
   re-uploads the bound tensor and renders it.  The canvas auto-fills the panel
   area by default (`fit=True`).

!!! tip
    The four setup lines can be collapsed into a single chain:
    `view.panel("Viewer").canvas("gradient").bind(t)`
