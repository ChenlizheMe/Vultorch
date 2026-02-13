# 01 — Hello Tensor

> **Example file:** `examples/01_hello_tensor.py`

Tired of every research repo inventing its own janky tensor viewer with
matplotlib hacks and `cv2.imshow` spaghetti?  Yeah, us too.

Vultorch gets your CUDA tensor on screen in **four lines** — no saving PNGs,
no CPU round-trips, no `plt.pause(0.001)` nonsense.

## The mental model

There are exactly four objects you need to know:

| Object | What it is | One-liner |
|--------|------------|-----------|
| **View** | The OS window | `vultorch.View("title", w, h)` |
| **Panel** | A dockable sub-window inside the View | `view.panel("name")` |
| **Canvas** | A GPU image slot inside a Panel | `panel.canvas("name")` |
| **bind()** | Connects a tensor to a Canvas | `canvas.bind(t)` |

Chain them together, call `run()`, done.

## Full code

```python
import torch
import vultorch

# A 256×256 RGB gradient — any (H,W,C) float32 CUDA tensor works
H, W = 256, 256
x = torch.linspace(0, 1, W, device="cuda")
y = torch.linspace(0, 1, H, device="cuda")
t = torch.stack([
    x.unsqueeze(0).expand(H, W),
    y.unsqueeze(1).expand(H, W),
    torch.full((H, W), 0.3, device="cuda"),
], dim=-1)  # (256, 256, 3)

view   = vultorch.View("01 - Hello Tensor", 512, 512)
panel  = view.panel("Viewer")
canvas = panel.canvas("gradient")
canvas.bind(t)
view.run()  # blocks until you close the window
```

That's it. No event loop boilerplate, no `begin_frame()` / `end_frame()`.

## What just happened?

1. **Data** — we made an RGB gradient on CUDA. Vultorch accepts `(H,W)`,
   `(H,W,1)`, `(H,W,3)`, or `(H,W,4)`, in float32 / float16 / uint8.
   It handles RGBA expansion for you.

2. **Object tree** — `View` → `Panel` → `Canvas` → `bind(tensor)`.
   The canvas auto-fills its panel by default (`fit=True`).

3. **Run** — `view.run()` enters a blocking event loop. Every frame the
   canvas re-uploads the bound tensor and renders it. Close the window
   to exit.

!!! tip
    The four setup lines collapse into a one-liner if you're feeling fancy:
    `view.panel("Viewer").canvas("gradient").bind(t)`
