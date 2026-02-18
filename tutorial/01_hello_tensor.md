# 01 — Hello Tensor

> **Example file:** `examples/01_hello_tensor.py`

Tired of every research repo inventing its own janky tensor viewer with
matplotlib hacks and `cv2.imshow` spaghetti?  Yeah, us too.

Vultorch gets your CUDA tensor on screen in **four lines** — no saving PNGs,
no CPU round-trips, no `plt.pause(0.001)` nonsense.

## The mental model

If you've ever used matplotlib, you know this pattern: figure → axes → plot.
Vultorch is the same idea, but for GPU tensors:

```
View          ← the OS window (like plt.figure)
 └─ Panel     ← a named region inside it (like plt.subplot)
     └─ Canvas  ← a display slot that shows a tensor (like ax.imshow)
         └─ bind(tensor)  ← connects data to the slot
```

Four objects, that's it:

| Object | What it is | One-liner |
|--------|------------|-----------|
| **View** | The OS window | `vultorch.View("title", w, h)` |
| **Panel** | A dockable sub-window inside the View | `view.panel("name")` |
| **Canvas** | A GPU image slot inside a Panel | `panel.canvas("name")` |
| **bind()** | Connects a tensor to a Canvas | `canvas.bind(t)` |

Chain them together, call `run()`, done.

!!! info "What's a Panel, exactly?"
    A Panel is a movable, resizable sub-window inside your main window.
    Think of it like a floating sticky note that you can drag around,
    snap to the edges, or stack with other panels.  You don't need to
    manage any of this — Vultorch auto-arranges them for you.  You'll
    see more of this in the next chapter.

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
   The canvas auto-fills its panel by default (`fit=True`), meaning
   it stretches to use all available space — like a single `imshow`
   that fills the whole figure.

3. **Run** — `view.run()` is a **blocking loop** (like `plt.show()`).
   It keeps the window open, re-draws the tensor every frame (~60 times
   per second), and handles OS events (resize, close, etc.) for you.
   Close the window to exit — your Python script resumes after `run()`
   returns.

!!! tip
    The four setup lines collapse into a one-liner if you're feeling fancy:
    `view.panel("Viewer").canvas("gradient").bind(t)`

!!! info "Why not just use `plt.imshow`?"
    matplotlib copies your tensor to CPU, converts it to a numpy array,
    renders it on the CPU via Agg, then blits it to a window.  Vultorch
    keeps everything on the GPU — the tensor goes straight from CUDA
    memory to your screen via Vulkan.  That's why it can refresh at
    60 FPS even for large images.
