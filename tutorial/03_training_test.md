# 03 — Training Test

> **Example file:** `examples/03_training_test.py`

Ever stared at a wall of decreasing loss numbers in your terminal for ten
minutes, feeling confident, only to discover the model's output is a solid
grey rectangle? Yeah, us too.

Reading loss values off a scrolling console is about as reliable as reading
tea leaves. This chapter puts GT and prediction side by side on screen so
you can *see* whether the network is actually learning.

## What we're building

A tiny MLP (2 → 64 → 64 → 3) fitting a 256×256 PyTorch logo in real time.
The window has three panels:

| Area | Content |
|------|---------|
| Left | **GT** panel — the target image (what you're fitting) |
| Right | **Prediction** panel — live network output, updated every frame |
| Bottom | **Info** panel — FPS, loss, iteration, progress bar, and a slider |

Everything on screen, nothing buried in the terminal.

## New friends

Chapters 01 and 02 were all static — `bind()` + `run()`, done.
This time we bring three new toys:

| New thing | What it does | How to use |
|-----------|-------------|------------|
| **on_frame** | Per-frame callback — train and update here | `@view.on_frame` |
| **Panel.on_frame** | Per-panel callback — widgets go here | `@info_panel.on_frame` |
| **create_tensor** | GPU shared-memory tensor | `vultorch.create_tensor(H, W, ...)` |
| **vultorch.imread** | Load image with zero dependencies | `vultorch.imread(path, channels=3)` |
| **side="bottom"** | Dock a panel to the bottom edge | `view.panel("Info", side="bottom")` |

Write any PyTorch code inside the view callback; put widgets inside the
panel callback.  Vultorch handles the tensor-to-screen dance every frame.

## Full code

```python
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import vultorch


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"

img_path = Path(__file__).resolve().parents[1] / "docs" / "images" / "pytorch_logo.png"
gt = vultorch.imread(img_path, channels=3, size=(256, 256), device=device)

H, W = gt.shape[0], gt.shape[1]
ys = torch.linspace(-1.0, 1.0, H, device=device)
xs = torch.linspace(-1.0, 1.0, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")
coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
target = gt.reshape(-1, 3)

model = TinyMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

# -- View + panels (high-level declarative API) -------------------------
view = vultorch.View("03 - Training Test", 1280, 760)
info_panel = view.panel("Info", side="bottom", width=0.28)
gt_panel = view.panel("GT", side="left", width=0.5)
pred_panel = view.panel("Prediction")

gt_panel.canvas("gt").bind(gt)

# 4 channels — zero-copy GPU display path
pred_rgba = vultorch.create_tensor(H, W, channels=4, device=device,
                                   name="pred", window=view.window)
pred_rgba[:, :, 3] = 1.0
pred_panel.canvas("pred").bind(pred_rgba)

state = {
    "iter": 0,
    "loss": 1.0,
    "ema": 1.0,
    "steps_per_frame": 6,
}


@view.on_frame
def train():
    for _ in range(state["steps_per_frame"]):
        optimizer.zero_grad(set_to_none=True)
        out = model(coords)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()

        state["iter"] += 1
        state["loss"] = loss.item()
        state["ema"] = state["ema"] * 0.98 + state["loss"] * 0.02

    with torch.no_grad():
        pred = model(coords).reshape(H, W, 3).clamp_(0, 1)
        pred_rgba[:, :, :3] = pred


@info_panel.on_frame
def draw_info():
    info_panel.text(f"FPS: {view.fps:.1f}")
    info_panel.text(f"Iteration: {state['iter']}")
    info_panel.text(f"Loss (MSE): {state['loss']:.6f}")
    info_panel.text(f"EMA Loss: {state['ema']:.6f}")

    state["steps_per_frame"] = info_panel.slider_int(
        "Steps / Frame", 1, 32, default=6
    )
    progress = min(1.0, state["iter"] / 3000.0)
    info_panel.progress(progress,
                        overlay=f"Training progress {progress * 100:.1f}%")
    info_panel.text_wrapped(
        "Left is GT, right is prediction. "
        "Increase 'Steps / Frame' for faster fitting."
    )


view.run()
```

That's it. Run it and watch the grey blob on the right morph into the
PyTorch logo in a few seconds.

## What just happened?

1. **Data** — `vultorch.imread` loads the image straight into a float32
   CUDA tensor (no PIL, no numpy). Pixel coordinates get `meshgrid`'d into
   `(H*W, 2)`, normalized to `[-1, 1]`.

2. **Model** — a two-hidden-layer MLP (64 wide). Takes `(x, y)`, outputs
   `(r, g, b)`. Small enough to run inside a per-frame callback without
   tanking your framerate.

3. **Layout** — `side="bottom"` docks Info at the bottom 28 % of the
   window.  `side="left"` puts GT on the left half of the remaining
   space.  Prediction fills whatever is left.  No manual docking code.

4. **Two callbacks** — `@view.on_frame` runs the training loop.
   `@info_panel.on_frame` draws widgets (text, slider, progress bar)
   inside the Info panel.  Vultorch opens / closes the ImGui window
   for you.

## Key takeaways

1. **`@view.on_frame`** — you can run arbitrary PyTorch code in the
   callback. At the end of each frame, Vultorch uploads every bound
   tensor to the screen automatically.

2. **`create_tensor`** — looks and feels like `torch.zeros`, but the
   underlying memory is Vulkan/CUDA shared. Display is zero-copy.

3. **Declarative layout** — `side="left"` / `"right"` / `"bottom"` /
   `"top"` splits the window without manual docking code.

4. **Panel widgets** — `@panel.on_frame` runs inside the panel's ImGui
   window.  Use `panel.text()`, `panel.slider_int()`, `panel.progress()`
   instead of raw `ui.*` calls.

5. **No terminal spam** — all live stats live in the Info panel.
   Your console stays clean for warnings and tracebacks.

!!! tip
    Crank `Steps / Frame` up to 32 for blazing-fast convergence.
    But don't get too greedy — go too high and your framerate will
    drop because each frame spends more time training.

!!! note
    `create_tensor` is called once at init, not every frame.
    After that you just write into the tensor each frame — practically free.
