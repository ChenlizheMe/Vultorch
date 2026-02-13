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
This time we bring two new toys:

| New thing | What it does | How to use |
|-----------|-------------|------------|
| **on_frame** | Per-frame callback — train and update here | `@view.on_frame` |
| **create_tensor** | GPU shared-memory tensor | `vultorch.create_tensor(H, W, ...)` |

Write any PyTorch code inside the callback; Vultorch handles the
tensor-to-screen dance every frame.

## Full code

```python
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vultorch
ui = vultorch.ui

try:
    from PIL import Image
except ImportError as exc:
    raise RuntimeError("Please install pillow: pip install pillow") from exc


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

# Load the target image
img_path = Path(__file__).resolve().parents[1] / "docs" / "images" / "pytorch_logo.png"
img = Image.open(img_path).convert("RGB").resize((256, 256), Image.BILINEAR)
gt = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).to(device)

H, W = gt.shape[0], gt.shape[1]
ys = torch.linspace(-1.0, 1.0, H, device=device)
xs = torch.linspace(-1.0, 1.0, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")
coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (H*W, 2)
target = gt.reshape(-1, 3)                               # (H*W, 3)

model = TinyMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

view = vultorch.View("03 - Training Test", 1280, 760)
gt_panel = view.panel("GT")
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
    "layout_done": False,
}


@view.on_frame
def train_and_render():
    # ---- first-frame layout: top row left/right, bottom info ----
    if not state["layout_done"]:
        dockspace_id = ui.dock_space_over_viewport(flags=8)
        ui.dock_builder_remove_node(dockspace_id)
        ui.dock_builder_add_node(dockspace_id, 1 << 10)
        ui.dock_builder_set_node_size(dockspace_id, 1280.0, 760.0)

        info_node, top_node = ui.dock_builder_split_node(dockspace_id, 3, 0.28)
        left_node, right_node = ui.dock_builder_split_node(top_node, 0, 0.5)

        ui.dock_builder_dock_window("GT", left_node)
        ui.dock_builder_dock_window("Prediction", right_node)
        ui.dock_builder_dock_window("Info", info_node)
        ui.dock_builder_finish(dockspace_id)
        state["layout_done"] = True

    # ---- train a few steps ----
    for _ in range(state["steps_per_frame"]):
        optimizer.zero_grad(set_to_none=True)
        out = model(coords)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()

        state["iter"] += 1
        state["loss"] = loss.item()
        state["ema"] = state["ema"] * 0.98 + state["loss"] * 0.02

    # ---- write prediction into the display tensor ----
    with torch.no_grad():
        pred = model(coords).reshape(H, W, 3).clamp_(0, 1)
        pred_rgba[:, :, :3] = pred

    # ---- Info panel ----
    ui.begin("Info", True, 0)
    ui.text(f"FPS: {view.fps:.1f}")
    ui.text(f"Iteration: {state['iter']}")
    ui.text(f"Loss (MSE): {state['loss']:.6f}")
    ui.text(f"EMA Loss: {state['ema']:.6f}")

    state["steps_per_frame"] = ui.slider_int(
        "Steps / Frame", state["steps_per_frame"], 1, 32
    )
    progress = min(1.0, state["iter"] / 3000.0)
    ui.progress_bar(progress, overlay=f"Training progress {progress * 100:.1f}%")
    ui.text_wrapped(
        "Left is GT, right is prediction. Increase 'Steps / Frame' for faster fitting."
    )
    ui.end()


view.run()
```

That's it. Run it and watch the grey blob on the right morph into the
PyTorch logo in a few seconds.

## What just happened?

1. **Data** — PIL loads the image, we convert it to a float32 CUDA tensor
   for GT. Pixel coordinates get `meshgrid`'d into `(H*W, 2)`, normalized
   to `[-1, 1]`.

2. **Model** — a two-hidden-layer MLP (64 wide). Takes `(x, y)`, outputs
   `(r, g, b)`. Small enough to run inside a per-frame callback without
   tanking your framerate.

3. **on_frame callback** — called once per frame. It does three things:
   set up the dock layout on the first frame, run N training steps, then
   write the prediction into `pred_rgba`.

4. **Info panel** — drawn with ImGui's `ui.begin()` / `ui.end()`. Text,
   sliders, progress bars — any widget you want, right in the window.

## Key takeaways

1. **`@view.on_frame`** — you can run arbitrary PyTorch code in the
   callback. At the end of each frame, Vultorch uploads every bound
   tensor to the screen automatically.

2. **`create_tensor`** — looks and feels like `torch.zeros`, but the
   underlying memory is Vulkan/CUDA shared. Display is zero-copy.

3. **Manual docking** — `dock_builder_split_node` lets you slice the
   window any way you like. Directions: `0=left, 1=right, 2=up, 3=down`,
   ratio is a float.

4. **No terminal spam** — all live stats live in the Info panel.
   Your console stays clean for warnings and tracebacks.

!!! tip
    Crank `Steps / Frame` up to 32 for blazing-fast convergence.
    But don't get too greedy — go too high and your framerate will
    drop because each frame spends more time training.

!!! note
    `create_tensor` is called once at init, not every frame.
    After that you just write into the tensor each frame — practically free.
