# 08 — GT vs Prediction

> **Example file:** `examples/08_gt_vs_pred.py`

Every neural rendering researcher does the same thing a hundred times
a day: train a model, look at the result, compare it to the ground truth,
squint at the diff, wonder where the error is hiding.

The usual flow: save a PNG, open it in matplotlib next to the GT, compute
PSNR in a separate cell, plot the loss in TensorBoard, alt-tab between
three windows while your GPU sits idle. By the time you've assembled
your comparison, you've forgotten what hyperparameter you changed.

This chapter puts it all in one window: GT, prediction, error heatmap,
loss curve, and PSNR — all live, all 60 fps, all zero-copy.

## New friends

| New thing | What it does | Why it matters |
|-----------|-------------|----------------|
| Error heatmap | `|GT - pred|` amplified and turbo-colormapped | Sees errors that are invisible in raw pixel comparison |
| `panel.plot()` | Draws a line chart from a Python list | Live loss and PSNR curves without TensorBoard |
| `panel.progress()` | A progress bar | Quick visual for training completion |
| PSNR | $-10 \log_{10}(\text{MSE})$ computed from the live loss | The standard metric for image reconstruction quality |
| Error mode combo | Switch between L1, L2, per-channel max | Different error norms reveal different problems |
| Error gain slider | Amplify subtle errors for visibility | Low error regions become visible when multiplied by 5–20× |

## What we're building

A coordinate MLP fitting a target image (same as example 03, but upgraded).
Three image panels — GT, prediction, error heatmap — plus a metrics sidebar
with live loss/PSNR curves and controls.

## Full code

```python
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import vultorch

if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"

# Load target image
img_path = Path(__file__).resolve().parents[1] / "docs" / "images" / "pytorch_logo.png"
gt = vultorch.imread(img_path, channels=3, size=(256, 256), device=device)
H, W = gt.shape[0], gt.shape[1]

# Coordinate grid for the MLP
ys = torch.linspace(-1.0, 1.0, H, device=device)
xs = torch.linspace(-1.0, 1.0, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")
coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
target = gt.reshape(-1, 3)

# Simple coordinate MLP
class CoordMLP(nn.Module):
    def __init__(self, hidden=64, layers=3):
        super().__init__()
        net = [nn.Linear(2, hidden), nn.ReLU(inplace=True)]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
        net += [nn.Linear(hidden, 3), nn.Sigmoid()]
        self.net = nn.Sequential(*net)
    def forward(self, x):
        return self.net(x)

model = CoordMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

# Turbo colormap LUT (see example 07 for details)
TURBO_LUT = ...  # 256×3 on GPU

def apply_turbo(values):
    idx = (values.clamp(0, 1) * 255).long()
    return TURBO_LUT[idx]

# View + panels
view = vultorch.View("08 - GT vs Prediction", 1280, 1000)
metrics_panel = view.panel("Metrics", side="right", width=0.28)
gt_panel = view.panel("Ground Truth")
pred_panel = view.panel("Prediction")
error_panel = view.panel("Error Map")

gt_panel.canvas("gt").bind(gt)

pred_rgba = vultorch.create_tensor(H, W, 4, device, name="pred",
                                    window=view.window)
pred_rgba[:, :, 3] = 1.0
pred_panel.canvas("pred").bind(pred_rgba)

error_rgba = vultorch.create_tensor(H, W, 4, device, name="error",
                                     window=view.window)
error_rgba[:, :, 3] = 1.0
error_panel.canvas("error").bind(error_rgba)

ERROR_MODES = ["L1", "L2", "Per-Channel Max"]
state = {"iter": 0, "loss": 1.0, "psnr": 0.0, "steps_per_frame": 6,
         "error_mode": 0, "error_gain": 5.0,
         "loss_history": [], "psnr_history": []}


def compute_error_map(gt_img, pred_img, mode, gain):
    if mode == 0:    err = (gt_img - pred_img).abs().mean(dim=-1)
    elif mode == 1:  err = ((gt_img - pred_img)**2).mean(dim=-1).sqrt()
    else:            err = (gt_img - pred_img).abs().max(dim=-1).values
    return apply_turbo((err * gain).clamp(0, 1))


def compute_psnr(mse):
    return -10.0 * math.log10(mse) if mse > 0 else 50.0


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

    with torch.no_grad():
        pred = model(coords).reshape(H, W, 3).clamp_(0, 1)
        pred_rgba[:, :, :3] = pred

        error_color = compute_error_map(gt, pred, state["error_mode"],
                                         state["error_gain"])
        error_rgba[:, :, :3] = error_color
        state["psnr"] = compute_psnr(state["loss"])

    state["loss_history"].append(state["loss"])
    state["psnr_history"].append(state["psnr"])
    if len(state["loss_history"]) > 300:
        state["loss_history"] = state["loss_history"][-300:]
        state["psnr_history"] = state["psnr_history"][-300:]


@metrics_panel.on_frame
def draw_metrics():
    metrics_panel.text(f"FPS: {view.fps:.1f}")
    metrics_panel.text(f"Iteration: {state['iter']}")
    metrics_panel.separator()
    metrics_panel.text(f"MSE Loss: {state['loss']:.6f}")
    metrics_panel.text(f"PSNR: {state['psnr']:.2f} dB")
    metrics_panel.separator()

    metrics_panel.text("Loss Curve")
    if state["loss_history"]:
        metrics_panel.plot(state["loss_history"], label="##loss",
                           overlay=f"{state['loss']:.5f}", height=80)

    metrics_panel.text("PSNR Curve")
    if state["psnr_history"]:
        metrics_panel.plot(state["psnr_history"], label="##psnr",
                           overlay=f"{state['psnr']:.1f} dB", height=80)

    metrics_panel.separator()
    state["steps_per_frame"] = metrics_panel.slider_int(
        "Steps/Frame", 1, 32, default=6)
    state["error_mode"] = metrics_panel.combo("Error Mode", ERROR_MODES)
    state["error_gain"] = metrics_panel.slider("Error Gain", 1.0, 20.0,
                                                default=5.0)

    progress = min(1.0, state["iter"] / 5000.0)
    metrics_panel.progress(progress, overlay=f"{progress*100:.0f}%")


view.run()
```

*(Abridged — see `examples/08_gt_vs_pred.py` for the complete code including turbo LUT.)*

## What just happened?

### Error heatmap — seeing the invisible

The raw `|GT - pred|` difference is usually tiny floats near zero. If you
display it directly, you see a nearly-black image and conclude everything
is fine. Bad idea.

```python
err = (gt_img - pred_img).abs().mean(dim=-1)   # L1 error per pixel
err = (err * gain).clamp(0, 1)                  # amplify by 5–20×
heatmap = apply_turbo(err)                       # turbo colormap
```

The gain slider lets you amplify subtle errors until they become visible.
At 5× gain, you'll see exactly where the network struggles — edges,
fine textures, high-frequency regions. This is the kind of insight you
never get from a single PSNR number.

### PSNR — the neural rendering metric

$$\text{PSNR} = -10 \log_{10}(\text{MSE})$$

Updated every frame from the live loss. In one number, you know whether
your model is at 20 dB (blurry mess), 30 dB (decent), or 40 dB (sharp).
The live curve tells you when training plateaus.

```python
def compute_psnr(mse):
    return -10.0 * math.log10(mse) if mse > 0 else 50.0
```

### panel.plot() — loss curves without TensorBoard

```python
metrics_panel.plot(state["loss_history"],
                   label="##loss",
                   overlay=f"{state['loss']:.5f}",
                   height=80)
```

Takes a Python list of floats, draws a sparkline. The `overlay` text
shows on top of the chart. We keep the last 300 values for a rolling
window. No external logging library needed.

### Error modes — different norms for different bugs

- **L1** (`abs().mean(dim=-1)`) — average absolute error. Shows where
  the prediction is generally wrong.
- **L2** (`square().mean(dim=-1).sqrt()`) — root mean square error.
  Amplifies outliers more than L1.
- **Per-channel max** (`abs().max(dim=-1)`) — worst-case channel.
  Reveals color channel mismatches (e.g. the blue channel is wrong
  but R and G are fine).

Switch between them at runtime with the combo. Different error norms
make different problems visible.

### Upgrading from example 03

Example 03 had GT and prediction side by side. This example adds:

- **Error heatmap** — see *where* the model is wrong, not just *how much*.
- **PSNR + loss curves** — real-time metrics, not just a text counter.
- **Error gain** — amplify subtle errors for visibility.
- **Error mode switching** — L1/L2/per-channel at runtime.

This is the difference between "my model is training" and "I can debug
my model while it trains."

## Key takeaways

1. **Three-panel comparison** — GT, prediction, error heatmap. The
   fundamental visual debugging layout for any reconstruction task.

2. **Error heatmap = amplified diff + colormap** — the gain slider is
   the key. Low errors are invisible without amplification.

3. **Live PSNR** — one number that summarizes reconstruction quality.
   $-10 \log_{10}(\text{MSE})$, computed from the loss you already have.

4. **panel.plot()** — instant loss/metric curves inside the same window.
   No TensorBoard, no wandb, no alt-tab.

5. **Error modes at runtime** — different norms reveal different problems.
   L1 for general errors, L2 for outliers, per-channel max for color bugs.

!!! tip
    Increase "Error Gain" to 15–20× after training converges. You'll see
    the residual error pattern — it usually concentrates on edges and
    high-frequency textures, which tells you whether you need positional
    encoding or a deeper network.

!!! note
    This example uses a tiny 3-layer MLP for speed. Replace it with
    your actual model and training loop — the visualization code stays
    exactly the same.
