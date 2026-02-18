# 09 — Live Hyperparameter Tuning

> **Example file:** `examples/09_live_tuning.py`

You're training a model. The loss is stuck. You want to try a
different learning rate. So you kill the process, edit the script,
restart, wait for initialization, wait for the first 500 iterations
to re-cover lost ground, and *then* see if your new LR helps.

Or — you could just drag a slider.

This example changes the training recipe on the fly: learning rate,
optimizer, loss function, weight decay. No restart. No checkpoint.
The model keeps training with the new settings applied instantly.

The secret weapon is `step()` / `end_step()` — they give your
training loop ownership of the main loop, while Vultorch handles
rendering on each step.

## New friends

| New thing | What it does | Why it matters |
|-----------|-------------|----------------|
| `view.step()` | Process one frame, return `False` on close | Your training `while` loop owns the outer iteration |
| `view.end_step()` | Finish the current frame | Pair with `step()` — replaces `run()` |
| `view.close()` | Explicitly destroy the window | Called in `finally` for clean shutdown |
| Log-scale LR slider | `slider` from -5 to -1, then `10 ** value` | Linear sliders are useless for learning rates — you need log scale |
| Optimizer hot-swap | Detect combo change, rebuild optimizer | Switch Adam ↔ SGD ↔ AdamW without restarting |
| `compute_loss()` | MSE / L1 / Huber selected by combo | Different losses for different problems, switchable at runtime |

## What we're building

The same coordinate MLP from example 08, but now with a control sidebar
that lets you:

- **Drag the LR** on a log-scale slider (1e-5 to 0.1)
- **Switch optimizer** between Adam, SGD (with momentum), AdamW
- **Change loss function** between MSE, L1, Huber
- **Adjust weight decay** in real time
- **Reset the model** to random weights with one button
- **See the effect instantly** in the prediction image and loss curve

## Full code

```python
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import vultorch

device = "cuda"

# Load target image
img_path = Path(__file__).resolve().parents[1] / "docs" / "images" / "pytorch_logo.png"
gt = vultorch.imread(img_path, channels=3, size=(256, 256), device=device)
H, W = gt.shape[0], gt.shape[1]

# Coordinate grid
coords = ...  # (H*W, 2) in [-1, 1]
target = gt.reshape(-1, 3)


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


OPTIMIZER_NAMES = ["Adam", "SGD", "AdamW"]
LOSS_NAMES = ["MSE", "L1", "Huber"]


def make_optimizer(model, name, lr, weight_decay):
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
    elif name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr,
                               momentum=0.9, weight_decay=weight_decay)
    elif name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)


def compute_loss(pred, target, loss_name):
    if loss_name == "MSE":  return F.mse_loss(pred, target)
    elif loss_name == "L1": return F.l1_loss(pred, target)
    elif loss_name == "Huber": return F.smooth_l1_loss(pred, target)


model = CoordMLP().to(device)
optimizer = make_optimizer(model, "Adam", 2e-3, 0.0)

# View + panels
view = vultorch.View("09 - Live Hyperparameter Tuning", 1100, 900)
ctrl = view.panel("Controls", side="left", width=0.30)
pred_panel = view.panel("Prediction")
metrics_panel = view.panel("Metrics")

pred_rgba = vultorch.create_tensor(H, W, 4, device, name="pred",
                                    window=view.window)
pred_rgba[:, :, 3] = 1.0
pred_panel.canvas("pred").bind(pred_rgba)

state = {"iter": 0, "loss": 1.0, "lr": 2e-3,
         "optimizer_idx": 0, "loss_idx": 0, "weight_decay": 0.0,
         "steps_per_frame": 6, "needs_reset": False,
         "prev_optimizer_idx": 0, ...}


@ctrl.on_frame
def draw_controls():
    log_lr = ctrl.slider("log10(LR)", -5.0, -1.0, default=-2.7)
    state["lr"] = 10.0 ** log_lr
    for pg in optimizer.param_groups:
        pg["lr"] = state["lr"]

    state["optimizer_idx"] = ctrl.combo("Optimizer", OPTIMIZER_NAMES)
    state["loss_idx"] = ctrl.combo("Loss", LOSS_NAMES)
    state["weight_decay"] = ctrl.slider("Weight Decay", 0.0, 0.1)

    if ctrl.button("Reset Model"):
        state["needs_reset"] = True


# Training loop — step()/end_step() instead of run()
try:
    while view.step():
        if state["needs_reset"]:
            model = CoordMLP().to(device)
            optimizer = make_optimizer(...)
            state["iter"] = 0
            state["needs_reset"] = False

        if state["optimizer_idx"] != state["prev_optimizer_idx"]:
            optimizer = make_optimizer(...)
            state["prev_optimizer_idx"] = state["optimizer_idx"]

        for _ in range(state["steps_per_frame"]):
            optimizer.zero_grad(set_to_none=True)
            out = model(coords)
            loss = compute_loss(out, target, LOSS_NAMES[state["loss_idx"]])
            loss.backward()
            optimizer.step()
            state["iter"] += 1

        with torch.no_grad():
            pred = model(coords).reshape(H, W, 3).clamp_(0, 1)
            pred_rgba[:, :, :3] = pred

        view.end_step()
finally:
    view.close()
```

*(Abridged — see `examples/09_live_tuning.py` for the complete code.)*

## What just happened?

### step() / end_step() — your training loop in charge

In examples 01–08 we used `view.run()`. That's convenient — Vultorch
owns the loop and calls your callbacks. But for training, *you* want
to own the loop:

```python
try:
    while view.step():
        # ... your training code here ...
        view.end_step()
finally:
    view.close()
```

`step()` processes one frame's worth of input and rendering. It returns
`False` when the user closes the window. `end_step()` finishes the frame.
Your training code goes between them.

Think of it like matplotlib's `ion()` mode — the plot updates, but your
script keeps running. Except here it's 60 fps, zero-copy, and you get
sliders.

### Log-scale LR slider

Linear sliders are terrible for learning rates. The difference between
1e-4 and 2e-4 matters, but on a linear slider from 0 to 0.1, that's
an invisible notch. Log scale fixes this:

```python
log_lr = ctrl.slider("log10(LR)", -5.0, -1.0, default=-2.7)
state["lr"] = 10.0 ** log_lr
```

Slider position -5.0 = LR 1e-5, position -1.0 = LR 0.1. Now you can
finely control anything from tiny fine-tuning rates to aggressive
warm-up rates.

To apply it immediately:

```python
for pg in optimizer.param_groups:
    pg["lr"] = state["lr"]
```

PyTorch optimizers read `lr` from `param_groups` on every step. Change
it there, and the next `optimizer.step()` uses the new value. No
re-creation needed.

### Optimizer hot-swap

When you switch the combo from Adam to SGD, we need to create a new
optimizer object — there's no way to morph one into the other:

```python
if state["optimizer_idx"] != state["prev_optimizer_idx"]:
    optimizer = make_optimizer(
        model, OPTIMIZER_NAMES[state["optimizer_idx"]],
        state["lr"], state["weight_decay"])
    state["prev_optimizer_idx"] = state["optimizer_idx"]
```

The model's parameters stay the same — only the optimizer state
(momentum buffers, Adam's running averages) gets reset. This is
actually useful: sometimes switching to SGD for a few iterations
can shake the model out of a local minimum that Adam got stuck in.

### Loss function switching

Different loss functions emphasize different error patterns:

- **MSE** — penalizes large errors quadratically. Standard for PSNR.
- **L1** — penalizes all errors equally. More robust to outliers, but
  gradients are constant near zero (can cause oscillation).
- **Huber** — MSE near zero, L1 far away. The "best of both worlds"
  that everyone uses for depth estimation.

Switching at runtime lets you see the effect on convergence without
restarting. Try MSE for 1000 iterations, then switch to Huber — you
might see the loss drop further as Huber handles outlier pixels better.

### Model reset

```python
if ctrl.button("Reset Model"):
    state["needs_reset"] = True
```

Deferred to the training loop (not inside the callback) so that the
model re-creation happens at the right time. The new model gets fresh
random weights, a fresh optimizer, and the counters reset to zero.

## Key takeaways

1. **`step()` / `end_step()`** — use these instead of `run()` when
   you want custom training loop control. Your `while` loop owns
   the iteration.

2. **Log-scale LR slider** — learning rates span orders of magnitude.
   A linear slider is useless; use `10 ** slider_value`.

3. **Optimizer hot-swap** — change Adam to SGD to AdamW at runtime.
   The model parameters survive, only optimizer state resets.

4. **Loss function switching** — MSE, L1, Huber at runtime. Different
   losses reveal different training dynamics.

5. **Immediate feedback** — every slider change takes effect on the
   very next training step. No restart, no checkpoint, no boilerplate.

!!! tip
    Start with Adam at LR 2e-3, then once the loss plateaus, try
    switching to SGD with momentum. The different optimization
    landscape can sometimes improve final quality.

!!! note
    PSNR is always computed from MSE regardless of which loss function
    is active, so the PSNR number stays comparable across loss modes.
