"""
09 - Live Hyperparameter Tuning
================================
Change learning rate, optimizer, weight decay, and loss function while
the network is training — no restart, no checkpoint, no boilerplate.

This example uses step()/end_step() instead of run(), giving the
training loop control of the outer iteration while Vultorch handles
rendering on each step.

Key concepts
------------
- step() / end_step()  : Training-loop-owned event loop
- Runtime LR change    : Modify optimizer param_groups live
- Optimizer hot-swap   : Switch Adam ↔ SGD ↔ AdamW without restart
- Loss function combo  : MSE / L1 / Huber at runtime
- Model reset          : Reinitialize weights with one button

Layout
------
Left sidebar : Hyperparameter controls (LR, optimizer, loss, etc.)
Top          : Prediction image (live)
Bottom       : Metrics (loss curve, PSNR, iteration counter)
"""

from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import vultorch

if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"

# ── Load target image ────────────────────────────────────────────
img_path = Path(__file__).resolve().parents[1] / "docs" / "images" / "pytorch_logo.png"
gt = vultorch.imread(img_path, channels=3, size=(256, 256), device=device)
H, W = gt.shape[0], gt.shape[1]

# ── Coordinate grid ──────────────────────────────────────────────
ys = torch.linspace(-1.0, 1.0, H, device=device)
xs = torch.linspace(-1.0, 1.0, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")
coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
target = gt.reshape(-1, 3)


# ── Model ─────────────────────────────────────────────────────────
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


def make_model():
    return CoordMLP().to(device)


model = make_model()

# ── Optimizer / loss configuration ────────────────────────────────
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
    return torch.optim.Adam(model.parameters(), lr=lr)


def compute_loss(pred, target, loss_name):
    if loss_name == "MSE":
        return F.mse_loss(pred, target)
    elif loss_name == "L1":
        return F.l1_loss(pred, target)
    elif loss_name == "Huber":
        return F.smooth_l1_loss(pred, target)
    return F.mse_loss(pred, target)


def compute_psnr(mse):
    if mse <= 0:
        return 50.0
    return -10.0 * math.log10(mse)


# ── State ─────────────────────────────────────────────────────────
state = {
    "lr": 2e-3,
    "optimizer_idx": 0,       # Adam
    "loss_idx": 0,            # MSE
    "weight_decay": 0.0,
    "steps_per_frame": 6,
    "iter": 0,
    "loss": 1.0,
    "psnr": 0.0,
    "loss_history": [],
    "psnr_history": [],
    "prev_optimizer_idx": 0,  # for detecting hot-swap
    "needs_reset": False,
}

optimizer = make_optimizer(model, "Adam", state["lr"], state["weight_decay"])

# ── View + panels ─────────────────────────────────────────────────
view = vultorch.View("09 - Live Hyperparameter Tuning", 1100, 900)
ctrl = view.panel("Controls", side="left", width=0.30)
pred_panel = view.panel("Prediction")
metrics_panel = view.panel("Metrics")

pred_rgba = vultorch.create_tensor(H, W, 4, device, name="pred",
                                    window=view.window)
pred_rgba[:, :, 3] = 1.0
pred_panel.canvas("pred").bind(pred_rgba)


# ── Control panel ─────────────────────────────────────────────────
@ctrl.on_frame
def draw_controls():
    ctrl.text(f"FPS: {view.fps:.1f}")
    ctrl.text(f"Iteration: {state['iter']}")
    ctrl.separator()

    # ── Learning rate (log-scale slider mapped to linear) ──
    ctrl.text("Learning Rate")
    # Slider from -5 to -1 (log10 scale), default = log10(2e-3) ≈ -2.7
    log_lr = ctrl.slider("log10(LR)", -5.0, -1.0, default=-2.7)
    state["lr"] = 10.0 ** log_lr
    ctrl.text(f"  LR = {state['lr']:.2e}")

    # Apply LR change to current optimizer
    for pg in optimizer.param_groups:
        pg["lr"] = state["lr"]

    ctrl.separator()

    # ── Optimizer selection ──
    ctrl.text("Optimizer")
    state["optimizer_idx"] = ctrl.combo("Optimizer", OPTIMIZER_NAMES,
                                         default=0)

    # ── Weight decay ──
    state["weight_decay"] = ctrl.slider("Weight Decay", 0.0, 0.1,
                                         default=0.0)

    ctrl.separator()

    # ── Loss function ──
    ctrl.text("Loss Function")
    state["loss_idx"] = ctrl.combo("Loss", LOSS_NAMES, default=0)

    ctrl.separator()

    # ── Steps per frame ──
    state["steps_per_frame"] = ctrl.slider_int("Steps/Frame", 1, 32,
                                                default=6)

    ctrl.separator()

    # ── Reset button ──
    if ctrl.button("Reset Model", width=140):
        state["needs_reset"] = True

    ctrl.separator()
    ctrl.text_wrapped(
        "Change any hyperparameter while training continues. "
        "The optimizer is hot-swapped when you switch — no restart needed. "
        "Watch how different LR / optimizer / loss combinations affect "
        "convergence speed and final quality."
    )


# ── Metrics panel ─────────────────────────────────────────────────
@metrics_panel.on_frame
def draw_metrics():
    metrics_panel.text(f"MSE Loss: {state['loss']:.6f}")
    metrics_panel.text(f"PSNR: {state['psnr']:.2f} dB")
    metrics_panel.text(f"Optimizer: {OPTIMIZER_NAMES[state['optimizer_idx']]}")
    metrics_panel.text(f"Loss fn: {LOSS_NAMES[state['loss_idx']]}")
    metrics_panel.separator()

    # Loss curve
    metrics_panel.text("Loss Curve")
    if state["loss_history"]:
        metrics_panel.plot(state["loss_history"], label="##loss",
                           overlay=f"{state['loss']:.5f}", height=80)

    # PSNR curve
    metrics_panel.text("PSNR Curve")
    if state["psnr_history"]:
        metrics_panel.plot(state["psnr_history"], label="##psnr",
                           overlay=f"{state['psnr']:.1f} dB", height=80)


# ── Training loop using step() / end_step() ──────────────────────
try:
    while view.step():
        # ── Handle model reset ──
        if state["needs_reset"]:
            model = make_model()
            optimizer = make_optimizer(
                model, OPTIMIZER_NAMES[state["optimizer_idx"]],
                state["lr"], state["weight_decay"])
            state["iter"] = 0
            state["loss"] = 1.0
            state["psnr"] = 0.0
            state["loss_history"].clear()
            state["psnr_history"].clear()
            state["prev_optimizer_idx"] = state["optimizer_idx"]
            state["needs_reset"] = False

        # ── Handle optimizer hot-swap ──
        if state["optimizer_idx"] != state["prev_optimizer_idx"]:
            optimizer = make_optimizer(
                model, OPTIMIZER_NAMES[state["optimizer_idx"]],
                state["lr"], state["weight_decay"])
            state["prev_optimizer_idx"] = state["optimizer_idx"]

        # ── Training steps ──
        for _ in range(state["steps_per_frame"]):
            optimizer.zero_grad(set_to_none=True)
            out = model(coords)
            loss = compute_loss(out, target, LOSS_NAMES[state["loss_idx"]])
            loss.backward()
            optimizer.step()

            state["iter"] += 1
            state["loss"] = loss.item()

        # ── Update display ──
        with torch.no_grad():
            pred = model(coords).reshape(H, W, 3).clamp_(0, 1)
            pred_rgba[:, :, :3] = pred

            # Compute PSNR from MSE (always use MSE for PSNR regardless of loss fn)
            mse = F.mse_loss(pred.reshape(-1, 3), target).item()
            state["psnr"] = compute_psnr(mse)

        # History
        state["loss_history"].append(state["loss"])
        state["psnr_history"].append(state["psnr"])
        if len(state["loss_history"]) > 300:
            state["loss_history"] = state["loss_history"][-300:]
            state["psnr_history"] = state["psnr_history"][-300:]

        view.end_step()
finally:
    view.close()
