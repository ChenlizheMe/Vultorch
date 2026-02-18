"""
08 - GT vs Prediction
=====================
Train a coordinate MLP to fit a target image, with live comparison:
GT | Prediction | Error heatmap, plus real-time PSNR and loss curves.

This is the neural rendering researcher's daily driver — the workflow
you'd use to debug NeRF, 3DGS, or any per-pixel reconstruction model.

Key concepts
------------
- Three-panel comparison : GT, prediction, error map side by side
- Error heatmap          : |GT - pred| amplified and turbo-colormapped
- Real-time PSNR         : Computed every N iterations from MSE
- Loss curve             : panel.plot() for live loss history
- Error mode combo       : Switch between L1, L2, per-channel error

Layout
------
Top-left    : GT image
Top-right   : Prediction (live training output)
Bottom-left : Error heatmap (turbo colormap)
Bottom-right: Metrics panel (loss curve, PSNR, controls)
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


model = CoordMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

# ── Turbo colormap LUT ───────────────────────────────────────────
_turbo_data = [
    (0.18995, 0.07176, 0.23217), (0.22500, 0.16354, 0.45096),
    (0.25107, 0.25237, 0.63374), (0.26816, 0.33825, 0.78410),
    (0.27628, 0.42118, 0.89867), (0.27543, 0.50115, 0.97314),
    (0.25862, 0.57958, 0.99977), (0.21382, 0.65886, 0.97959),
    (0.15844, 0.73551, 0.92305), (0.11167, 0.80569, 0.83891),
    (0.09267, 0.86554, 0.73729), (0.12014, 0.91193, 0.62376),
    (0.21005, 0.94471, 0.50370), (0.36246, 0.96358, 0.38218),
    (0.53937, 0.96733, 0.26478), (0.70818, 0.95619, 0.16024),
    (0.84890, 0.93174, 0.07844), (0.94613, 0.88965, 0.02170),
    (0.99314, 0.82592, 0.00815), (0.99593, 0.73782, 0.00596),
    (0.97686, 0.64362, 0.03964), (0.94277, 0.54879, 0.09171),
    (0.89434, 0.45784, 0.13384), (0.83565, 0.37445, 0.15964),
    (0.76849, 0.29883, 0.16793), (0.69665, 0.23170, 0.16267),
    (0.62189, 0.17379, 0.14827), (0.54688, 0.12566, 0.12896),
    (0.47323, 0.08735, 0.10879), (0.40232, 0.05829, 0.09011),
    (0.33529, 0.03650, 0.07393), (0.27004, 0.01514, 0.05070),
]


def _build_turbo_lut(n=256):
    keys = torch.tensor(_turbo_data, dtype=torch.float32)
    keys_t = keys.permute(1, 0).unsqueeze(0)
    lut = F.interpolate(keys_t, size=n, mode="linear",
                        align_corners=True)
    return lut.squeeze(0).permute(1, 0).contiguous().to(device)


TURBO_LUT = _build_turbo_lut()


def apply_turbo(values):
    idx = (values.clamp(0, 1) * 255).long()
    return TURBO_LUT[idx]


# ── View + panels ─────────────────────────────────────────────────
view = vultorch.View("08 - GT vs Prediction", 1280, 1000)
metrics_panel = view.panel("Metrics", side="right", width=0.28)
gt_panel = view.panel("Ground Truth")
pred_panel = view.panel("Prediction")
error_panel = view.panel("Error Map")

# Display tensors
gt_panel.canvas("gt").bind(gt)

pred_rgba = vultorch.create_tensor(H, W, 4, device, name="pred",
                                    window=view.window)
pred_rgba[:, :, 3] = 1.0
pred_panel.canvas("pred").bind(pred_rgba)

error_rgba = vultorch.create_tensor(H, W, 4, device, name="error",
                                     window=view.window)
error_rgba[:, :, 3] = 1.0
error_panel.canvas("error").bind(error_rgba)

# ── Training state ────────────────────────────────────────────────
ERROR_MODES = ["L1", "L2", "Per-Channel Max"]

state = {
    "iter": 0,
    "loss": 1.0,
    "psnr": 0.0,
    "steps_per_frame": 6,
    "error_mode": 0,
    "error_gain": 5.0,
    "loss_history": [],
    "psnr_history": [],
}


def compute_error_map(gt_img, pred_img, mode, gain):
    """Compute error map and apply turbo colormap."""
    if mode == 0:  # L1
        err = (gt_img - pred_img).abs().mean(dim=-1)
    elif mode == 1:  # L2
        err = ((gt_img - pred_img) ** 2).mean(dim=-1).sqrt()
    else:  # Per-channel max
        err = (gt_img - pred_img).abs().max(dim=-1).values
    # Amplify and clamp
    err = (err * gain).clamp(0, 1)
    return apply_turbo(err)


def compute_psnr(mse):
    """PSNR from MSE (assuming signal range [0, 1])."""
    if mse <= 0:
        return 50.0
    return -10.0 * math.log10(mse)


# ── Training + display update ────────────────────────────────────
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

    # Update prediction display
    with torch.no_grad():
        pred = model(coords).reshape(H, W, 3).clamp_(0, 1)
        pred_rgba[:, :, :3] = pred

        # Update error map
        error_color = compute_error_map(gt, pred, state["error_mode"],
                                         state["error_gain"])
        error_rgba[:, :, :3] = error_color

        # PSNR
        state["psnr"] = compute_psnr(state["loss"])

    # History (keep last 300 entries)
    state["loss_history"].append(state["loss"])
    state["psnr_history"].append(state["psnr"])
    if len(state["loss_history"]) > 300:
        state["loss_history"] = state["loss_history"][-300:]
        state["psnr_history"] = state["psnr_history"][-300:]


# ── Metrics panel ─────────────────────────────────────────────────
@metrics_panel.on_frame
def draw_metrics():
    metrics_panel.text(f"FPS: {view.fps:.1f}")
    metrics_panel.text(f"Iteration: {state['iter']}")
    metrics_panel.separator()

    # Key metrics
    metrics_panel.text(f"MSE Loss: {state['loss']:.6f}")
    metrics_panel.text(f"PSNR: {state['psnr']:.2f} dB")
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

    metrics_panel.separator()

    # Controls
    state["steps_per_frame"] = metrics_panel.slider_int(
        "Steps/Frame", 1, 32, default=6)

    metrics_panel.text("Error Visualization")
    state["error_mode"] = metrics_panel.combo(
        "Error Mode", ERROR_MODES, default=0)
    state["error_gain"] = metrics_panel.slider(
        "Error Gain", 1.0, 20.0, default=5.0)

    metrics_panel.separator()

    # Progress estimate
    progress = min(1.0, state["iter"] / 5000.0)
    metrics_panel.progress(progress,
                           overlay=f"{progress * 100:.0f}%")

    metrics_panel.separator()
    metrics_panel.text_wrapped(
        "Three-panel comparison: GT image, live prediction, and "
        "error heatmap (turbo colormap). Adjust Error Gain to "
        "amplify subtle differences."
    )


view.run()
