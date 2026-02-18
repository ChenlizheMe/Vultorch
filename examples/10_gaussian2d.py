"""
10 - 2D Gaussian Splatting
==========================
Differentiable 2D Gaussian rendering — the core algorithm behind 3DGS,
distilled to a pure-PyTorch, single-file example with live visualization.

Initialize N random 2D Gaussians (position, scale, rotation, color, opacity),
alpha-composite them onto a canvas, compare to a target image via MSE,
and optimize with backprop. Watch the Gaussians converge in real time.

Key concepts
------------
- Differentiable splatting : 2D Gaussian → alpha composite → loss → backward
- nn.Parameter             : Position, scale, rotation, color, opacity as learnable
- Alpha compositing        : Front-to-back blending (the 3DGS formula)
- Live Gaussian overlay    : See where each Gaussian sits and how it adapts
- step() / end_step()      : Training loop integration

Layout
------
Left sidebar : Controls (num gaussians, LR, reset, display options)
Top-left     : Target image
Top-right    : Rendered result (live)
Bottom       : Metrics (loss, PSNR, curves)
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
gt = vultorch.imread(img_path, channels=3, size=(128, 128), device=device)
H, W = gt.shape[0], gt.shape[1]

# ── Pixel coordinate grid ────────────────────────────────────────
ys = torch.linspace(0, 1, H, device=device)
xs = torch.linspace(0, 1, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")
pixel_coords = torch.stack([xx, yy], dim=-1)  # (H, W, 2) in [0, 1]


# ── 2D Gaussian Splatting model ───────────────────────────────────
class GaussianModel2D(nn.Module):
    """N learnable 2D Gaussians with position, scale, rotation, color, opacity."""

    def __init__(self, n_gaussians=200):
        super().__init__()
        self.n = n_gaussians
        # Positions in [0, 1] range
        self.means = nn.Parameter(torch.rand(n_gaussians, 2, device=device))
        # Log-scale (so scale is always positive after exp)
        self.log_scales = nn.Parameter(
            torch.full((n_gaussians, 2), -3.0, device=device))
        # Rotation angle (radians)
        self.rotations = nn.Parameter(
            torch.zeros(n_gaussians, device=device))
        # Color in [0, 1] via sigmoid
        self.raw_colors = nn.Parameter(
            torch.randn(n_gaussians, 3, device=device))
        # Opacity in [0, 1] via sigmoid
        self.raw_opacities = nn.Parameter(
            torch.zeros(n_gaussians, device=device))

    def forward(self, coords):
        """Render all Gaussians onto the coordinate grid via alpha compositing.

        Args:
            coords: (H, W, 2) pixel coordinates in [0, 1]

        Returns:
            (H, W, 3) rendered image
        """
        H, W = coords.shape[0], coords.shape[1]

        means = self.means                                   # (N, 2)
        scales = self.log_scales.exp()                       # (N, 2)
        colors = torch.sigmoid(self.raw_colors)              # (N, 3)
        opacities = torch.sigmoid(self.raw_opacities)        # (N,)

        # Build 2×2 covariance from scale + rotation
        cos_r = torch.cos(self.rotations)                    # (N,)
        sin_r = torch.sin(self.rotations)                    # (N,)

        # Rotation matrix R = [[cos, -sin], [sin, cos]]
        # Covariance = R @ diag(s^2) @ R^T
        sx2 = scales[:, 0] ** 2                              # (N,)
        sy2 = scales[:, 1] ** 2                              # (N,)

        # Elements of Sigma^{-1} (inverse covariance)
        a = cos_r ** 2 / (2 * sx2) + sin_r ** 2 / (2 * sy2)
        b = -sin_r * cos_r / (2 * sx2) + sin_r * cos_r / (2 * sy2)
        c = sin_r ** 2 / (2 * sx2) + cos_r ** 2 / (2 * sy2)

        # Evaluate Gaussian: exp(-0.5 * (dx, dy) @ Sigma^{-1} @ (dx, dy)^T)
        # coords: (H, W, 2), means: (N, 2)
        dx = coords[:, :, 0].unsqueeze(-1) - means[:, 0]    # (H, W, N)
        dy = coords[:, :, 1].unsqueeze(-1) - means[:, 1]    # (H, W, N)

        exponent = -(a * dx * dx + 2 * b * dx * dy + c * dy * dy)
        alpha = opacities.unsqueeze(0).unsqueeze(0) * torch.exp(exponent)
        alpha = alpha.clamp(0, 0.99)                         # (H, W, N)

        # Front-to-back alpha compositing
        # Sort by... well, in 2D we just composite in parameter order.
        # T_i = prod(1 - alpha_j, j < i)
        one_minus_alpha = 1.0 - alpha                        # (H, W, N)
        # Cumulative transmittance (exclusive prefix product)
        # T_i = cumprod of (1 - alpha) shifted by 1
        T = torch.ones(H, W, 1, device=device)
        transmittance = torch.cumprod(
            torch.cat([T, one_minus_alpha[:, :, :-1]], dim=-1), dim=-1)

        # Weight for each Gaussian: alpha_i * T_i
        weights = alpha * transmittance                      # (H, W, N)

        # Final color: sum of weighted colors
        colors_expanded = colors.unsqueeze(0).unsqueeze(0)   # (1, 1, N, 3)
        rendered = (weights.unsqueeze(-1) * colors_expanded).sum(dim=2)

        # Background: white where transmittance remains
        bg = (transmittance[:, :, -1] * one_minus_alpha[:, :, -1]).unsqueeze(-1)
        rendered = rendered + bg  # (H, W, 1) broadcasts to (H, W, 3)

        return rendered.clamp(0, 1)


def compute_psnr(mse):
    if mse <= 0:
        return 50.0
    return -10.0 * math.log10(mse)


# ── State ─────────────────────────────────────────────────────────
N_GAUSSIANS = 200

state = {
    "n_gaussians": N_GAUSSIANS,
    "lr": 5e-3,
    "iter": 0,
    "loss": 1.0,
    "psnr": 0.0,
    "steps_per_frame": 4,
    "loss_history": [],
    "psnr_history": [],
    "needs_reset": False,
    "show_centers": True,
}

model = GaussianModel2D(N_GAUSSIANS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=state["lr"])

# ── View + panels ─────────────────────────────────────────────────
view = vultorch.View("10 - 2D Gaussian Splatting", 1100, 900)
ctrl = view.panel("Controls", side="left", width=0.28)
render_panel = view.panel("Rendered")
metrics_panel = view.panel("Metrics")

# Display tensor
render_rgba = vultorch.create_tensor(H, W, 4, device, name="render",
                                      window=view.window)
render_rgba[:, :, 3] = 1.0
render_panel.canvas("render").bind(render_rgba)


# ── Controls ──────────────────────────────────────────────────────
@ctrl.on_frame
def draw_controls():
    ctrl.text(f"FPS: {view.fps:.1f}")
    ctrl.text(f"Iteration: {state['iter']}")
    ctrl.text(f"Gaussians: {state['n_gaussians']}")
    ctrl.separator()

    # Learning rate
    log_lr = ctrl.slider("log10(LR)", -5.0, -1.0, default=-2.3)
    state["lr"] = 10.0 ** log_lr
    ctrl.text(f"  LR = {state['lr']:.2e}")
    for pg in optimizer.param_groups:
        pg["lr"] = state["lr"]

    ctrl.separator()

    state["steps_per_frame"] = ctrl.slider_int("Steps/Frame", 1, 16,
                                                default=4)
    state["show_centers"] = ctrl.checkbox("Show Centers", default=True)

    ctrl.separator()

    if ctrl.button("Reset (200)", width=120):
        state["n_gaussians"] = 200
        state["needs_reset"] = True
    if ctrl.button("Reset (500)", width=120):
        state["n_gaussians"] = 500
        state["needs_reset"] = True
    if ctrl.button("Reset (1000)", width=120):
        state["n_gaussians"] = 1000
        state["needs_reset"] = True

    ctrl.separator()
    ctrl.text_wrapped(
        "2D Gaussian Splatting: each Gaussian has learnable position, "
        "scale, rotation, color, and opacity. Alpha composited onto "
        "the canvas and optimized via MSE against the target image. "
        "This is exactly how 3D Gaussian Splatting works, but in 2D."
    )


# ── Metrics ───────────────────────────────────────────────────────
@metrics_panel.on_frame
def draw_metrics():
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


# ── Training loop ────────────────────────────────────────────────
try:
    while view.step():
        # Handle reset
        if state["needs_reset"]:
            model = GaussianModel2D(state["n_gaussians"]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=state["lr"])
            state["iter"] = 0
            state["loss"] = 1.0
            state["psnr"] = 0.0
            state["loss_history"].clear()
            state["psnr_history"].clear()
            state["needs_reset"] = False

        # Training steps
        for _ in range(state["steps_per_frame"]):
            optimizer.zero_grad(set_to_none=True)
            rendered = model(pixel_coords)
            loss = F.mse_loss(rendered, gt)
            loss.backward()
            optimizer.step()

            state["iter"] += 1
            state["loss"] = loss.item()

        # Update display
        with torch.no_grad():
            rendered = model(pixel_coords).clamp_(0, 1)
            render_rgba[:, :, :3] = rendered

            # Draw Gaussian centers as bright dots
            if state["show_centers"]:
                means = model.means.detach().clamp(0, 1)
                opacities = torch.sigmoid(model.raw_opacities.detach())
                # Only show Gaussians with opacity > 0.1
                visible = opacities > 0.1
                vis_means = means[visible]
                if vis_means.numel() > 0:
                    px = (vis_means[:, 0] * (W - 1)).long().clamp(0, W - 1)
                    py = (vis_means[:, 1] * (H - 1)).long().clamp(0, H - 1)
                    render_rgba[py, px, 0] = 1.0
                    render_rgba[py, px, 1] = 0.0
                    render_rgba[py, px, 2] = 0.0

            # PSNR
            state["psnr"] = compute_psnr(state["loss"])

        # History
        state["loss_history"].append(state["loss"])
        state["psnr_history"].append(state["psnr"])
        if len(state["loss_history"]) > 300:
            state["loss_history"] = state["loss_history"][-300:]
            state["psnr_history"] = state["psnr_history"][-300:]

        view.end_step()
finally:
    view.close()
