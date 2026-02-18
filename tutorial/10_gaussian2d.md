# 10 — 2D Gaussian Splatting

> **Example file:** `examples/10_gaussian2d.py`

3D Gaussian Splatting (3DGS) took neural rendering by storm. But the
core idea is surprisingly simple: scatter a bunch of colored blobs,
alpha-composite them together, compare to a target image, and backprop.

This example strips 3DGS down to its 2D essence. No camera projection,
no spherical harmonics, no tile-based rasterizer. Just N learnable
2D Gaussians — each with position, scale, rotation, color, and opacity —
rendered via differentiable alpha compositing and optimized with plain
PyTorch autograd.

Watch the Gaussians wander across the canvas, scale up, rotate, change
color, and gradually reassemble the target image. That's the algorithm
that renders photorealistic scenes at 100+ fps.

## New friends

| New thing | What it does | Why it matters |
|-----------|-------------|----------------|
| `nn.Parameter` for Gaussians | Position, log-scale, rotation, raw color, raw opacity | Every Gaussian attribute is learnable — autograd handles the rest |
| Inverse covariance | $\Sigma^{-1}$ from scale + rotation | The math that gives each Gaussian its elliptical shape |
| Alpha compositing | $C = \sum_i \alpha_i T_i c_i$ where $T_i = \prod_{j<i}(1-\alpha_j)$ | The standard front-to-back blending formula from volume rendering |
| `torch.cumprod` | Cumulative product for transmittance | Efficient parallel computation of $T_i$ for all Gaussians |
| Gaussian center overlay | Red dots at each mean position | See where the Gaussians are, not just what they render |

## What we're building

N random 2D Gaussians, each defined by 5 properties:

1. **Position** (μ) — where on the canvas, in [0, 1]²
2. **Scale** (σ) — how wide in x and y (stored as log-scale for positivity)
3. **Rotation** (θ) — orientation angle in radians
4. **Color** (c) — RGB via sigmoid
5. **Opacity** (α) — how opaque, via sigmoid

Each frame: render all Gaussians → MSE loss vs target → backward → step.
The Gaussians gradually converge to reproduce the target image.

## Full code

```python
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import vultorch

device = "cuda"

# Load target
img_path = Path(__file__).resolve().parents[1] / "docs" / "images" / "pytorch_logo.png"
gt = vultorch.imread(img_path, channels=3, size=(128, 128), device=device)
H, W = gt.shape[0], gt.shape[1]

# Pixel coordinates in [0, 1]
ys = torch.linspace(0, 1, H, device=device)
xs = torch.linspace(0, 1, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")
pixel_coords = torch.stack([xx, yy], dim=-1)


class GaussianModel2D(nn.Module):
    def __init__(self, n_gaussians=200):
        super().__init__()
        self.n = n_gaussians
        self.means = nn.Parameter(torch.rand(n_gaussians, 2, device=device))
        self.log_scales = nn.Parameter(
            torch.full((n_gaussians, 2), -3.0, device=device))
        self.rotations = nn.Parameter(
            torch.zeros(n_gaussians, device=device))
        self.raw_colors = nn.Parameter(
            torch.randn(n_gaussians, 3, device=device))
        self.raw_opacities = nn.Parameter(
            torch.zeros(n_gaussians, device=device))

    def forward(self, coords):
        means = self.means
        scales = self.log_scales.exp()
        colors = torch.sigmoid(self.raw_colors)
        opacities = torch.sigmoid(self.raw_opacities)

        cos_r = torch.cos(self.rotations)
        sin_r = torch.sin(self.rotations)
        sx2 = scales[:, 0] ** 2
        sy2 = scales[:, 1] ** 2

        # Inverse covariance elements
        a = cos_r**2 / (2*sx2) + sin_r**2 / (2*sy2)
        b = -sin_r*cos_r / (2*sx2) + sin_r*cos_r / (2*sy2)
        c = sin_r**2 / (2*sx2) + cos_r**2 / (2*sy2)

        dx = coords[:,:,0].unsqueeze(-1) - means[:,0]
        dy = coords[:,:,1].unsqueeze(-1) - means[:,1]

        exponent = -(a*dx*dx + 2*b*dx*dy + c*dy*dy)
        alpha = opacities * torch.exp(exponent)
        alpha = alpha.clamp(0, 0.99)

        # Front-to-back alpha compositing
        T = torch.ones(H, W, 1, device=device)
        transmittance = torch.cumprod(
            torch.cat([T, (1-alpha)[:,:,:-1]], dim=-1), dim=-1)
        weights = alpha * transmittance

        rendered = (weights.unsqueeze(-1) *
                    colors.unsqueeze(0).unsqueeze(0)).sum(dim=2)
        bg = (transmittance[:,:,-1] * (1 - alpha[:,:,-1])).unsqueeze(-1)
        rendered = rendered + bg  # (H,W,1) broadcasts to (H,W,3)

        return rendered.clamp(0, 1)


model = GaussianModel2D(200).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

# View + panels
view = vultorch.View("10 - 2D Gaussian Splatting", 1100, 900)
ctrl = view.panel("Controls", side="left", width=0.28)
gt_panel = view.panel("Ground Truth")
render_panel = view.panel("Rendered")
metrics_panel = view.panel("Metrics")

# ... setup canvases, state, callbacks ...

try:
    while view.step():
        for _ in range(steps_per_frame):
            rendered = model(pixel_coords)
            loss = F.mse_loss(rendered, gt)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            render_rgba[:, :, :3] = model(pixel_coords).clamp_(0, 1)

        view.end_step()
finally:
    view.close()
```

*(Abridged — see `examples/10_gaussian2d.py` for the complete code.)*

## What just happened?

### The 2D Gaussian formula

Each Gaussian is an elliptical blob. Its shape is defined by:

$$G(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

where $\Sigma = R \begin{pmatrix} \sigma_x^2 & 0 \\ 0 & \sigma_y^2 \end{pmatrix} R^T$ and $R$ is the rotation matrix.

In code, we compute the inverse covariance elements directly:

```python
a = cos_r**2 / (2*sx2) + sin_r**2 / (2*sy2)
b = -sin_r*cos_r / (2*sx2) + sin_r*cos_r / (2*sy2)
c = sin_r**2 / (2*sx2) + cos_r**2 / (2*sy2)

exponent = -(a*dx*dx + 2*b*dx*dy + c*dy*dy)
```

The `log_scales` trick ensures scales are always positive after `exp()`.
The rotation angle is unconstrained — any real number works.

### Alpha compositing — the volume rendering equation

The color at each pixel is a weighted sum of all Gaussians:

$$C = \sum_{i=1}^{N} \alpha_i T_i c_i + T_N \cdot c_{bg}$$

where transmittance $T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$ tracks how
much light reaches Gaussian $i$ through all the ones in front of it.

```python
transmittance = torch.cumprod(
    torch.cat([T, (1-alpha)[:,:,:-1]], dim=-1), dim=-1)
weights = alpha * transmittance
rendered = (weights.unsqueeze(-1) * colors).sum(dim=2)
```

This is exactly the same equation used in NeRF's volume rendering and
3DGS's rasterization. The only difference: NeRF samples along rays,
3DGS splats projected Gaussians, and we splat 2D Gaussians directly.

### Why it works

Each Gaussian has 9 learnable parameters (2 position + 2 scale +
1 rotation + 3 color + 1 opacity). With 200 Gaussians that's 1800
parameters — tiny compared to any neural network, but enough to
approximate a 128×128 image.

The key insight: Gaussians are a natural basis for image reconstruction.
They can overlap, have soft edges, and their gradients are smooth.
Autograd handles the chain rule through the compositing equation.

### Gaussian center overlay

```python
means = model.means.detach().clamp(0, 1)
px = (means[:, 0] * (W - 1)).long().clamp(0, W - 1)
py = (means[:, 1] * (H - 1)).long().clamp(0, H - 1)
render_rgba[py, px, 0] = 1.0  # red dot
```

This shows where each Gaussian's center currently sits. Early in training
you see random red dots. As training progresses, they cluster where the
image has detail — more Gaussians where there's more to reconstruct.

## Key takeaways

1. **2D Gaussian Splatting is 3DGS without the camera** — same alpha
   compositing, same learnable ellipses, same optimization. Just simpler.

2. **`nn.Parameter` for everything** — position, scale, rotation, color,
   opacity. PyTorch autograd differentiates through all of it.

3. **Log-scale trick** — store scale as `log_scales`, use `exp()` to
   guarantee positivity without clamping.

4. **`torch.cumprod` for transmittance** — efficient parallel computation
   of the front-to-back compositing weights.

5. **From 2D to 3D** — to get 3DGS, add: 3D positions, camera projection,
   spherical harmonics for view-dependent color, tile-based sorting.
   The core compositing math stays the same.

!!! tip
    Try increasing the number of Gaussians (500 or 1000) and watch how
    the optimization dynamics change. More Gaussians = finer detail, but
    slower convergence per Gaussian.

!!! note
    This is a pedagogical implementation — real 3DGS uses CUDA kernels
    for tile-based rasterization that are 100× faster. But the math is
    identical.
