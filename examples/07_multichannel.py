"""
07 - Multi-Channel Viewer
=========================
Ray-sphere intersection on the GPU, displaying RGB, depth, normal,
and alpha simultaneously in four panels — the neural rendering workflow.

Key concepts
------------
- Multiple canvases    : Four tensors displayed at once, zero-copy
- Depth visualization  : Turbo colormap applied in pure PyTorch
- Normal visualization : World-space normals mapped to [0, 1] RGB
- Alpha mask           : Binary hit/miss as single-channel display
- Procedural rendering : Ray-sphere intersection in ~30 lines of PyTorch

Layout
------
Top-left     : RGB shading
Top-right    : Depth (turbo colormap)
Bottom-left  : World-space normals
Bottom-right : Alpha mask
Left sidebar : Sphere / light / camera controls

Why this matters for neural rendering
--------------------------------------
NeRF / 3DGS training produces multiple output channels per pixel:
RGB color, depth, normals, opacity.  Being able to see all four in
real time at 60 fps — with zero-copy, no matplotlib, no disk writes —
is exactly what you need during development.
"""

import math

import torch
import torch.nn.functional as F
import vultorch

if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"

# ── Image dimensions ──────────────────────────────────────────────
H, W = 256, 256

# ── View + panels ─────────────────────────────────────────────────
view = vultorch.View("07 - Multi-Channel Viewer", 512, 1024)
ctrl = view.panel("Controls", side="left", width=0.50)
rgb_panel = view.panel("RGB")
depth_panel = view.panel("Depth")
normal_panel = view.panel("Normal")
alpha_panel = view.panel("Alpha")

# ── Allocate display tensors (zero-copy) ──────────────────────────
rgb_tensor = vultorch.create_tensor(H, W, 4, device, name="rgb",
                                     window=view.window)
depth_tensor = vultorch.create_tensor(H, W, 4, device, name="depth",
                                       window=view.window)
normal_tensor = vultorch.create_tensor(H, W, 4, device, name="normal",
                                        window=view.window)
alpha_tensor = vultorch.create_tensor(H, W, 4, device, name="alpha",
                                       window=view.window)

rgb_panel.canvas("rgb").bind(rgb_tensor)
depth_panel.canvas("depth").bind(depth_tensor)
normal_panel.canvas("normal").bind(normal_tensor)
alpha_panel.canvas("alpha").bind(alpha_tensor)

# ── Turbo colormap (256 entries) ──────────────────────────────────
# Approximate Google turbo colormap as a 256×3 LUT on GPU
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


def _build_turbo_lut(n: int = 256) -> torch.Tensor:
    """Build a 256×3 turbo colormap LUT via linear interpolation."""
    keys = torch.tensor(_turbo_data, dtype=torch.float32)  # (32, 3)
    # Interpolate to n entries
    keys_t = keys.permute(1, 0).unsqueeze(0)  # (1, 3, 32)
    lut = F.interpolate(keys_t, size=n, mode="linear",
                        align_corners=True)  # (1, 3, 256)
    return lut.squeeze(0).permute(1, 0).contiguous().to(device)  # (256, 3)


TURBO_LUT = _build_turbo_lut()


def apply_turbo(values: torch.Tensor) -> torch.Tensor:
    """Map a [0, 1] float tensor (H, W) to (H, W, 3) turbo colors."""
    idx = (values.clamp(0, 1) * 255).long()  # (H, W)
    return TURBO_LUT[idx]                      # (H, W, 3)


# ── Precompute ray directions ────────────────────────────────────
ys = torch.linspace(1, -1, H, device=device)
xs = torch.linspace(-1, 1, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")

# ── State ────────────────────────────────────────────────────────
state = {
    "sphere_r": 0.6,
    "light_az": 0.5,
    "light_el": 0.8,
    "ambient": 0.1,
    "bg_r": 0.12,
    "bg_g": 0.12,
    "bg_b": 0.14,
}


def render_sphere():
    """Ray-sphere intersection + Lambertian shading, all on GPU."""
    r = state["sphere_r"]

    # Camera at (0, 0, -2), looking at origin, simple pinhole
    ray_o = torch.tensor([0.0, 0.0, -2.0], device=device)
    ray_d = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)  # (H, W, 3)
    ray_d = ray_d / ray_d.norm(dim=-1, keepdim=True)

    # Sphere at origin with radius r
    # |ray_o + t * ray_d|^2 = r^2
    oc = ray_o  # sphere center = origin
    a = (ray_d * ray_d).sum(-1)                           # ~1 (normalized)
    b = 2.0 * (ray_d * oc).sum(-1)
    c_val = (oc * oc).sum() - r * r

    disc = b * b - 4.0 * a * c_val
    hit = disc > 0

    # Nearest intersection
    sqrt_disc = torch.sqrt(disc.clamp(min=0))
    t = (-b - sqrt_disc) / (2.0 * a)
    t = t.clamp(min=0)

    # Hit point and normal
    point = ray_o + t.unsqueeze(-1) * ray_d    # (H, W, 3)
    normal = point / (point.norm(dim=-1, keepdim=True) + 1e-8)

    # Light direction (from azimuth / elevation)
    az = state["light_az"]
    el = state["light_el"]
    light_dir = torch.tensor([
        math.cos(el) * math.sin(az),
        math.sin(el),
        math.cos(el) * math.cos(az),
    ], device=device)
    light_dir = light_dir / light_dir.norm()

    # Lambertian shading
    ndotl = (normal * light_dir).sum(-1).clamp(min=0)
    shade = state["ambient"] + (1.0 - state["ambient"]) * ndotl

    # Depth (normalize to [0, 1] for display)
    depth_raw = t * hit.float()
    d_min = depth_raw[hit].min() if hit.any() else torch.tensor(0.0)
    d_max = depth_raw[hit].max() if hit.any() else torch.tensor(1.0)
    depth_norm = ((depth_raw - d_min) / (d_max - d_min + 1e-8)).clamp(0, 1)
    depth_norm = depth_norm * hit.float()

    # ── Write to display tensors ───────────────────────────────
    bg = torch.tensor([state["bg_r"], state["bg_g"], state["bg_b"]],
                      device=device)

    # RGB: shade * white (sphere) + background
    rgb_color = shade.unsqueeze(-1) * torch.ones(1, 1, 3, device=device)
    rgb_result = torch.where(hit.unsqueeze(-1), rgb_color, bg)
    rgb_tensor[:, :, :3] = rgb_result
    rgb_tensor[:, :, 3] = 1.0

    # Depth: turbo colormap
    depth_color = apply_turbo(depth_norm)
    depth_color = torch.where(hit.unsqueeze(-1), depth_color, bg)
    depth_tensor[:, :, :3] = depth_color
    depth_tensor[:, :, 3] = 1.0

    # Normal: map [-1,1] → [0,1]
    normal_color = (normal * 0.5 + 0.5) * hit.unsqueeze(-1).float()
    normal_color = torch.where(hit.unsqueeze(-1), normal_color, bg)
    normal_tensor[:, :, :3] = normal_color
    normal_tensor[:, :, 3] = 1.0

    # Alpha: white where hit, dark where miss
    alpha_val = hit.float()
    alpha_tensor[:, :, 0] = alpha_val
    alpha_tensor[:, :, 1] = alpha_val
    alpha_tensor[:, :, 2] = alpha_val
    alpha_tensor[:, :, 3] = 1.0


# ── Controls ──────────────────────────────────────────────────────
@ctrl.on_frame
def draw_controls():
    ctrl.text(f"FPS: {view.fps:.1f}")
    ctrl.text(f"Resolution: {W}×{H}")
    ctrl.separator()

    ctrl.text("Sphere")
    state["sphere_r"] = ctrl.slider("Radius", 0.1, 1.5, default=0.6)

    ctrl.separator()
    ctrl.text("Light Direction")
    state["light_az"] = ctrl.slider("Azimuth", -3.14, 3.14, default=0.5)
    state["light_el"] = ctrl.slider("Elevation", -1.5, 1.5, default=0.8)
    state["ambient"] = ctrl.slider("Ambient", 0.0, 1.0, default=0.1)

    ctrl.separator()
    bg = ctrl.color_picker("Background", default=(0.12, 0.12, 0.14))
    state["bg_r"], state["bg_g"], state["bg_b"] = bg

    ctrl.separator()
    ctrl.text_wrapped(
        "Ray-sphere intersection computed on the GPU. "
        "Four channels (RGB, depth, normal, alpha) displayed "
        "simultaneously via zero-copy tensors."
    )


# ── Main loop ─────────────────────────────────────────────────────
@view.on_frame
def update():
    render_sphere()


view.run()
