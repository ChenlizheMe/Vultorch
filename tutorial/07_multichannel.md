# 07 — Multi-Channel Viewer

> **Example file:** `examples/07_multichannel.py`

If you're doing neural rendering — NeRF, 3D Gaussian Splatting,
whatever the next acronym is — your model doesn't just produce a
pretty picture. It produces RGB *and* depth *and* normals *and* alpha.
Every. Single. Pixel.

During development you need to see all of these at once. The standard
workflow: save four PNGs, open them in four matplotlib windows, squint
at them side by side, realize the depth map is upside-down, save again,
reopen, repeat until you question your career choices.

This chapter replaces all of that with four zero-copy panels updating
at 60 fps.

## New friends

| New thing | What it does | Why it matters |
|-----------|-------------|----------------|
| Four `create_tensor` calls | Four independent GPU-shared textures in one window | Each output channel gets its own live display |
| Turbo colormap | Maps a scalar `[0, 1]` tensor to a colored `(H, W, 3)` image | Depth and other scalar fields are invisible in grayscale; turbo makes structure obvious |
| Normal → RGB mapping | `normal * 0.5 + 0.5` converts `[-1, 1]` normals to `[0, 1]` colors | The standard convention: X→red, Y→green, Z→blue |
| Ray-sphere intersection | All-GPU procedural rendering in ~30 lines of PyTorch | Demonstrates that any GPU computation can feed into Vultorch |

## What we're building

A procedural ray-sphere renderer with four live outputs and a control sidebar.
The entire computation — rays, intersections, shading, colormap — runs on the
GPU. All four display tensors are zero-copy, so nothing is ever copied to CPU.

## Full code

```python
import math

import torch
import torch.nn.functional as F
import vultorch

if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"

H, W = 256, 256

view = vultorch.View("07 - Multi-Channel Viewer", 512, 1024)
ctrl = view.panel("Controls", side="left", width=0.20)
rgb_panel = view.panel("RGB")
depth_panel = view.panel("Depth")
normal_panel = view.panel("Normal")
alpha_panel = view.panel("Alpha")

# Four zero-copy display tensors
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

# --- Turbo colormap LUT (256 entries, built once) ---
_turbo_data = [
    (0.18995, 0.07176, 0.23217), (0.22500, 0.16354, 0.45096),
    # ... (32 key colors, interpolated to 256)
]
TURBO_LUT = ...  # see full source for the complete LUT build

def apply_turbo(values):
    """Map [0,1] float tensor (H,W) → (H,W,3) turbo colors."""
    idx = (values.clamp(0, 1) * 255).long()
    return TURBO_LUT[idx]

# Precompute ray directions
ys = torch.linspace(1, -1, H, device=device)
xs = torch.linspace(-1, 1, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")

state = {"sphere_r": 0.6, "light_az": 0.5, "light_el": 0.8,
         "ambient": 0.1, "bg_r": 0.12, "bg_g": 0.12, "bg_b": 0.14}


def render_sphere():
    r = state["sphere_r"]
    ray_o = torch.tensor([0.0, 0.0, -2.0], device=device)
    ray_d = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    ray_d = ray_d / ray_d.norm(dim=-1, keepdim=True)

    # Quadratic formula for ray-sphere intersection
    b = 2.0 * (ray_d * ray_o).sum(-1)
    c_val = (ray_o * ray_o).sum() - r * r
    disc = b * b - 4.0 * c_val
    hit = disc > 0

    t = (-b - torch.sqrt(disc.clamp(min=0))) / 2.0
    t = t.clamp(min=0)
    point = ray_o + t.unsqueeze(-1) * ray_d
    normal = point / (point.norm(dim=-1, keepdim=True) + 1e-8)

    # Lambertian shading
    az, el = state["light_az"], state["light_el"]
    light_dir = torch.tensor([math.cos(el)*math.sin(az),
                               math.sin(el),
                               math.cos(el)*math.cos(az)], device=device)
    light_dir = light_dir / light_dir.norm()
    shade = state["ambient"] + (1 - state["ambient"]) * \
            (normal * light_dir).sum(-1).clamp(min=0)

    bg = torch.tensor([state["bg_r"], state["bg_g"], state["bg_b"]],
                      device=device)

    # RGB
    rgb = torch.where(hit.unsqueeze(-1),
                      shade.unsqueeze(-1) * torch.ones(1,1,3, device=device), bg)
    rgb_tensor[:,:,:3] = rgb;  rgb_tensor[:,:,3] = 1.0

    # Depth (turbo colormap)
    depth_raw = t * hit.float()
    d_min = depth_raw[hit].min() if hit.any() else torch.tensor(0.0)
    d_max = depth_raw[hit].max() if hit.any() else torch.tensor(1.0)
    depth_norm = ((depth_raw - d_min) / (d_max - d_min + 1e-8)).clamp(0,1)
    depth_color = torch.where(hit.unsqueeze(-1), apply_turbo(depth_norm), bg)
    depth_tensor[:,:,:3] = depth_color;  depth_tensor[:,:,3] = 1.0

    # Normals ([-1,1] → [0,1])
    nc = torch.where(hit.unsqueeze(-1), normal * 0.5 + 0.5, bg)
    normal_tensor[:,:,:3] = nc;  normal_tensor[:,:,3] = 1.0

    # Alpha
    a = hit.float()
    alpha_tensor[:,:,0] = a; alpha_tensor[:,:,1] = a
    alpha_tensor[:,:,2] = a; alpha_tensor[:,:,3] = 1.0


@ctrl.on_frame
def draw_controls():
    ctrl.text(f"FPS: {view.fps:.1f}")
    ctrl.separator()
    state["sphere_r"] = ctrl.slider("Radius", 0.1, 1.5, default=0.6)
    ctrl.separator()
    state["light_az"] = ctrl.slider("Light Az", -3.14, 3.14, default=0.5)
    state["light_el"] = ctrl.slider("Light El", -1.5, 1.5, default=0.8)
    state["ambient"]  = ctrl.slider("Ambient", 0.0, 1.0, default=0.1)
    ctrl.separator()
    bg = ctrl.color_picker("Background", default=(0.12, 0.12, 0.14))
    state["bg_r"], state["bg_g"], state["bg_b"] = bg


@view.on_frame
def update():
    render_sphere()


view.run()
```

*(The listing above is abridged — see `examples/07_multichannel.py` for the
complete turbo colormap LUT.)*

## What just happened?

### Four panels, four tensors, one window

```python
rgb_tensor    = vultorch.create_tensor(H, W, 4, device, name="rgb", ...)
depth_tensor  = vultorch.create_tensor(H, W, 4, device, name="depth", ...)
normal_tensor = vultorch.create_tensor(H, W, 4, device, name="normal", ...)
alpha_tensor  = vultorch.create_tensor(H, W, 4, device, name="alpha", ...)
```

Each call allocates a separate Vulkan-shared tensor. Each panel binds
one of them. All four update every frame with no CPU involvement — the
data path is CUDA → Vulkan → screen.

This is the workflow for neural rendering: your model forward-pass fills
four tensors, and the viewer shows all of them simultaneously.

### Turbo colormap — making depth visible

Raw depth values are floats in some arbitrary range. Displaying them
directly gives you a nearly-black image with invisible gradients. The
turbo colormap maps `[0, 1]` scalars to a perceptually uniform rainbow
so you can actually *see* the depth structure:

```python
def apply_turbo(values):
    idx = (values.clamp(0, 1) * 255).long()   # quantize to 256 bins
    return TURBO_LUT[idx]                       # lookup (H, W, 3)
```

This runs entirely on the GPU — no numpy, no matplotlib.

### Normal → RGB convention

The standard way to visualize surface normals: map each component from
`[-1, 1]` to `[0, 1]` and assign it to a color channel:

```python
normal_color = normal * 0.5 + 0.5   # X→R, Y→G, Z→B
```

A surface pointing right is red, up is green, towards the camera is blue.
Every neural rendering paper uses this convention, so you'll recognize
the visual immediately.

### Ray-sphere intersection

The entire renderer is ~30 lines of PyTorch. The key is the quadratic
formula for ray-sphere intersection:

$$t = \frac{-b - \sqrt{b^2 - 4ac}}{2a}$$

where $a = \|d\|^2$, $b = 2 \langle o, d \rangle$, $c = \|o\|^2 - r^2$.
This runs in parallel for all $H \times W$ rays in one GPU kernel call.
Replace this with your neural network's forward pass and you've got a
NeRF viewer.

## Key takeaways

1. **Multiple zero-copy tensors** — call `create_tensor` once per output
   channel, bind each to its own panel. All updates happen on the GPU.

2. **Turbo colormap** — a GPU LUT indexed by `(values * 255).long()`.
   Essential for depth, disparity, attention weights, loss heatmaps —
   any scalar field that would be invisible in grayscale.

3. **Normal → RGB** — `n * 0.5 + 0.5`. Three characters of code,
   universally understood in the graphics/vision community.

4. **Procedural → neural** — this example uses ray-sphere intersection
   as a stand-in. Replace `render_sphere()` with your model's forward
   pass and you have a live multi-channel neural rendering viewer.

5. **Zero-copy scalability** — four 256×256×4 textures updating every
   frame. The bottleneck is your computation, not the display pipeline.

!!! tip
    Drag the panel borders to rearrange the four views — put depth next
    to RGB for comparison, or stack normals on top of alpha. The layout
    is fully user-configurable at runtime.
