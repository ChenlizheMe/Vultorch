# 11 — 3D Surface Inspector

> **Example file:** `examples/11_3d_inspector.py`

You've rendered a depth map, a normal map, or a texture from your NeRF.
Now you want to see it from a different angle, catch the light on it,
check for artifacts at grazing angles. In matplotlib you'd be wrestling
with `plot_surface()` and `set_azim()`. Here, you just drag the mouse.

SceneView is Vultorch's 3D viewer widget. It takes any tensor, maps it
onto a plane in 3D space, adds Blinn-Phong lighting, and lets you orbit
around it with mouse drag. MSAA anti-aliasing, adjustable FOV, light
direction, shininess — all in real time.

## New friends

| New thing | What it does | Why it matters |
|-----------|-------------|----------------|
| `SceneView` | 3D plane viewer with orbit camera | Inspect textures/outputs in 3D with mouse interaction |
| `Camera` | azimuth, elevation, distance, fov | Orbit around the scene; dragged by mouse or set programmatically |
| `Light` | direction, intensity, ambient, specular, shininess | Blinn-Phong shading with full control |
| `.set_tensor()` | Upload any tensor to the 3D scene | RGB, RGBA, grayscale — auto-expanded to RGBA |
| `.render()` | Draw the scene as an ImGui image | Handles mouse drag, camera sync, resize, all automatically |
| `.msaa` | Multi-sample anti-aliasing (1/2/4/8) | Smooth edges at different quality/performance tradeoffs |
| `.background` | Background color tuple | Sets the clear color behind the 3D plane |

## What we're building

Four procedural textures generated on the GPU — checkerboard, radial
gradient, sine pattern, normal map — displayed one at a time on a 3D
plane. Mouse drag to orbit, with a control sidebar for camera, lighting,
MSAA, and background color.

## Full code

```python
import math

import torch
import vultorch
from vultorch import ui

device = "cuda"
H, W = 256, 256

ys = torch.linspace(-1, 1, H, device=device)
xs = torch.linspace(-1, 1, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")


def make_checkerboard(freq=8.0):
    check = ((xx * freq).floor() + (yy * freq).floor()) % 2
    return torch.stack([check*0.9+0.1, check*0.1+0.2, check*0.3+0.4], dim=-1)


def make_radial_gradient():
    r = (xx**2 + yy**2).sqrt()
    val = (1.0 - r).clamp(0, 1)
    return torch.stack([val, val*0.5, val*0.2], dim=-1)


def make_sine_pattern(freq=6.0):
    s1 = (torch.sin(xx * freq * math.pi) * 0.5 + 0.5)
    s2 = (torch.sin(yy * freq * math.pi) * 0.5 + 0.5)
    val = s1 * s2
    return torch.stack([val*0.2+0.1, val*0.8+0.1, val*0.5+0.3], dim=-1)


def make_normal_map():
    nx = xx * 0.5 + 0.5
    ny = yy * 0.5 + 0.5
    nz = (1.0 - xx**2 - yy**2).clamp(0, 1).sqrt() * 0.5 + 0.5
    return torch.stack([nx, ny, nz], dim=-1)


TEXTURE_NAMES = ["Checkerboard", "Radial Gradient", "Sine Pattern", "Normal Map"]
TEXTURE_FNS = [make_checkerboard, make_radial_gradient,
               make_sine_pattern, make_normal_map]

# View + panels
view = vultorch.View("11 - 3D Surface Inspector", 1200, 800)
ctrl = view.panel("Controls", side="left", width=0.28)
scene_panel = view.panel("3D View")

# SceneView lives inside the panel
scene = vultorch.SceneView("Inspector", 800, 600, msaa=4)
current_texture = make_checkerboard()


@ctrl.on_frame
def draw_controls():
    state["texture_idx"] = ctrl.combo("Texture", TEXTURE_NAMES)
    state["fov"] = ctrl.slider("FOV", 10.0, 120.0, default=45.0)
    state["distance"] = ctrl.slider("Distance", 1.0, 10.0, default=3.0)
    state["auto_rotate"] = ctrl.checkbox("Auto Rotate")

    # Light controls
    state["light_az"] = ctrl.slider("Light Az", -3.14, 3.14)
    state["light_el"] = ctrl.slider("Light El", -3.14, 3.14)
    state["ambient"] = ctrl.slider("Ambient", 0.0, 1.0, default=0.15)
    state["specular"] = ctrl.slider("Specular", 0.0, 2.0, default=0.5)
    state["shininess"] = ctrl.slider("Shininess", 1.0, 128.0, default=32.0)

    state["msaa_idx"] = ctrl.combo("MSAA", ["1", "2", "4", "8"])
    state["bg_color"] = ctrl.color_picker("Background")

    if ctrl.button("Reset Camera"):
        scene.camera.reset()


@scene_panel.on_frame
def draw_scene():
    # Apply settings to camera, light, background
    scene.camera.fov = state["fov"]
    scene.camera.distance = state["distance"]
    if state["auto_rotate"]:
        scene.camera.azimuth += 0.02

    scene.light.direction = (cos(el)*sin(az), sin(el), cos(el)*cos(az))
    scene.light.ambient = state["ambient"]
    scene.light.specular = state["specular"]
    scene.light.shininess = state["shininess"]
    scene.msaa = msaa_val
    scene.background = state["bg_color"]

    scene.set_tensor(current_texture)
    scene.render()  # draws the 3D view right here


view.run()
```

*(Abridged — see `examples/11_3d_inspector.py` for the complete code.)*

## What just happened?

### SceneView — 3D in a panel

SceneView is a self-contained 3D viewer widget. You create it once,
then call `set_tensor()` and `render()` each frame:

```python
scene = vultorch.SceneView("Inspector", 800, 600, msaa=4)

# Inside a panel callback:
scene.set_tensor(my_tensor)  # upload the texture
scene.render()               # render + display + handle mouse
```

`render()` does everything: pushes camera/light settings to the GPU,
renders the scene offscreen, displays it as an ImGui image, handles
mouse drag for orbit/pan/zoom, and pulls the camera state back so
Python sees the updated azimuth/elevation.

### Camera — orbit with math.mouse

The camera is defined by 5 values:

| Property | Default | What it controls |
|----------|---------|-----------------|
| `azimuth` | 0.0 | Horizontal rotation (radians) |
| `elevation` | 0.6 | Vertical rotation (radians) |
| `distance` | 3.0 | Distance from target |
| `target` | (0,0,0) | Look-at point |
| `fov` | 45.0 | Field of view (degrees) |

Left-drag rotates (azimuth + elevation), right-drag pans (target),
middle-drag/scroll zooms (distance). All built-in — no code needed.

You can also set values programmatically:

```python
scene.camera.fov = 90.0       # wide-angle lens
scene.camera.distance = 5.0   # far away
scene.camera.azimuth += 0.02  # auto-rotate
```

### Light — Blinn-Phong shading

The light is a directional light source with Blinn-Phong shading:

```python
scene.light.direction = (0.3, -1.0, 0.5)  # direction vector
scene.light.intensity = 1.0                # overall brightness
scene.light.ambient = 0.15                 # fill light
scene.light.specular = 0.5                 # highlight strength
scene.light.shininess = 32.0               # highlight sharpness
```

Low ambient + high specular = dramatic, contrasty look. High ambient
+ low specular = flat, evenly-lit look. Adjusting these interactively
helps you spot surface artifacts that only show up at certain lighting
angles.

### MSAA — anti-aliasing quality

MSAA (Multi-Sample Anti-Aliasing) smooths jagged edges:

| MSAA | Samples/pixel | Quality | Performance |
|------|--------------|---------|-------------|
| 1 | 1 | Aliased | Fastest |
| 2 | 2 | Slightly smooth | Fast |
| 4 | 4 | Smooth | Moderate |
| 8 | 8 | Very smooth | Slowest |

```python
scene.msaa = 4  # good default
```

For most visualization work, 4× MSAA is the sweet spot. Drop to 1
if you need maximum frame rate, or go to 8 for screenshots.

### Procedural textures on GPU

All four textures are pure PyTorch tensor operations — no CPU, no PIL,
no file I/O:

```python
def make_checkerboard(freq=8.0):
    check = ((xx * freq).floor() + (yy * freq).floor()) % 2
    return torch.stack([check*0.9+0.1, check*0.1+0.2, check*0.3+0.4], dim=-1)
```

This is important because in a real neural rendering pipeline, the tensor
you'd display here would be your NeRF's rendered output, your 3DGS's
depth map, or your diffusion model's generated texture. The SceneView
doesn't care where the tensor came from — it just displays it.

## Key takeaways

1. **SceneView** = 3D plane viewer. Upload a tensor, render, orbit with mouse.
   That's it.

2. **Camera** has azimuth/elevation/distance/fov — set by mouse drag or
   by Python code. `camera.reset()` goes back to defaults.

3. **Light** is Blinn-Phong: direction, intensity, ambient, specular, shininess.
   Interactive lighting reveals surface artifacts.

4. **MSAA** goes from 1 (fast, aliased) to 8 (slow, smooth). Use 4 for
   daily work.

5. **Any tensor works** — RGB, RGBA, grayscale. SceneView auto-expands
   to RGBA internally.

!!! tip
    Use auto-rotate (`scene.camera.azimuth += 0.02`) to quickly scan
    a surface for artifacts — problems that are invisible from the
    default angle become obvious when the surface rotates.

!!! note
    SceneView renders a flat plane. For actual 3D geometry (meshes,
    point clouds), you'd need to extend the C++ renderer. But for
    inspecting per-pixel outputs like depth maps, normal maps, and
    textures, a lit 3D plane with orbit camera is exactly what you need.
