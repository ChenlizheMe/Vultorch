"""
11 - 3D Surface Inspector
=========================
View a texture on a 3D plane with orbit camera, Blinn-Phong lighting,
and MSAA — the SceneView widget in action.

Generate procedural textures on the GPU, map them onto a 3D plane,
and inspect the result from any angle with mouse-drag orbit controls.
Adjust lighting, camera, MSAA, and background in real time.

Key concepts
------------
- SceneView           : 3D plane viewer with orbit camera and lighting
- Camera              : azimuth, elevation, distance, fov (mouse drag)
- Light               : direction, intensity, ambient, specular, shininess
- MSAA                : Multi-sample anti-aliasing (1/2/4/8)
- Procedural textures : Generated on GPU, displayed instantly

Layout
------
Left sidebar : Camera, light, MSAA, background controls
Right        : 3D view panel with SceneView (mouse drag to orbit)
"""

import math

import torch
import torch.nn.functional as F
import vultorch
from vultorch import ui

if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"

# ── Procedural texture generation ─────────────────────────────────
H, W = 256, 256

ys = torch.linspace(-1, 1, H, device=device)
xs = torch.linspace(-1, 1, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")


def make_checkerboard(freq=8.0):
    """Classic checkerboard pattern."""
    check = ((xx * freq).floor() + (yy * freq).floor()) % 2
    rgb = torch.stack([check * 0.9 + 0.1,
                       check * 0.1 + 0.2,
                       check * 0.3 + 0.4], dim=-1)
    return rgb


def make_radial_gradient():
    """Radial gradient from center — looks like a heat spot."""
    r = (xx ** 2 + yy ** 2).sqrt()
    val = (1.0 - r).clamp(0, 1)
    # Map to warm colors
    rgb = torch.stack([val,
                       val * 0.5,
                       val * 0.2], dim=-1)
    return rgb


def make_sine_pattern(freq=6.0):
    """Interference pattern of two sine waves."""
    s1 = (torch.sin(xx * freq * math.pi) * 0.5 + 0.5)
    s2 = (torch.sin(yy * freq * math.pi) * 0.5 + 0.5)
    val = s1 * s2
    rgb = torch.stack([val * 0.2 + 0.1,
                       val * 0.8 + 0.1,
                       val * 0.5 + 0.3], dim=-1)
    return rgb


def make_normal_map():
    """Fake normal map — direction-dependent colors like a world-space normal output."""
    nx = xx * 0.5 + 0.5
    ny = yy * 0.5 + 0.5
    nz_sq = (1.0 - xx ** 2 - yy ** 2).clamp(0, 1)
    nz = nz_sq.sqrt() * 0.5 + 0.5
    rgb = torch.stack([nx, ny, nz], dim=-1)
    return rgb


TEXTURE_NAMES = ["Checkerboard", "Radial Gradient", "Sine Pattern", "Normal Map"]
TEXTURE_FNS = [make_checkerboard, make_radial_gradient,
               make_sine_pattern, make_normal_map]

MSAA_OPTIONS = ["1", "2", "4", "8"]

# ── State ─────────────────────────────────────────────────────────
state = {
    "texture_idx": 0,
    "msaa_idx": 2,          # default = 4
    "fov": 45.0,
    "distance": 3.0,
    "light_az": 0.3,
    "light_el": -1.0,
    "light_intensity": 1.0,
    "ambient": 0.15,
    "specular": 0.5,
    "shininess": 32.0,
    "bg_color": (0.12, 0.12, 0.14),
    "auto_rotate": False,
    "prev_texture_idx": -1,
}

# ── View + panels ─────────────────────────────────────────────────
view = vultorch.View("11 - 3D Surface Inspector", 1200, 800)
ctrl = view.panel("Controls", side="left", width=0.28)
scene_panel = view.panel("3D View")

# ── SceneView setup ──────────────────────────────────────────────
scene = vultorch.SceneView("Inspector", 800, 600, msaa=4)

# Initial texture
current_texture = make_checkerboard()


# ── Controls ──────────────────────────────────────────────────────
@ctrl.on_frame
def draw_controls():
    ctrl.text(f"FPS: {view.fps:.1f}")
    ctrl.separator()

    # ── Texture selection ──
    ctrl.text("Texture")
    state["texture_idx"] = ctrl.combo("Texture", TEXTURE_NAMES, default=0)

    ctrl.separator()

    # ── Camera ──
    ctrl.text("Camera")
    state["fov"] = ctrl.slider("FOV", 10.0, 120.0, default=45.0)
    state["distance"] = ctrl.slider("Distance", 1.0, 10.0, default=3.0)
    state["auto_rotate"] = ctrl.checkbox("Auto Rotate", default=False)

    ctrl.separator()

    # ── Light ──
    ctrl.text("Light")
    state["light_az"] = ctrl.slider("Light Az", -3.14, 3.14, default=0.3)
    state["light_el"] = ctrl.slider("Light El", -3.14, 3.14, default=-1.0)
    state["light_intensity"] = ctrl.slider("Intensity", 0.0, 3.0, default=1.0)
    state["ambient"] = ctrl.slider("Ambient", 0.0, 1.0, default=0.15)
    state["specular"] = ctrl.slider("Specular", 0.0, 2.0, default=0.5)
    state["shininess"] = ctrl.slider("Shininess", 1.0, 128.0, default=32.0)

    ctrl.separator()

    # ── MSAA ──
    ctrl.text("Anti-Aliasing")
    state["msaa_idx"] = ctrl.combo("MSAA", MSAA_OPTIONS, default=2)

    ctrl.separator()

    # ── Background ──
    state["bg_color"] = ctrl.color_picker("Background",
                                           default=(0.12, 0.12, 0.14))

    ctrl.separator()

    # Camera info readback
    cam = scene.camera
    ctrl.text(f"Azimuth:   {cam.azimuth:.2f}")
    ctrl.text(f"Elevation: {cam.elevation:.2f}")

    if ctrl.button("Reset Camera", width=120):
        scene.camera.reset()

    ctrl.separator()
    ctrl.text_wrapped(
        "Left-drag to orbit, right-drag to pan, "
        "middle-drag/scroll to zoom. "
        "All controls update in real time."
    )


# ── 3D View panel ────────────────────────────────────────────────
@scene_panel.on_frame
def draw_scene():
    global current_texture

    # Regenerate texture if selection changed
    if state["texture_idx"] != state["prev_texture_idx"]:
        current_texture = TEXTURE_FNS[state["texture_idx"]]()
        state["prev_texture_idx"] = state["texture_idx"]

    # Apply camera settings
    scene.camera.fov = state["fov"]
    scene.camera.distance = state["distance"]

    # Auto-rotate
    if state["auto_rotate"]:
        scene.camera.azimuth += 0.02

    # Apply light settings
    az = state["light_az"]
    el = state["light_el"]
    scene.light.direction = (
        math.cos(el) * math.sin(az),
        math.sin(el),
        math.cos(el) * math.cos(az),
    )
    scene.light.intensity = state["light_intensity"]
    scene.light.ambient = state["ambient"]
    scene.light.specular = state["specular"]
    scene.light.shininess = state["shininess"]

    # Apply MSAA
    msaa_val = int(MSAA_OPTIONS[state["msaa_idx"]])
    scene.msaa = msaa_val

    # Apply background
    scene.background = state["bg_color"]

    # Upload texture and render
    scene.set_tensor(current_texture)
    scene.render()


view.run()
