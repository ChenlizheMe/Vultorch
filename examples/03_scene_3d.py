"""03 — 3D Scene: render a CUDA tensor on a lit 3D plane.

Demonstrates SceneView with Blinn-Phong lighting, MSAA, and
orbit camera controlled by mouse drag.
"""

import math
import torch
import vultorch
from vultorch import ui

win = vultorch.Window("3D Scene", 1280, 720)
scene = vultorch.SceneView("3D View", 1024, 768, msaa=4)

# Generate a procedural texture on GPU
H, W = 256, 256
y = torch.linspace(-1, 1, H, device="cuda").unsqueeze(1).expand(H, W)
x = torch.linspace(-1, 1, W, device="cuda").unsqueeze(0).expand(H, W)
r = (x ** 2 + y ** 2).sqrt()

while win.poll():
    if not win.begin_frame():
        continue
    t = ui.get_time()

    # ── Animated procedural texture ─────────────────────────────
    wave = (r * 8.0 - t * 2.0).sin() * 0.5 + 0.5
    tensor = torch.stack([
        wave,
        (wave * 0.7 + 0.3).clamp(0, 1),
        torch.full((H, W), 0.9, device="cuda"),
        torch.ones(H, W, device="cuda"),
    ], dim=-1)

    # ── Controls ────────────────────────────────────────────────
    ui.begin("Light Controls")
    ui.text(f"FPS: {ui.get_io_framerate():.0f}")
    ui.separator()
    scene.light.ambient = ui.slider_float("Ambient", scene.light.ambient, 0.0, 1.0)
    scene.light.intensity = ui.slider_float("Intensity", scene.light.intensity, 0.0, 3.0)
    scene.light.shininess = ui.slider_float("Shininess", scene.light.shininess, 1.0, 128.0)
    scene.light.specular = ui.slider_float("Specular", scene.light.specular, 0.0, 1.0)

    bg = scene.background
    new_bg = ui.color_edit3("Background", *bg)
    scene.background = new_bg

    if ui.button("Reset Camera"):
        scene.camera.reset()
    ui.end()

    # ── 3D Scene ────────────────────────────────────────────────
    ui.begin("3D View")
    scene.set_tensor(tensor)
    scene.render()
    ui.end()

    # ── Camera info ─────────────────────────────────────────────
    ui.begin("Camera")
    cam = scene.camera
    ui.text(f"Azimuth:   {math.degrees(cam.azimuth):.1f} deg")
    ui.text(f"Elevation: {math.degrees(cam.elevation):.1f} deg")
    ui.text(f"Distance:  {cam.distance:.2f}")
    ui.text(f"FOV:       {cam.fov:.0f} deg")
    ui.text_wrapped("Left-drag: orbit | Right-drag: pan | Scroll: zoom")
    ui.end()

    win.end_frame()
win.destroy()
