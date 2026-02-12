"""Vultorch Docking Demo — drag windows to rearrange layout.

ImGui Docking branch enables:
  • Drag any window's title bar onto another to create split/tab layouts
  • Drag to screen edges to dock left/right/top/bottom
  • Double-click a tab to undock
  • All layout is saved/restored in imgui.ini automatically

Usage:
    conda activate neuralshader
    python examples/docking_demo.py
"""

import math
import vultorch
from vultorch import ui

# ── Optional: torch for live tensor display ─────────────────────────
try:
    import torch
    HAS_TORCH = torch.cuda.is_available() and vultorch.HAS_CUDA
except ImportError:
    HAS_TORCH = False

# ── Create window ───────────────────────────────────────────────────
WIN_W, WIN_H = 1440, 900
win = vultorch.Window("Vultorch Docking Demo", WIN_W, WIN_H)

# ── State ───────────────────────────────────────────────────────────
counter = 0
slider_val = 0.5
check_val = True
color = (0.4, 0.7, 1.0)
combo_idx = 0
loss_history: list = []
first_frame = True

# ── Direction constants (ImGuiDir) ──────────────────────────────────
DIR_LEFT  = 0
DIR_RIGHT = 1
DIR_UP    = 2
DIR_DOWN  = 3

# ── Tensor + 3D scene (if CUDA available) ───────────────────────────
tensor = None
scene = None
if HAS_TORCH:
    tensor = torch.zeros(256, 256, 4, dtype=torch.float32, device="cuda")
    scene = vultorch.SceneView("3D Preview", 800, 600, msaa=4)


def setup_initial_layout(dockspace_id: int):
    """Build the initial docked layout programmatically.

    Layout:
      ┌──────────┬────────────────────────┐
      │          │       2D Tensor        │
      │ Controls ├────────────────────────┤
      │          │       3D Scene         │
      ├──────────┴────────────────────────┤
      │  Training Monitor  │    Info      │
      └───────────────────────────────────┘
    """
    # Clear any existing layout in this dockspace
    ui.dock_builder_remove_node(dockspace_id)

    # Create root node covering the full viewport
    # Flag 1 << 10 = ImGuiDockNodeFlags_DockSpace
    ui.dock_builder_add_node(dockspace_id, 1 << 10)
    ui.dock_builder_set_node_size(dockspace_id, float(WIN_W), float(WIN_H))

    # Split: bottom 25% ↔ top 75%
    id_bottom, id_top = ui.dock_builder_split_node(
        dockspace_id, DIR_DOWN, 0.25)

    # Top: left 22% (Controls) ↔ right 78% (main area)
    id_left, id_main = ui.dock_builder_split_node(
        id_top, DIR_LEFT, 0.22)

    # Main area: upper 55% (2D Tensor) ↔ lower 45% (3D Scene)
    id_tensor, id_scene = ui.dock_builder_split_node(
        id_main, DIR_UP, 0.55)

    # Bottom: left 60% (Training Monitor) ↔ right 40% (Info)
    id_monitor, id_info = ui.dock_builder_split_node(
        id_bottom, DIR_LEFT, 0.60)

    # Assign windows to nodes
    ui.dock_builder_dock_window("Controls", id_left)
    ui.dock_builder_dock_window("2D Tensor", id_tensor)
    ui.dock_builder_dock_window("3D Scene", id_scene)
    ui.dock_builder_dock_window("Training Monitor", id_monitor)
    ui.dock_builder_dock_window("Info", id_info)

    # Finalize
    ui.dock_builder_finish(dockspace_id)

# ═════════════════════════════════════════════════════════════════════
#  Main loop
# ═════════════════════════════════════════════════════════════════════
while win.poll():
    if not win.begin_frame():
        continue

    t = ui.get_time()

    # ── Full-viewport DockSpace ─────────────────────────────────────
    # PassthruCentralNode = 1 << 3 = 8
    dockspace_id = ui.dock_space_over_viewport(flags=8)

    # Set up initial layout on first frame (after DockSpace creates the ID)
    if first_frame:
        setup_initial_layout(dockspace_id)
        first_frame = False

    # ── Menu bar ────────────────────────────────────────────────────
    if ui.begin_main_menu_bar():
        if ui.begin_menu("File"):
            if ui.menu_item("Quit", "Ctrl+Q"):
                break
            ui.end_menu()
        if ui.begin_menu("View"):
            ui.menu_item("Demo Window", selected=False)
            ui.end_menu()
        if ui.begin_menu("Help"):
            if ui.begin_menu("About"):
                ui.text(f"Vultorch v{vultorch.__version__}")
                ui.text("ImGui Docking Branch")
                ui.text("Drag windows to rearrange!")
                ui.end_menu()
            ui.end_menu()
        ui.end_main_menu_bar()

    # ── Window 1: Controls ──────────────────────────────────────────
    ui.begin("Controls")
    ui.text(f"FPS: {ui.get_io_framerate():.1f}")
    ui.separator()

    if ui.button("Click Me!"):
        counter += 1
    ui.same_line()
    ui.text(f"Count: {counter}")

    slider_val = ui.slider_float("Opacity", slider_val, 0.0, 1.0)
    check_val = ui.checkbox("Enable Feature", check_val)
    color = ui.color_edit3("Tint", *color)
    combo_idx = ui.combo("Mode", combo_idx,
                         ["Diffuse", "Normal", "Depth", "AO"])
    ui.end()

    # ── Window 2: Training Monitor ──────────────────────────────────
    ui.begin("Training Monitor")
    # Fake loss curve
    loss_history.append(math.exp(-t * 0.3) * math.sin(t * 2.0) * 0.5 + 0.5)
    if len(loss_history) > 300:
        loss_history.pop(0)

    ui.text("Loss Curve")
    if loss_history:
        ui.plot_lines("##loss", loss_history, width=0, height=150)

    progress = min(t / 15.0, 1.0)
    ui.progress_bar(progress, overlay=f"Epoch {int(progress * 100)}%")
    ui.separator()
    ui.text(f"Slider = {slider_val:.3f}")
    ui.text(f"Color  = ({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
    ui.end()

    # ── Window 3: 2D Tensor View ────────────────────────────────────
    ui.begin("2D Tensor")
    if HAS_TORCH and tensor is not None:
        # Animate a colorful pattern
        phase = math.fmod(t * 0.5, 1.0)
        y = torch.linspace(0, 1, 256, device="cuda").unsqueeze(1).expand(256, 256)
        x = torch.linspace(0, 1, 256, device="cuda").unsqueeze(0).expand(256, 256)
        tensor = torch.stack([
            (x + phase).fmod(1.0),
            y,
            torch.full((256, 256), 0.8, device="cuda"),
            torch.ones(256, 256, device="cuda"),
        ], dim=-1)

        ui.text(f"Shape: {list(tensor.shape)}  dtype: float32  device: cuda")
        vultorch.show(tensor, name="animated", filter="linear")
    else:
        ui.text_wrapped("CUDA not available — install PyTorch with CUDA support.")
    ui.end()

    # ── Window 4: 3D Scene View ─────────────────────────────────────
    ui.begin("3D Scene")
    if scene is not None and tensor is not None:
        scene.set_tensor(tensor)

        # Inline controls
        scene.light.ambient = ui.slider_float("Ambient", scene.light.ambient, 0.0, 1.0)
        scene.light.intensity = ui.slider_float("Light", scene.light.intensity, 0.0, 3.0)
        scene.light.shininess = ui.slider_float("Shininess", scene.light.shininess, 1.0, 128.0)

        # Render (orbit camera with mouse drag built-in)
        scene.render()
    else:
        ui.text_wrapped("CUDA not available.")
    ui.end()

    # ── Window 5: Info ──────────────────────────────────────────────
    ui.begin("Info")
    ui.text_wrapped(
        "Try dragging window title bars!\n\n"
        "• Drag a window onto another window to create tabs\n"
        "• Drag to the edge of another window to split it\n"
        "• Drag to the screen edge for full-side docking\n"
        "• The layout is automatically saved to imgui.ini"
    )
    ui.separator()
    ui.text(f"Vultorch v{vultorch.__version__}")
    if vultorch.HAS_CUDA:
        ui.text("CUDA: enabled")
    else:
        ui.text_disabled("CUDA: disabled")
    ui.end()

    win.end_frame()

win.destroy()
