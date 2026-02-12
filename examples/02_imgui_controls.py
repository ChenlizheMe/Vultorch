"""02 — ImGui Controls: demonstrate built-in UI widgets with docking layout.

Shows sliders, buttons, checkboxes, color pickers, combo boxes,
plots, and progress bars — all driven from Python.
"""

import math
import torch
import vultorch
from vultorch import ui

WIN_W, WIN_H = 1024, 720
win = vultorch.Window("ImGui Controls", WIN_W, WIN_H)

# ── Direction constants (ImGuiDir) ──────────────────────────────────
DIR_LEFT  = 0
DIR_RIGHT = 1
DIR_UP    = 2
DIR_DOWN  = 3

# ── State variables ─────────────────────────────────────────────────
counter = 0
slider_f = 0.5
slider_i = 50
check_a = True
check_b = False
color = (0.2, 0.6, 1.0)
combo_idx = 0
combo_items = ["Viridis", "Plasma", "Inferno", "Magma", "Grayscale"]
text_input = "Hello Vultorch!"
loss_history: list = []
first_frame = True

# ── Animated tensor ─────────────────────────────────────────────────
H, W = 128, 128
y = torch.linspace(0, 1, H, device="cuda").unsqueeze(1).expand(H, W)
x = torch.linspace(0, 1, W, device="cuda").unsqueeze(0).expand(H, W)


def setup_initial_layout(dockspace_id: int):
    """Build initial docked layout: Controls (left) | Tensor Preview (right)."""
    ui.dock_builder_remove_node(dockspace_id)
    ui.dock_builder_add_node(dockspace_id, 1 << 10)
    ui.dock_builder_set_node_size(dockspace_id, float(WIN_W), float(WIN_H))

    id_left, id_right = ui.dock_builder_split_node(
        dockspace_id, DIR_LEFT, 0.40)

    ui.dock_builder_dock_window("Controls", id_left)
    ui.dock_builder_dock_window("Tensor Preview", id_right)
    ui.dock_builder_finish(dockspace_id)


while win.poll():
    if not win.begin_frame():
        continue
    t = ui.get_time()

    # ── Full-viewport DockSpace ─────────────────────────────────
    dockspace_id = ui.dock_space_over_viewport(flags=8)
    if first_frame:
        setup_initial_layout(dockspace_id)
        first_frame = False

    # ── Controls panel ──────────────────────────────────────────
    ui.begin("Controls")
    ui.text(f"FPS: {ui.get_io_framerate():.0f}")
    ui.separator()

    # Button
    if ui.button("Click Me"):
        counter += 1
    ui.same_line()
    ui.text(f"Count: {counter}")

    # Sliders
    slider_f = ui.slider_float("Float Slider", slider_f, 0.0, 1.0)
    slider_i = ui.slider_int("Int Slider", slider_i, 0, 100)

    # Checkboxes
    check_a = ui.checkbox("Feature A", check_a)
    ui.same_line()
    check_b = ui.checkbox("Feature B", check_b)

    # Color picker
    color = ui.color_edit3("Tint Color", *color)

    # Combo box
    combo_idx = ui.combo("Colormap", combo_idx, combo_items)

    ui.separator()

    # Progress bar
    progress = math.fmod(t / 10.0, 1.0)
    ui.progress_bar(progress, overlay=f"{int(progress * 100)}%")

    # Plot
    loss_history.append(math.exp(-t * 0.2) * math.sin(t * 3) * 0.5 + 0.5)
    if len(loss_history) > 300:
        loss_history.pop(0)
    ui.plot_lines("##loss", loss_history, width=0, height=120)

    ui.end()

    # ── Tensor view (tinted by the color picker) ────────────────
    ui.begin("Tensor Preview")
    phase = math.fmod(t * 0.3, 1.0)
    tensor = torch.stack([
        ((x + phase) % 1.0) * color[0],
        y * color[1],
        torch.full((H, W), color[2], device="cuda"),
        torch.ones(H, W, device="cuda") * slider_f,
    ], dim=-1)
    vultorch.show(tensor, name="tinted")
    ui.text(f"Opacity: {slider_f:.2f}  Colormap: {combo_items[combo_idx]}")
    ui.end()

    win.end_frame()
win.destroy()
