"""05 — Zero-Copy Tensor: allocate Vulkan-shared memory for true zero-copy display.

The tensor returned by `vultorch.create_tensor()` shares GPU memory with Vulkan.
Any CUDA write is instantly visible — no memcpy, no staging buffer.
"""

import math
import torch
import vultorch
from vultorch import ui

WIN_W, WIN_H = 900, 600
win = vultorch.Window("Zero Copy Demo", WIN_W, WIN_H)

# ── Direction constants (ImGuiDir) ──────────────────────────────────
DIR_LEFT  = 0
DIR_RIGHT = 1
DIR_UP    = 2
DIR_DOWN  = 3

# Allocate a shared tensor — memory is shared between CUDA and Vulkan
tensor = vultorch.create_tensor(256, 256, channels=4, device="cuda:0")
# tensor is a standard torch.Tensor, fully usable with any PyTorch op

y = torch.linspace(0, 1, 256, device="cuda").unsqueeze(1).expand(256, 256)
x = torch.linspace(0, 1, 256, device="cuda").unsqueeze(0).expand(256, 256)

frame = 0
first_frame = True


def setup_initial_layout(dockspace_id: int):
    """Build initial layout: Tensor View (left 65%) | Info (right 35%)."""
    ui.dock_builder_remove_node(dockspace_id)
    ui.dock_builder_add_node(dockspace_id, 1 << 10)
    ui.dock_builder_set_node_size(dockspace_id, float(WIN_W), float(WIN_H))

    id_left, id_right = ui.dock_builder_split_node(
        dockspace_id, DIR_LEFT, 0.65)

    ui.dock_builder_dock_window("Zero-Copy Tensor", id_left)
    ui.dock_builder_dock_window("Info", id_right)
    ui.dock_builder_finish(dockspace_id)


while win.poll():
    if not win.begin_frame():
        continue
    t = ui.get_time()
    frame += 1

    # ── Full-viewport DockSpace ─────────────────────────────────
    dockspace_id = ui.dock_space_over_viewport(flags=8)
    if first_frame:
        setup_initial_layout(dockspace_id)
        first_frame = False

    # Write directly to the shared tensor — zero-copy!
    phase = math.fmod(t * 0.5, 1.0)
    tensor[:, :, 0] = ((x * 4 + t).sin() * 0.5 + 0.5)       # R
    tensor[:, :, 1] = ((y * 4 + t * 1.3).cos() * 0.5 + 0.5)  # G
    tensor[:, :, 2] = phase                                     # B
    tensor[:, :, 3] = 1.0                                       # A

    ui.begin("Zero-Copy Tensor")
    ui.text(f"FPS: {ui.get_io_framerate():.0f}  Frame: {frame}")
    ui.text(f"Shape: {list(tensor.shape)}  data_ptr: 0x{tensor.data_ptr():X}")
    ui.separator()
    vultorch.show(tensor, name="shared")
    ui.end()

    ui.begin("Info")
    ui.text_wrapped(
        "This tensor's memory is shared between CUDA and Vulkan via "
        "Vulkan external memory interop. Every CUDA write is immediately "
        "visible on screen — no GPU-GPU copy, no staging buffer."
    )
    ui.separator()
    ui.text(f"Tensor device: {tensor.device}")
    ui.text(f"Tensor dtype:  {tensor.dtype}")
    ui.text(f"Tensor shape:  {list(tensor.shape)}")
    ui.text(f"Data pointer:  0x{tensor.data_ptr():X}")
    ui.separator()
    ui.text(f"Time: {t:.1f}s")
    ui.text(f"Phase: {phase:.3f}")
    ui.end()

    win.end_frame()
win.destroy()
