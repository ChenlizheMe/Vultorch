"""05 — Zero-Copy Tensor: allocate Vulkan-shared memory for true zero-copy display.

The tensor returned by `vultorch.create_tensor()` shares GPU memory with Vulkan.
Any CUDA write is instantly visible — no memcpy, no staging buffer.
"""

import math
import torch
import vultorch
from vultorch import ui

win = vultorch.Window("Zero Copy Demo", 800, 600)

# Allocate a shared tensor — memory is shared between CUDA and Vulkan
tensor = vultorch.create_tensor(256, 256, channels=4, device="cuda:0")
# tensor is a standard torch.Tensor, fully usable with any PyTorch op

y = torch.linspace(0, 1, 256, device="cuda").unsqueeze(1).expand(256, 256)
x = torch.linspace(0, 1, 256, device="cuda").unsqueeze(0).expand(256, 256)

frame = 0

while win.poll():
    if not win.begin_frame():
        continue
    t = ui.get_time()
    frame += 1

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
    ui.text_wrapped(
        "This tensor's memory is shared between CUDA and Vulkan. "
        "Every write is immediately visible — no GPU-GPU copy needed."
    )
    vultorch.show(tensor, name="shared")
    ui.end()

    win.end_frame()
win.destroy()
