"""01 — Hello Tensor: the simplest vultorch program.

One window, one CUDA tensor, one line to display it.
"""

import torch
import vultorch
from vultorch import ui

# Create a colorful gradient tensor on the GPU
H, W = 256, 256
y = torch.linspace(0, 1, H, device="cuda").unsqueeze(1).expand(H, W)
x = torch.linspace(0, 1, W, device="cuda").unsqueeze(0).expand(H, W)
tensor = torch.stack([x, y, torch.full((H, W), 0.5, device="cuda"),
                      torch.ones(H, W, device="cuda")], dim=-1)  # RGBA

# Open a window and display it
win = vultorch.Window("Hello Tensor", 512, 512)
while win.poll():
    if not win.begin_frame():
        continue

    ui.begin("Tensor Viewer")
    ui.text(f"Shape: {list(tensor.shape)}  device: {tensor.device}")
    vultorch.show(tensor, name="gradient")
    ui.end()

    win.end_frame()
win.destroy()
