"""
01 - Hello Tensor
=================
Minimal example: display a CUDA tensor in a window with just a few lines.

Key concepts
------------
- View   : Top-level window that owns the event loop
- Panel  : A dockable sub-window (ImGui Window) inside the View
- Canvas : A GPU image slot inside a Panel that displays a tensor
- bind() : Connects a tensor to a canvas; it is re-uploaded every frame

Hierarchy: View -> Panel -> Canvas -> bind(tensor) -> run()
"""

import torch
import vultorch

# -- 1. Prepare data --------------------------------------------------------
# A 256x256 RGB gradient.  Any (H, W, C) float32 CUDA tensor will work.
H, W = 256, 256
x = torch.linspace(0, 1, W, device="cuda")            # horizontal ramp
y = torch.linspace(0, 1, H, device="cuda")            # vertical ramp
t = torch.stack([
    x.unsqueeze(0).expand(H, W),                       # R channel: left->right
    y.unsqueeze(1).expand(H, W),                       # G channel: top->bottom
    torch.full((H, W), 0.3, device="cuda"),            # B channel: constant
], dim=-1)                                             # shape: (256, 256, 3)

# -- 2. Create View -> Panel -> Canvas -> bind tensor -----------------------
view   = vultorch.View("01 - Hello Tensor", 512, 512)  # open a 512x512 window
panel  = view.panel("Viewer")                          # add one panel
canvas = panel.canvas("gradient")                      # add a canvas to it
canvas.bind(t)                                         # bind the tensor

# The four lines above can also be written as a one-liner:
# view.panel("Viewer").canvas("gradient").bind(t)

# -- 3. Run -----------------------------------------------------------------
# run() blocks until the user closes the window.
# The canvas auto-fills the panel and re-uploads from the bound tensor
# every frame.
view.run()
