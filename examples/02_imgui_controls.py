"""
02 - Multi-Panel
=================
Demonstrates multiple Panels and multiple Canvases per Panel.

Layout
------
Left side  : 3 separate panels stacked vertically, each with its own canvas.
Right side : 1 panel containing 3 canvases stacked inside it.

Key concepts
------------
- A View can own any number of Panels (auto-laid-out via docking).
- Each Panel can own any number of Canvases.
- Use ``side="left"`` / ``side="right"`` to place panels on a specific side.
- When a Panel has multiple canvases with ``fit=True`` (default), the
  available vertical space is divided equally among them.
"""

import torch
import vultorch

# -- 1. Prepare three different tensors ------------------------------------
H, W = 128, 128
device = "cuda"

# Red horizontal gradient
x = torch.linspace(0, 1, W, device=device)
red = torch.zeros(H, W, 3, device=device)
red[:, :, 0] = x.unsqueeze(0).expand(H, W)

# Green vertical gradient
y = torch.linspace(0, 1, H, device=device)
green = torch.zeros(H, W, 3, device=device)
green[:, :, 1] = y.unsqueeze(1).expand(H, W)

# Blue checkerboard (32-pixel squares)
blue = torch.zeros(H, W, 3, device=device)
cx = (torch.arange(W, device=device) // 32) % 2
cy = (torch.arange(H, device=device) // 32) % 2
blue[:, :, 2] = (cx.unsqueeze(0) ^ cy.unsqueeze(1)).float()

# -- 2. Create View --------------------------------------------------------
view = vultorch.View("02 - Multi-Panel", 800, 1200)

# -- 3. Left side: 3 panels, each with one canvas -------------------------
# Panels without side= are placed in the main (left) area and stacked
# vertically by the auto-layout.
panel_r = view.panel("Red")
panel_g = view.panel("Green")
panel_b = view.panel("Blue")

panel_r.canvas("red_img").bind(red)
panel_g.canvas("green_img").bind(green)
panel_b.canvas("blue_img").bind(blue)

# -- 4. Right side: 1 panel with 3 canvases inside ------------------------
# side="right" docks this panel to the right half of the window.
combined = view.panel("Combined", side="right", width=0.5)

# A single panel can host multiple canvases.  They stack vertically and
# share the panel's height equally (fit=True by default).
combined.canvas("c_red").bind(red)
combined.canvas("c_green").bind(green)
combined.canvas("c_blue").bind(blue)

# -- 5. Run ----------------------------------------------------------------
# All four panels render automatically.  No on_frame callback needed for
# static data.  Users can drag panel tabs to rearrange the layout.
view.run()
