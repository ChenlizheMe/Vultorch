# 02 — Multi-Panel

> **Example file:** `examples/02_imgui_controls.py`

A single View can host any number of Panels.  Each Panel is an independent
dockable window that can be dragged, resized, and rearranged by the user.
This chapter shows two layout patterns:

- **One canvas per panel** — three panels stacked vertically, each showing a different tensor.
- **Multiple canvases in one panel** — a single panel with three canvases that automatically divide the space equally.

## Layout diagram

```
┌──────────────────────────────────────────────────┐
│  Red (panel)             │                       │
│    canvas: red_img       │  Combined (panel)     │
├──────────────────────────┤                       │
│  Green (panel)           │    canvas: c_red      │
│    canvas: green_img     │    canvas: c_green    │
├──────────────────────────┤    canvas: c_blue     │
│  Blue (panel)            │                       │
│    canvas: blue_img      │                       │
└──────────────────────────────────────────────────┘
```

## Full code

```python
import torch
import vultorch

# -- 1. Prepare three tensors ----------------------------------------------
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

# Blue checkerboard
blue = torch.zeros(H, W, 3, device=device)
cx = (torch.arange(W, device=device) // 32) % 2
cy = (torch.arange(H, device=device) // 32) % 2
blue[:, :, 2] = (cx.unsqueeze(0) ^ cy.unsqueeze(1)).float()

# -- 2. Create View ---------------------------------------------------------
view = vultorch.View("02 - Multi-Panel", 1200, 600)

# -- 3. Left side: 3 panels, each with one canvas --------------------------
panel_r = view.panel("Red")
panel_g = view.panel("Green")
panel_b = view.panel("Blue")

panel_r.canvas("red_img").bind(red)
panel_g.canvas("green_img").bind(green)
panel_b.canvas("blue_img").bind(blue)

# -- 4. Right side: 1 panel with 3 canvases --------------------------------
combined = view.panel("Combined", side="right", width=0.5)
combined.canvas("c_red").bind(red)
combined.canvas("c_green").bind(green)
combined.canvas("c_blue").bind(blue)

# -- 5. Run -----------------------------------------------------------------
view.run()
```

## Key points

1. **Automatic layout** — panels without a `side=` argument are stacked
   vertically in the main area.  Use `side="left"` or `side="right"` to dock a
   panel to a specific side, with `width` setting the ratio (e.g. `0.5` = 50%).

2. **One canvas per panel vs. many** — each `panel.canvas()` call creates a new
   canvas inside that panel.  When there are multiple canvases with `fit=True`
   (the default), they automatically share the vertical space equally.

3. **No callback needed** — for static data, just `bind()` and `run()`.  The
   `@view.on_frame` decorator is only required when you need per-frame logic
   (upcoming in later chapters).

4. **User can rearrange** — all panels are dockable.  Users can drag title bars
   to reorder panels, detach them as floating windows, or resize borders.

!!! note
    The same tensor can be bound to multiple canvases simultaneously.
    In this example, `red`, `green`, and `blue` each appear in both the left
    panels and the right combined panel.
