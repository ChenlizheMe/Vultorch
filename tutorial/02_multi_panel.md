# 02 — Multi-Panel

> **Example file:** `examples/02_imgui_controls.py`

One panel is nice. But in practice you want to see your loss map, gradient field,
and output side-by-side — without writing a single line of layout code.

Good news: Vultorch panels are **dockable**. Just create them and they'll
arrange themselves. Users can drag, resize, and rearrange at will.

This chapter shows two patterns:

- **One canvas per panel** — three panels stacked, each with its own tensor.
- **Multiple canvases in one panel** — one panel, three canvases, auto-split.

## Layout

Here's what the window looks like:

| Left side (main area) | Right side (`side="right"`) |
|-----------------------|----------------------------|
| **Red** panel — `red_img` | **Combined** panel |
| **Green** panel — `green_img` | `c_red` canvas |
| **Blue** panel — `blue_img` | `c_green` canvas |
| | `c_blue` canvas |

Left: 3 separate panels, each one canvas. Right: 1 panel, 3 canvases sharing space.

## Full code

```python
import torch
import vultorch

H, W = 128, 128
device = "cuda"

# Three different tensors
x = torch.linspace(0, 1, W, device=device)
red = torch.zeros(H, W, 3, device=device)
red[:, :, 0] = x.unsqueeze(0).expand(H, W)

y = torch.linspace(0, 1, H, device=device)
green = torch.zeros(H, W, 3, device=device)
green[:, :, 1] = y.unsqueeze(1).expand(H, W)

blue = torch.zeros(H, W, 3, device=device)
cx = (torch.arange(W, device=device) // 32) % 2
cy = (torch.arange(H, device=device) // 32) % 2
blue[:, :, 2] = (cx.unsqueeze(0) ^ cy.unsqueeze(1)).float()

view = vultorch.View("02 - Multi-Panel", 1200, 600)

# Left: 3 panels, each with one canvas
panel_r = view.panel("Red")
panel_g = view.panel("Green")
panel_b = view.panel("Blue")
panel_r.canvas("red_img").bind(red)
panel_g.canvas("green_img").bind(green)
panel_b.canvas("blue_img").bind(blue)

# Right: 1 panel with 3 canvases (auto-split vertically)
combined = view.panel("Combined", side="right", width=0.5)
combined.canvas("c_red").bind(red)
combined.canvas("c_green").bind(green)
combined.canvas("c_blue").bind(blue)

view.run()
```

## Key takeaways

1. **Auto-layout** — panels without `side=` stack vertically.
   Add `side="right"` (or `"left"`) + `width=0.5` to dock to one side.

2. **Multi-canvas** — call `panel.canvas()` multiple times.
   When several canvases have `fit=True` (the default), they split the
   vertical space equally. No manual height math.

3. **Still no callback** — static data only needs `bind()` + `run()`.
   Dynamic updates come in a later chapter.

4. **Drag & drop** — all panels are dockable. Users can rearrange,
   float, or resize them at runtime.

!!! note
    The same tensor can be bound to multiple canvases at once —
    `red` appears in both the left panel and the right combined panel.
