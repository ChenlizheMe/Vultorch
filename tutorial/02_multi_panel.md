# 02 — Multi-Panel

> **Example file:** `examples/02_imgui_controls.py`

One panel is nice. But in practice you want to see your loss map, gradient field,
and output side-by-side — without writing a single line of layout code.

Good news: Vultorch panels **auto-arrange** themselves. You just
create them, and they stack up inside the window like slides in a
presentation. Want a panel on the right? Pass `side="right"` and
Vultorch splits the window for you — no CSS, no grid coordinates,
no fighting with subplot indices.

Even better: at runtime you can drag any panel's title bar to
rearrange, resize by dragging edges, or pull a panel out into its
own floating window. It's like a tiling window manager that you
didn't have to configure.

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

1. **Auto-layout** — panels you create without `side=` stack vertically
   in the main area.  Think of it as `plt.subplot(3, 1, ...)` but
   without counting rows and columns.

2. **`side="right"` + `width=0.5`** — this tells Vultorch: *"split the
   window and give this panel the right 50%."*  The value `0.5` is a
   ratio, not pixels.  `width=0.3` means 30% of the window.
   You can also use `"left"`, `"top"`, or `"bottom"`.

3. **Multi-canvas** — call `panel.canvas()` multiple times.
   When several canvases have `fit=True` (the default), they evenly
   split the panel's height.  `fit=True` just means *"stretch to fill
   available space"* — like how a single `imshow` fills the whole axes.
   No manual height math.

4. **Still no callback** — static data only needs `bind()` + `run()`.
   Dynamic updates come in a later chapter.

5. **Drag & drop** — try it: grab a panel's title bar and drag it to
   another edge. Pull it out into a floating window.  Everything is
   rearrangeable at runtime with your mouse.

!!! note
    The same tensor can be bound to multiple canvases at once —
    `red` appears in both the left panel and the right combined panel.
