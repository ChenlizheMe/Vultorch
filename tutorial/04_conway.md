# 04 — Conway's Game of Life

> **Example file:** `examples/04_conway.py`

Loss curves, training visualizations — all very serious.
Let's take a break and build something fun: Conway's Game of Life,
running entirely on the GPU, displayed in zero-copy, with buttons
and sliders to play god.

More importantly, this chapter shows that `create_tensor` isn't only
for neural networks. **Any GPU computation** can be displayed through
Vultorch's shared memory — simulations, procedural generation, physics,
anything that lives on CUDA.

## What we're building

A 256×256 cellular automaton with a control panel:

| Area | Content |
|------|---------|
| Left | **Controls** — play/pause, step, speed slider, pattern presets, color pickers |
| Right | **Grid** — the simulation, pixel-perfect (`filter="nearest"`) |

Everything runs on the GPU. The display tensor uses `create_tensor` for
zero-copy — the grid never touches the CPU.

## New friends

| New thing | What it does | Why it matters |
|-----------|-------------|----------------|
| `filter="nearest"` | Shows each pixel as a sharp square, no blurring | Without it, bilinear interpolation smudges cell boundaries |
| `side="left"` sidebar | Places a panel on the left, taking 22% of the window | Gives you a permanent control strip next to your visualization |
| `@panel.on_frame` | Per-panel widget callback | Widget calls (`button`, `slider`, `color_picker`) go inside here |
| `panel.button(label)` | A clickable button | Returns `True` on the frame it was clicked |
| `with panel.row():` | Puts the next widgets on the **same line** | Without it, widgets stack vertically (one per line).  Use `row()` to put two buttons side-by-side. It's a Python `with` block — everything inside the block goes on one line |
| `panel.color_picker` | An RGB color picker widget | Click the colored square to open a palette |
| Circular padding + conv2d | GPU-parallel neighbour count | The whole simulation is one convolution |

## The simulation trick

Counting neighbours in Conway's Game of Life is just a 2D convolution
with a 3×3 kernel of all ones (center zero):

```
1 1 1
1 0 1
1 1 1
```

PyTorch's `F.conv2d` does this in one GPU kernel call — no loops, no
per-cell logic. Circular padding wraps the edges so gliders fly off one
side and reappear on the other.

```python
kernel = torch.tensor([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], dtype=torch.float32, device=device)
padded = F.pad(inp, (1, 1, 1, 1), mode='circular')
neighbours = F.conv2d(padded, kernel.reshape(1, 1, 3, 3)).squeeze()
```

Then the rules are just two boolean masks:

```python
survive = (grid == 1) & ((neighbours == 2) | (neighbours == 3))
birth   = (grid == 0) & (neighbours == 3)
grid[:] = (survive | birth).float()
```

## Full code

```python
import torch
import vultorch

if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"

# ── Grid parameters ───────────────────────────────────────────────
GRID_H, GRID_W = 256, 256

# ── View + panels ─────────────────────────────────────────────────
view = vultorch.View("04 - Conway's Game of Life", 1024, 768)
grid_panel = view.panel("Grid")
ctrl_panel = view.panel("Controls", side="left", width=0.22)

# ── Display tensor (RGBA, zero-copy) ─────────────────────────────
display = vultorch.create_tensor(GRID_H, GRID_W, channels=4,
                                 device=device, name="grid",
                                 window=view.window)
canvas = grid_panel.canvas("grid", filter="nearest")
canvas.bind(display)

# ── Simulation state ─────────────────────────────────────────────
grid = torch.zeros(GRID_H, GRID_W, dtype=torch.float32, device=device)

state = {
    "running": False,
    "generation": 0,
    "speed": 1,
    "prob": 0.3,
    "alive_color": (0.0, 1.0, 0.4),
    "dead_color": (0.05, 0.05, 0.08),
}


def randomize():
    grid[:] = (torch.rand(GRID_H, GRID_W, device=device) < state["prob"]).float()
    state["generation"] = 0

def clear():
    grid.zero_()
    state["generation"] = 0

def step_simulation():
    kernel = torch.tensor([[1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]], dtype=torch.float32, device=device)
    inp = grid.unsqueeze(0).unsqueeze(0)
    k = kernel.unsqueeze(0).unsqueeze(0)
    padded = torch.nn.functional.pad(inp, (1, 1, 1, 1), mode='circular')
    neighbours = torch.nn.functional.conv2d(padded, k).squeeze()

    survive = (grid == 1) & ((neighbours == 2) | (neighbours == 3))
    birth = (grid == 0) & (neighbours == 3)
    grid[:] = (survive | birth).float()
    state["generation"] += 1

def grid_to_display():
    alive_r, alive_g, alive_b = state["alive_color"]
    dead_r, dead_g, dead_b = state["dead_color"]
    display[:, :, 0] = dead_r + (alive_r - dead_r) * grid
    display[:, :, 1] = dead_g + (alive_g - dead_g) * grid
    display[:, :, 2] = dead_b + (alive_b - dead_b) * grid
    display[:, :, 3] = 1.0

randomize()


@view.on_frame
def update():
    if state["running"]:
        for _ in range(state["speed"]):
            step_simulation()
    grid_to_display()


@ctrl_panel.on_frame
def draw_controls():
    ctrl_panel.text(f"Generation: {state['generation']}")
    ctrl_panel.text(f"Alive cells: {int(grid.sum().item())}")
    ctrl_panel.text(f"FPS: {view.fps:.1f}")
    ctrl_panel.separator()

    with ctrl_panel.row():
        label = "Pause" if state["running"] else "Play"
        if ctrl_panel.button(label, width=80):
            state["running"] = not state["running"]
        if ctrl_panel.button("Step", width=80):
            step_simulation()

    with ctrl_panel.row():
        if ctrl_panel.button("Randomize", width=80):
            randomize()
        if ctrl_panel.button("Clear", width=80):
            clear()

    ctrl_panel.separator()
    state["speed"] = ctrl_panel.slider_int("Speed", 1, 20, default=1)
    state["prob"] = ctrl_panel.slider("Cell Probability", 0.05, 0.8,
                                       default=0.3)

    ctrl_panel.separator()
    ctrl_panel.text("Colors")
    state["alive_color"] = ctrl_panel.color_picker(
        "Alive", default=(0.0, 1.0, 0.4))
    state["dead_color"] = ctrl_panel.color_picker(
        "Dead", default=(0.05, 0.05, 0.08))

    ctrl_panel.separator()
    ctrl_panel.text("Patterns")
    with ctrl_panel.row():
        if ctrl_panel.button("Glider", width=80):
            clear()
            grid[1, 2] = 1; grid[2, 3] = 1
            grid[3, 1] = 1; grid[3, 2] = 1; grid[3, 3] = 1
        if ctrl_panel.button("Pulsar", width=80):
            clear()
            # ... place pulsar pattern ...
        if ctrl_panel.button("Gosper Gun", width=100):
            clear()
            # ... place Gosper glider gun ...

    ctrl_panel.separator()
    ctrl_panel.text_wrapped(
        "Click Play to start, or Step to advance one generation. "
        "Use Randomize to reset with random cells."
    )


view.run()
```

*(The full example file includes helper functions for all pattern placements.)*

## What just happened?

1. **Grid** — a plain `(256, 256)` float32 CUDA tensor.  `1.0` = alive,
   `0.0` = dead.  No classes, no fancy data structures — just a tensor.

2. **Simulation** — `step_simulation()` uses `F.conv2d` with circular
   padding to count neighbours, then applies the birth/survive rules
   with boolean masks.  The entire generation runs in two GPU kernels.

3. **Display** — `create_tensor` allocates shared Vulkan/CUDA memory.
   `grid_to_display()` lerps between dead and alive colors and writes
   into it.  Zero copy to screen.

4. **Controls** — `@ctrl_panel.on_frame` draws all widgets inside the
   Controls panel.  `panel.button()`, `panel.slider_int()`,
   `panel.color_picker()`, and `with panel.row()` keep the layout
   compact.  State lives in a plain Python dict.

## Key takeaways

1. **`create_tensor` is for everything** — not just neural networks.
   Any GPU computation that produces an image-like tensor can be
   displayed with zero copy.

2. **`filter="nearest"`** — crucial for pixel-art / grid simulations.
   Without it, bilinear interpolation blurs cell boundaries.  Think of
   it like `plt.imshow(data, interpolation='nearest')` — same idea,
   you want to see the actual pixels.

3. **Convolution = neighbour counting** — a cute trick that replaces
   nested Python loops with a single GPU kernel.  The game runs at
   hundreds of FPS even at large grid sizes.

4. **Panel widgets** — inside `@panel.on_frame` you call
   `panel.button()`, `panel.slider_int()`, `panel.color_picker()`.
   Each call creates one interactive element.  They stack top-to-bottom
   automatically, like lines of `print()` output.  No positioning
   code needed.

5. **`with panel.row():`** — by default widgets go one-per-line.
   Wrap several widget calls in `with panel.row():` to put them on
   the **same line** instead.  It's just a Python `with` block —
   nothing exotic.

6. **Pattern presets** — the Glider, Pulsar, and Gosper Gun buttons
   demonstrate how to set initial conditions by writing directly into
   the grid tensor.

!!! tip
    Crank the Speed slider to 20 and watch the grid evolve at 20
    generations per frame.  On a modern GPU you'll still hold 60+ FPS.

!!! note
    The grid wraps around thanks to `mode='circular'` padding.
    Gliders that fly off the right edge reappear on the left.
