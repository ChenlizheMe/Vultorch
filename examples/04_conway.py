"""
04 - Conway's Game of Life
==========================
Interactive cellular automaton running entirely on the GPU.

Key concepts
------------
- create_tensor   : Zero-copy shared memory — the grid is never copied
- on_frame         : Per-frame simulation step + rendering
- Panel widgets    : Buttons, sliders, checkboxes to control the simulation
- filter="nearest" : Pixel-perfect display without interpolation

Controls (Info panel)
---------------------
- Play / Pause      : toggle simulation
- Step              : advance one generation while paused
- Randomize         : fill the grid with random cells
- Clear             : kill all cells
- Speed             : generations per frame (1–20)
- Cell probability  : density for Randomize
- Draw with mouse   : left-click on the grid to toggle cells
"""

import torch
import vultorch

if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"

# ── Grid parameters ────────────────────────────────────────────────────
GRID_H, GRID_W = 256, 256

# ── View + panels ──────────────────────────────────────────────────────
view = vultorch.View("04 - Conway's Game of Life", 1024, 768)
grid_panel = view.panel("Grid")
ctrl_panel = view.panel("Controls", side="left", width=0.22)

# ── Display tensor (RGBA, zero-copy) ──────────────────────────────────
display = vultorch.create_tensor(GRID_H, GRID_W, channels=4,
                                 device=device, name="grid",
                                 window=view.window)
canvas = grid_panel.canvas("grid", filter="nearest")
canvas.bind(display)

# ── Simulation state ──────────────────────────────────────────────────
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
    """Fill the grid with random cells."""
    grid[:] = (torch.rand(GRID_H, GRID_W, device=device) < state["prob"]).float()
    state["generation"] = 0


def clear():
    """Kill all cells."""
    grid.zero_()
    state["generation"] = 0


def step_simulation():
    """Advance one generation using Conway's rules on the GPU.

    Rules:
        - A live cell with 2 or 3 neighbours survives.
        - A dead cell with exactly 3 neighbours becomes alive.
        - All other cells die or stay dead.
    """
    # Count neighbours using 2D convolution with a 3x3 kernel
    kernel = torch.tensor([[1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]], dtype=torch.float32, device=device)
    # Reshape for conv2d: (N, C, H, W) and (outC, inC, kH, kW)
    inp = grid.unsqueeze(0).unsqueeze(0)       # (1, 1, H, W)
    k = kernel.unsqueeze(0).unsqueeze(0)       # (1, 1, 3, 3)

    # Wrap edges using circular padding
    padded = torch.nn.functional.pad(inp, (1, 1, 1, 1), mode='circular')
    neighbours = torch.nn.functional.conv2d(padded, k).squeeze()  # (H, W)

    # Apply rules
    survive = (grid == 1) & ((neighbours == 2) | (neighbours == 3))
    birth = (grid == 0) & (neighbours == 3)
    grid[:] = (survive | birth).float()

    state["generation"] += 1


def grid_to_display():
    """Write simulation grid into the RGBA display tensor."""
    alive_r, alive_g, alive_b = state["alive_color"]
    dead_r, dead_g, dead_b = state["dead_color"]

    g = grid.unsqueeze(-1)  # (H, W, 1)
    display[:, :, 0] = dead_r + (alive_r - dead_r) * grid
    display[:, :, 1] = dead_g + (alive_g - dead_g) * grid
    display[:, :, 2] = dead_b + (alive_b - dead_b) * grid
    display[:, :, 3] = 1.0


# ── Seed initial state ────────────────────────────────────────────────
randomize()


@view.on_frame
def update():
    # ── Simulation ─────────────────────────────────────────────────
    if state["running"]:
        for _ in range(state["speed"]):
            step_simulation()

    # ── Update display ─────────────────────────────────────────────
    grid_to_display()


@ctrl_panel.on_frame
def draw_controls():
    ctrl_panel.text(f"Generation: {state['generation']}")
    ctrl_panel.text(f"Alive cells: {int(grid.sum().item())}")
    ctrl_panel.text(f"FPS: {view.fps:.1f}")
    ctrl_panel.separator()

    # Play / Pause
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
            _place_pulsar(GRID_H // 2, GRID_W // 2)
        if ctrl_panel.button("LWSS", width=80):
            clear()
            _place_lwss(GRID_H // 2, GRID_W // 4)

    if ctrl_panel.button("Gosper Gun", width=100):
        clear()
        _place_gosper_gun(GRID_H // 2 - 5, 1)

    ctrl_panel.separator()
    ctrl_panel.text_wrapped(
        "Click Play to start, or Step to advance one generation. "
        "Use Randomize to reset with random cells."
    )


# ── Pattern helpers ────────────────────────────────────────────────────

def _place_cells(row, col, offsets):
    """Place a pattern given a list of (dr, dc) offsets."""
    for dr, dc in offsets:
        r, c = (row + dr) % GRID_H, (col + dc) % GRID_W
        grid[r, c] = 1.0


def _place_pulsar(row, col):
    """Place a period-3 pulsar oscillator."""
    offsets = []
    for sign_r in (-1, 1):
        for sign_c in (-1, 1):
            for i in (2, 3, 4):
                offsets.append((sign_r * 1, sign_c * i))
                offsets.append((sign_r * i, sign_c * 1))
                offsets.append((sign_r * 6, sign_c * i))
                offsets.append((sign_r * i, sign_c * 6))
    _place_cells(row, col, offsets)


def _place_lwss(row, col):
    """Place a Lightweight Spaceship (LWSS)."""
    offsets = [
        (0, 1), (0, 4),
        (1, 0),
        (2, 0), (2, 4),
        (3, 0), (3, 1), (3, 2), (3, 3),
    ]
    _place_cells(row, col, offsets)


def _place_gosper_gun(row, col):
    """Place a Gosper glider gun."""
    offsets = [
        # Left block
        (4, 0), (5, 0), (4, 1), (5, 1),
        # Left structure
        (4, 10), (5, 10), (6, 10),
        (3, 11), (7, 11),
        (2, 12), (8, 12),
        (2, 13), (8, 13),
        (5, 14),
        (3, 15), (7, 15),
        (4, 16), (5, 16), (6, 16),
        (5, 17),
        # Right structure
        (2, 20), (3, 20), (4, 20),
        (2, 21), (3, 21), (4, 21),
        (1, 22), (5, 22),
        (0, 24), (1, 24), (5, 24), (6, 24),
        # Right block
        (2, 34), (3, 34), (2, 35), (3, 35),
    ]
    _place_cells(row, col, offsets)


view.run()
