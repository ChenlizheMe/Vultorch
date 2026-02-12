<div align="center">

# 🔥 Vultorch

**Real-time Torch Visualization Window · Vulkan Zero-Copy**

Visualize CUDA tensors at GPU speed — zero CPU readback, zero staging buffers.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Vulkan](https://img.shields.io/badge/Vulkan-1.2-red.svg)](https://vulkan.org)

**[🇨🇳 中文](README_CN.md) · [🌐 Website](https://ChenlizheMe.github.io/Vultorch/)**

<br>

<img src="docs/images/example.png" alt="Vultorch screenshot" width="720">

</div>

---

## Overview

Vultorch displays CUDA tensors in a native window — data never leaves the GPU.
`show()` performs a fast GPU-GPU copy; `create_tensor()` eliminates even that via Vulkan shared memory.

```python
vultorch.show(tensor)           # GPU-only, no CPU readback
tensor = vultorch.create_tensor(...)  # true zero-copy, no memcpy at all
```

## Key Features

- **GPU-only display** — `vultorch.show(tensor)` does a fast GPU-GPU copy to Vulkan, no CPU readback ever
- **True zero-copy** — `vultorch.create_tensor()` returns a torch.Tensor backed by Vulkan shared memory — zero memcpy
- **One-line API** — `vultorch.show(tensor)` handles format conversion, upload, and display
- **Built-in ImGui** — Sliders, buttons, color pickers, plots, docking layout — all from Python
- **3D scene view** — Map textures onto lit 3D planes with orbit camera, MSAA, Blinn-Phong shading
- **Docking windows** — Drag-and-drop window arrangement (ImGui docking branch)

## Quick Start

```bash
pip install vultorch
```

```python
import torch, vultorch
from vultorch import ui

# Your neural texture output (or any CUDA tensor)
texture = torch.rand(512, 512, 4, device="cuda")

win = vultorch.Window("Neural Texture Viewer", 800, 600)
while win.poll():
    if not win.begin_frame(): continue
    ui.begin("Output")
    vultorch.show(texture)  # GPU-only, no CPU round-trip
    ui.end()
    win.end_frame()
win.destroy()
```

### True Zero-Copy

```python
# Shared GPU memory — writes are instantly visible on screen
tensor = vultorch.create_tensor(512, 512, channels=4)
tensor[:] = model(input)   # write directly, no copy needed
```

### 3D Scene

```python
scene = vultorch.SceneView("3D", 800, 600, msaa=4)
scene.set_tensor(texture)
scene.render()  # orbit camera, Blinn-Phong lighting
```

## Examples

| Example | Description |
|---------|-------------|
| [`01_hello_tensor.py`](examples/01_hello_tensor.py) | Minimal tensor display |
| [`02_imgui_controls.py`](examples/02_imgui_controls.py) | ImGui widgets: sliders, plots, colors |
| [`03_scene_3d.py`](examples/03_scene_3d.py) | 3D scene with lighting + orbit camera |
| [`04_docking_layout.py`](examples/04_docking_layout.py) | Drag-and-drop dockable window layout |
| [`05_zero_copy.py`](examples/05_zero_copy.py) | True zero-copy shared tensor |

```bash
python examples/01_hello_tensor.py
```

## Building from Source

### Prerequisites

| Component | Required | Notes |
|-----------|----------|-------|
| **GPU** | ✅ | Any Vulkan-capable GPU (NVIDIA, AMD, Intel) |
| **Vulkan** | Runtime | Ships with your GPU driver — no separate install needed |
| **Vulkan SDK** | Build only | [lunarg.com/vulkan-sdk](https://vulkan.lunarg.com/sdk/home) — only for building from source |
| **CUDA Toolkit** | Optional | For `show()` and `create_tensor()` |
| **Python 3.8+** | ✅ | |
| **CMake 3.25+** | Build only | + Ninja |

### Clone & Build

```bash
git clone --recursive https://github.com/ChenlizheMe/Vultorch.git
cd Vultorch
```

**One command** — configure, compile, and produce a wheel in `dist/`:

```bash
# Windows
build.bat

# Linux
./build.sh
```

The wheel appears in `dist/`. Install it:

```bash
pip install dist/vultorch-*.whl
```

The build script auto-detects the active Python environment, so the wheel
matches whichever `python` is on your PATH (conda, venv, system, etc.).

## Architecture

```
vultorch/
├── src/                    # C++ core
│   ├── engine.cpp/h        # Vulkan + SDL3 + ImGui engine
│   ├── tensor_texture.*    # CUDA ↔ Vulkan zero-copy interop
│   ├── scene_renderer.*    # Offscreen 3D renderer (MSAA, Blinn-Phong)
│   ├── bindings.cpp        # pybind11 Python bindings
│   └── shaders/            # GLSL shaders → SPIR-V
├── vultorch/               # Python package
│   └── __init__.py         # High-level API (Window, show, SceneView)
├── external/               # Git submodules
│   ├── pybind11/           # C++ ↔ Python binding
│   ├── SDL/                # Window / input (SDL3)
│   └── imgui/              # Dear ImGui (docking branch)
├── examples/               # Ready-to-run demos
├── tools/                  # Build utilities
└── docs/                   # GitHub Pages website
```

## License

[MIT](LICENSE)

---

<div align="center">

**[Examples](examples/) · [Website](https://ChenlizheMe.github.io/Vultorch/) · [中文文档](README_CN.md)**

</div>
