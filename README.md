<div align="center">

# 🔥 Vultorch

**One line to visualize any CUDA tensor.**

Vulkan-based real-time GPU tensor viewer for PyTorch with built-in ImGui UI.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)

[🇨🇳 中文文档](README_CN.md)

</div>

---

## What is Vultorch?

Vultorch renders CUDA tensors directly on screen through Vulkan — the data **never leaves the GPU**. No CPU readback, no staging buffers, no OpenGL. Just one Python call:

```python
vultorch.show(tensor)
```

It also bundles [Dear ImGui](https://github.com/ocornut/imgui) (docking branch), so you get sliders, plots, buttons, and dockable window layouts for free.

## Features

| Feature | Description |
|---------|-------------|
| **One-line display** | `vultorch.show(tensor)` — that's it |
| **GPU → GPU** | Vulkan external memory interop, zero CPU involvement |
| **True zero-copy** | `vultorch.create_tensor()` shares memory between CUDA and Vulkan |
| **ImGui built-in** | Sliders, buttons, color pickers, plots — all from Python |
| **Docking layout** | Drag-and-drop window arrangement (ImGui docking branch) |
| **3D scene view** | Map a tensor onto a lit 3D plane with orbit camera + MSAA |
| **DLPack interop** | Standard `torch.from_dlpack()` for shared tensor creation |

## Quick Start

### Install

```bash
pip install vultorch
```

### Hello Tensor

```python
import torch
import vultorch
from vultorch import ui

tensor = torch.rand(256, 256, 4, device="cuda")

win = vultorch.Window("Hello", 512, 512)
while win.poll():
    if not win.begin_frame():
        continue
    ui.begin("Viewer")
    vultorch.show(tensor)
    ui.end()
    win.end_frame()
win.destroy()
```

### Zero-Copy Tensor

```python
# Memory is shared between CUDA and Vulkan — writes are instant
tensor = vultorch.create_tensor(256, 256, channels=4)
tensor[:, :, 0] = torch.linspace(0, 1, 256, device="cuda")  # visible immediately
```

### 3D Scene

```python
scene = vultorch.SceneView("3D", 800, 600, msaa=4)
scene.set_tensor(tensor)
scene.render()  # orbit camera, Blinn-Phong lighting
```

## Examples

| Example | Description |
|---------|-------------|
| [`01_hello_tensor.py`](examples/01_hello_tensor.py) | Minimal tensor display |
| [`02_imgui_controls.py`](examples/02_imgui_controls.py) | ImGui widgets showcase |
| [`03_scene_3d.py`](examples/03_scene_3d.py) | 3D scene with lighting and orbit camera |
| [`04_docking_layout.py`](examples/04_docking_layout.py) | Dockable window layout with DockBuilder |
| [`05_zero_copy.py`](examples/05_zero_copy.py) | True zero-copy shared tensor |

Run any example:
```bash
python examples/01_hello_tensor.py
```

## Building from Source

### Prerequisites

- **GPU** with Vulkan support (any modern NVIDIA / AMD / Intel)
- **Vulkan SDK** — [lunarg.com/vulkan-sdk](https://vulkan.lunarg.com/sdk/home)
- **CUDA Toolkit** (optional, for tensor display)
- **Python 3.9+** with pip
- **CMake 3.25+** and **Ninja**

### Clone

```bash
git clone --recursive https://github.com/vultorch/vultorch.git
cd vultorch
```

### Build & Install

```powershell
# Build a wheel for your current Python and install it
.\build.ps1

# Or: fast dev iteration (cmake only, no wheel)
.\build.ps1 -Dev
```

Or manually:

```bash
pip install .
```

### Multi-version wheels (CI)

```powershell
.\build_wheels.ps1 -Versions "3.9","3.10","3.11","3.12"
```

## Architecture

```
vultorch/
├── src/                    # C++ core
│   ├── engine.cpp/h        # Vulkan + SDL3 + ImGui engine
│   ├── tensor_texture.*    # CUDA ↔ Vulkan zero-copy interop
│   ├── scene_renderer.*    # Offscreen 3D renderer (MSAA, Blinn-Phong)
│   ├── bindings.cpp        # pybind11 bindings
│   └── shaders/            # GLSL vertex/fragment shaders
├── vultorch/               # Python package
│   └── __init__.py         # High-level API (Window, show, SceneView)
├── external/               # Git submodules
│   ├── pybind11/           # C++ ↔ Python bindings
│   ├── SDL/                # Window + input (SDL3)
│   └── imgui/              # Dear ImGui (docking branch)
└── examples/               # Ready-to-run demos
```

## Requirements

| Component | Required | Notes |
|-----------|----------|-------|
| GPU | ✅ | Any Vulkan-capable GPU |
| Vulkan SDK | Build only | Not needed at runtime |
| CUDA Toolkit | Optional | Required for `show()` and `create_tensor()` |
| Python | 3.9+ | |
| PyTorch | Optional | Required for tensor operations |

## License

[MIT](LICENSE)

---

<div align="center">

**[Examples](examples/) · [API Reference](vultorch/__init__.py) · [中文文档](README_CN.md)**

</div>
