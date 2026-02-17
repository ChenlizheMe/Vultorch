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
- **Declarative API** — `View → Panel → Canvas` with auto layout and per-frame callback support
- **Built-in ImGui** — Sliders, buttons, color pickers, plots, docking layout — all from Python
- **3D scene view** — Map textures onto lit 3D planes with orbit camera, MSAA, Blinn-Phong shading
- **Docking windows** — Drag-and-drop window arrangement (ImGui docking branch)

## Quick Start

```bash
pip install vultorch
```

```python
import torch, vultorch

# Your neural texture output (or any CUDA tensor)
texture = torch.rand(512, 512, 4, device="cuda")

view = vultorch.View("Neural Texture Viewer", 800, 600)
panel = view.panel("Output")
panel.canvas("main").bind(texture)
view.run()
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
| [`02_imgui_controls.py`](examples/02_imgui_controls.py) | Multi-panel layout with docking |
| [`03_training_test.py`](examples/03_training_test.py) | Tiny network live training (GT vs prediction + bottom info panel) |

```bash
python examples/01_hello_tensor.py
```

---

## Building from Source

### Prerequisites

| Component | Required | Notes |
|-----------|----------|-------|
| **GPU** | ✅ | Any Vulkan-capable GPU (NVIDIA, AMD, Intel) |
| **Vulkan SDK** | ✅ Build | [lunarg.com/vulkan-sdk](https://vulkan.lunarg.com/sdk/home) — headers + glslangValidator |
| **Vulkan driver** | ✅ Runtime | Ships with your GPU driver |
| **CUDA Toolkit** | Optional | Enables GPU zero-copy in `show()` / `create_tensor()` |
| **Python 3.8+** | ✅ | With development headers (`python3-dev` on Linux) |
| **CMake 3.25+** | ✅ Build | |

### Step 1 — Clone (with submodules)

```bash
git clone --recursive https://github.com/ChenlizheMe/Vultorch.git
cd Vultorch
```

> If you forgot `--recursive`, run: `git submodule update --init --recursive`

### Step 2 — Configure

Choose the preset matching your platform:

```bash
# Windows (MSVC)
cmake --preset release-windows

# Linux / WSL2 (GCC + Make)
cmake --preset release-linux
```

CMake automatically detects your **active Python interpreter** and **CUDA toolkit** (if installed).

### Step 3 — Build

```bash
cmake --build --preset release-windows    # or release-linux
```

This executes three targets in order:

1. **`_vultorch`** — Compiles the C++ extension module (`.pyd` / `.so`) and SPIR-V shaders.
2. **`package_wheel`** — Runs `tools/make_wheel.py` to produce a pip-installable `.whl` in `dist/`.
3. **`docs`** *(optional)* — If `mkdocs` is installed, builds tutorial + API docs into `docs/tutorial/`.

### Step 4 — Install

```bash
pip install dist/vultorch-*.whl
```

Verify:

```python
python -c "import vultorch; print(vultorch.__version__, 'CUDA:', vultorch.HAS_CUDA)"
```

### WSL2 Quick Setup

A one-command setup script for Ubuntu WSL2:

```bash
sudo bash scripts/setup_wsl2.sh
```

This installs all system dependencies (CMake, Vulkan headers, SDL2 dev libs, Python dev).

---

## Packaging

### Single Wheel

The build already produces a wheel in `dist/`. You can also manually run:

```bash
python tools/make_wheel.py
```

This reads the compiled `_vultorch.*.pyd` / `.so` from `vultorch/`, bundles it with the Python package files, and outputs a platform-specific `.whl` to `dist/`.

### Multi-Version Wheels

Build wheels for multiple Python versions (requires conda):

```bash
# All default versions (3.8 – 3.12)
python scripts/build_wheels.py

# Specific versions
python scripts/build_wheels.py 3.10 3.11 3.12
```

Each version gets a separate conda environment; CMake is re-configured and rebuilt for each.

### Upload to PyPI

```bash
# Interactive token prompt
python scripts/upload_wheels.py

# Pass token directly
python scripts/upload_wheels.py --token pypi-YOUR_TOKEN
```

Requires `twine` (auto-installed if missing).

---

## Testing

Tests use **pytest** with two custom markers:

| Marker | Description |
|--------|-------------|
| `gpu` | Requires a Vulkan-capable GPU with CUDA |
| `slow` | Long-running tests |

### Run All Tests

```bash
pytest
```

### Run Only Non-GPU (Pure Python) Tests

```bash
pytest -m "not gpu"
```

### Run GPU Tests Only

```bash
pytest -m gpu
```

### Run with Verbose Output

```bash
pytest -ra -v
```

### Test Structure

| File | Scope |
|------|-------|
| `tests/conftest.py` | Shared fixtures + skip decorators |
| `tests/test_import.py` | Package import, version, module structure |
| `tests/test_camera_light.py` | Camera / Light data classes |
| `tests/test_normalize_tensor.py` | `_normalize_tensor()` dtype, shape, contiguity |
| `tests/test_show.py` | `show()` / `create_tensor()` error paths |
| `tests/test_declarative_api.py` | Canvas / Panel / View / RowContext (non-GPU) |
| `tests/test_edge_cases.py` | Edge-case and error-path coverage |
| `tests/test_ui_bindings.py` | All `vultorch.ui.*` functions exist |
| `tests/test_project_structure.py` | Type stubs, configs, examples, tutorials |
| `tests/test_tools_spv_to_header.py` | `tools/spv_to_header.py` |
| `tests/test_tools_make_wheel.py` | `tools/make_wheel.py` |
| `tests/test_scripts.py` | `scripts/build_wheels.py` + `upload_wheels.py` |
| `tests/test_gpu_integration.py` | GPU: Window, show(), create_tensor(), SceneView, ImGui |
| `tests/test_engine_bindings.py` | GPU: C++ Engine class bindings |
| `tests/test_panel_widgets_gpu.py` | GPU: Panel widgets + Canvas in real render loop |

---

## Documentation

### Tutorial & API Reference

Documentation is built with **MkDocs Material** and the **i18n** plugin (English + Chinese).
Source files live in `tutorial/`:

```
tutorial/
├── index.md / index.zh.md          # Home page
├── 01_hello_tensor.md / .zh.md     # Tutorial: Hello Tensor
├── 02_multi_panel.md / .zh.md      # Tutorial: Multi-Panel
├── 03_training_test.md / .zh.md    # Tutorial: Training Test
└── api.md / api.zh.md              # API Reference
```

### When Are Docs Generated?

Docs are built **automatically** as the last build target (after `package_wheel`), provided:

1. `mkdocs` is installed: `pip install mkdocs-material mkdocs-static-i18n`
2. `VULTORCH_BUILD_DOCS=ON` (default)

The generated site lands in `docs/tutorial/` (served via GitHub Pages).

### Build Docs Manually

```bash
mkdocs build --clean
```

### Preview Docs Locally

```bash
mkdocs serve
```

Opens at `http://127.0.0.1:8000/`.

### Disable Doc Build

```bash
cmake --preset release-windows -DVULTORCH_BUILD_DOCS=OFF
```

---

## Architecture

```
Vultorch/
├── CMakeLists.txt          # Build system (compile + wheel + docs)
├── CMakePresets.json        # Cross-platform build presets
├── pyproject.toml           # Python package metadata
├── src/                     # C++ core
│   ├── engine.cpp/h         # Vulkan + SDL3 + ImGui engine
│   ├── tensor_texture.*     # CUDA ↔ Vulkan zero-copy interop
│   ├── scene_renderer.*     # Offscreen 3D renderer (MSAA, Blinn-Phong)
│   ├── bindings.cpp         # pybind11 Python bindings
│   └── shaders/             # GLSL shaders → SPIR-V
├── vultorch/                # Python package
│   ├── __init__.py          # High-level API (Window, show, SceneView)
│   ├── app.py               # Declarative API (View, Panel, Canvas)
│   ├── __init__.pyi         # Type stubs
│   ├── ui.pyi               # ImGui binding stubs
│   └── py.typed             # PEP 561 marker
├── external/                # Git submodules
│   ├── pybind11/            # C++ ↔ Python binding
│   ├── SDL/                 # Window / input (SDL3)
│   └── imgui/               # Dear ImGui (docking branch)
├── examples/                # Ready-to-run demos
├── tests/                   # pytest tests (GPU + non-GPU)
├── tools/                   # Build-time utilities
│   ├── make_wheel.py        # Wheel packaging
│   └── spv_to_header.py     # SPIR-V → C header
├── scripts/                 # Developer scripts
│   ├── build_wheels.py      # Multi-Python wheel builder
│   ├── upload_wheels.py     # PyPI upload via twine
│   └── setup_wsl2.sh        # WSL2 environment setup
├── tutorial/                # MkDocs source (Markdown, EN + ZH)
└── docs/                    # Generated website (GitHub Pages)
```

## License

[MIT](LICENSE)

---

<div align="center">

**[Examples](examples/) · [API Docs](https://ChenlizheMe.github.io/Vultorch/tutorial/api/) · [Website](https://ChenlizheMe.github.io/Vultorch/) · [中文文档](README_CN.md)**

</div>
