<div align="center">

# 🔥 Vultorch

**实时 Torch 可视化窗口 · Vulkan 零拷贝**

以 GPU 速度可视化 CUDA 张量 — 零 CPU 回读、零中转缓冲。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Vulkan](https://img.shields.io/badge/Vulkan-1.2-red.svg)](https://vulkan.org)

**[🇬🇧 English](README.md) · [🌐 网站](https://ChenlizheMe.github.io/Vultorch/)**

<br>

<img src="docs/images/example.png" alt="Vultorch 截图" width="720">

</div>

---

## 概述

Vultorch 在原生窗口中显示 CUDA 张量 — 数据全程不离开 GPU。
`show()` 执行快速 GPU-GPU 拷贝；`create_tensor()` 通过 Vulkan 共享显存连这一步都省去。

```python
vultorch.show(tensor)           # 纯 GPU，无 CPU 回读
tensor = vultorch.create_tensor(...)  # 真零拷贝，无任何 memcpy
```

## 核心特性

- **纯 GPU 显示** — `vultorch.show(tensor)` 执行快速 GPU-GPU 拷贝至 Vulkan，绝不回读 CPU
- **真零拷贝** — `vultorch.create_tensor()` 返回由 Vulkan 共享显存支持的 torch.Tensor — 零 memcpy
- **声明式 API** — `View → Panel → Canvas`，自动布局并支持逐帧回调
- **内置 ImGui** — 滑条、按钮、颜色选择器、折线图、停靠布局 — 全部用 Python 调用
- **3D 场景** — 将纹理映射到带光照的 3D 平面，支持轨道相机 + MSAA + Blinn-Phong
- **停靠窗口** — 拖拽排列窗口（ImGui docking 分支）

## 快速开始

```bash
pip install vultorch
```

```python
import torch, vultorch

# 你的神经纹理输出（或任意 CUDA 张量）
texture = torch.rand(512, 512, 4, device="cuda")

view = vultorch.View("查看器", 800, 600)
panel = view.panel("输出")
panel.canvas("main").bind(texture)
view.run()
```

### 真正的零拷贝

```python
# CUDA 与 Vulkan 共享显存 — 写入即可见
tensor = vultorch.create_tensor(512, 512, channels=4)
tensor[:] = model(input)   # 直接写入，无需拷贝
```

### 3D 场景

```python
scene = vultorch.SceneView("3D", 800, 600, msaa=4)
scene.set_tensor(texture)
scene.render()  # 轨道相机，Blinn-Phong 光照
```

## 示例

| 示例 | 说明 |
|------|------|
| [`01_hello_tensor.py`](examples/01_hello_tensor.py) | 最简张量显示 |
| [`02_imgui_controls.py`](examples/02_imgui_controls.py) | 多面板停靠布局 |
| [`03_training_test.py`](examples/03_training_test.py) | 轻量网络实时训练（GT vs 预测 + 下方面板信息） |
| [`04_conway.py`](examples/04_conway.py) | 交互式康威生命游戏（GPU 零拷贝、按钮、颜色选择器） |

```bash
python examples/01_hello_tensor.py
```

---

## 从源码构建

### 前置条件

| 组件 | 必需 | 备注 |
|------|------|------|
| **GPU** | ✅ | 任何支持 Vulkan 的显卡（NVIDIA、AMD、Intel） |
| **Vulkan SDK** | ✅ 构建时 | [lunarg.com/vulkan-sdk](https://vulkan.lunarg.com/sdk/home) — 头文件 + glslangValidator |
| **Vulkan 驱动** | ✅ 运行时 | 随 GPU 驱动自带 |
| **CUDA Toolkit** | 可选 | 启用 `show()` / `create_tensor()` 的 GPU 零拷贝 |
| **Python 3.8+** | ✅ | 包含开发头文件（Linux 需 `python3-dev`） |
| **CMake 3.25+** | ✅ 构建时 | |

### 步骤 1 — 克隆（含子模块）

```bash
git clone --recursive https://github.com/ChenlizheMe/Vultorch.git
cd Vultorch
```

> 如果忘记 `--recursive`，执行：`git submodule update --init --recursive`

### 步骤 2 — 配置

选择与你的平台匹配的预设：

```bash
# Windows（MSVC）
cmake --preset release-windows

# Linux / WSL2（GCC + Make）
cmake --preset release-linux
```

CMake 自动检测当前 **活跃的 Python 解释器** 和 **CUDA 工具链**（如已安装）。

### 步骤 3 — 编译

```bash
cmake --build --preset release-windows    # 或 release-linux
```

此命令依次执行三个构建目标：

1. **`_vultorch`** — 编译 C++ 扩展模块（`.pyd` / `.so`）及 SPIR-V 着色器。
2. **`package_wheel`** — 运行 `tools/make_wheel.py`，在 `dist/` 中生成可 pip 安装的 `.whl`。
3. **`docs`**（可选） — 如果安装了 `mkdocs`，将教程 + API 文档构建到 `docs/tutorial/`。

### 步骤 4 — 安装

```bash
pip install dist/vultorch-*.whl
```

验证：

```python
python -c "import vultorch; print(vultorch.__version__, 'CUDA:', vultorch.HAS_CUDA)"
```

### WSL2 快速搭建

Ubuntu WSL2 一键配置脚本：

```bash
sudo bash scripts/setup_wsl2.sh
```

该脚本安装所有系统依赖（CMake、Vulkan 头文件、SDL2 开发库、Python 开发包）。

---

## 打包

### 单个 Wheel

构建过程已在 `dist/` 中生成 wheel。也可手动运行：

```bash
python tools/make_wheel.py
```

该脚本读取 `vultorch/` 中已编译的 `_vultorch.*.pyd` / `.so`，与 Python 包文件一同打包，输出平台特定的 `.whl` 到 `dist/`。

### 多版本 Wheel

为多个 Python 版本构建 wheel（需要 conda）：

```bash
# 所有默认版本（3.8 – 3.12）
python scripts/build_wheels.py

# 指定版本
python scripts/build_wheels.py 3.10 3.11 3.12
```

每个版本使用单独的 conda 环境；CMake 为每个版本重新配置和构建。

### 上传到 PyPI

```bash
# 交互式输入 API Token
python scripts/upload_wheels.py

# 直接传入 Token
python scripts/upload_wheels.py --token pypi-YOUR_TOKEN
```

需要 `twine`（缺失时自动安装）。

---

## 测试

测试使用 **pytest**，包含两个自定义标记（marker）：

| 标记 | 说明 |
|------|------|
| `gpu` | 需要支持 Vulkan 的 GPU 及 CUDA |
| `slow` | 长时间运行的测试 |

### 运行全部测试

```bash
pytest
```

### 只运行非 GPU（纯 Python）测试

```bash
pytest -m "not gpu"
```

### 只运行 GPU 测试

```bash
pytest -m gpu
```

### 详细输出

```bash
pytest -ra -v
```

### 测试文件结构

| 文件 | 覆盖范围 |
|------|----------|
| `tests/conftest.py` | 共享 fixtures + skip 装饰器 |
| `tests/test_import.py` | 包导入、版本、模块结构 |
| `tests/test_camera_light.py` | Camera / Light 数据类 |
| `tests/test_normalize_tensor.py` | `_normalize_tensor()` dtype、shape、连续性 |
| `tests/test_show.py` | `show()` / `create_tensor()` 错误路径 |
| `tests/test_declarative_api.py` | Canvas / Panel / View / RowContext（非 GPU） |
| `tests/test_edge_cases.py` | 边界情况与错误路径覆盖 |
| `tests/test_ui_bindings.py` | 所有 `vultorch.ui.*` 函数存在性 |
| `tests/test_project_structure.py` | 类型存根、配置文件、示例、教程 |
| `tests/test_tools_spv_to_header.py` | `tools/spv_to_header.py` |
| `tests/test_tools_make_wheel.py` | `tools/make_wheel.py` |
| `tests/test_scripts.py` | `scripts/build_wheels.py` + `upload_wheels.py` |
| `tests/test_gpu_integration.py` | GPU：Window、show()、create_tensor()、SceneView、ImGui |
| `tests/test_engine_bindings.py` | GPU：C++ Engine 类绑定 |
| `tests/test_panel_widgets_gpu.py` | GPU：Panel 控件 + Canvas 渲染循环 |

---

## 文档

### 教程与 API 参考

文档使用 **MkDocs Material** 和 **i18n** 插件构建（英文 + 中文）。
源文件位于 `tutorial/`：

```
tutorial/
├── index.md / index.zh.md          # 首页
├── 01_hello_tensor.md / .zh.md     # 教程：Hello Tensor
├── 02_multi_panel.md / .zh.md      # 教程：多面板
├── 03_training_test.md / .zh.md    # 教程：训练测试
└── api.md / api.zh.md              # API 参考文档
```

### 文档何时生成？

文档在最后一个构建目标（`package_wheel` 之后）**自动构建**，前提条件：

1. 已安装 `mkdocs`：`pip install mkdocs-material mkdocs-static-i18n`
2. `VULTORCH_BUILD_DOCS=ON`（默认开启）

生成的网站位于 `docs/tutorial/`（通过 GitHub Pages 托管）。

### 手动构建文档

```bash
mkdocs build --clean
```

### 本地预览文档

```bash
mkdocs serve
```

在 `http://127.0.0.1:8000/` 打开。

### 禁用文档构建

```bash
cmake --preset release-windows -DVULTORCH_BUILD_DOCS=OFF
```

---

## 项目结构

```
Vultorch/
├── CMakeLists.txt          # 构建系统（编译 + wheel + 文档）
├── CMakePresets.json        # 跨平台构建预设
├── pyproject.toml           # Python 包元数据
├── src/                     # C++ 核心
│   ├── engine.cpp/h         # Vulkan + SDL3 + ImGui 引擎
│   ├── tensor_texture.*     # CUDA ↔ Vulkan 零拷贝互操作
│   ├── scene_renderer.*     # 离屏 3D 渲染器（MSAA、Blinn-Phong）
│   ├── bindings.cpp         # pybind11 Python 绑定
│   └── shaders/             # GLSL 着色器 → SPIR-V
├── vultorch/                # Python 包
│   ├── __init__.py          # 高层 API（Window、show、SceneView）
│   ├── app.py               # 声明式 API（View、Panel、Canvas）
│   ├── __init__.pyi         # 类型存根
│   ├── ui.pyi               # ImGui 绑定存根
│   └── py.typed             # PEP 561 标记
├── external/                # Git 子模块
│   ├── pybind11/            # C++ ↔ Python 绑定库
│   ├── SDL/                 # 窗口与输入（SDL3）
│   └── imgui/               # Dear ImGui（docking 分支）
├── examples/                # 可直接运行的示例
├── tests/                   # pytest 测试（GPU + 非 GPU）
├── tools/                   # 构建期工具
│   ├── make_wheel.py        # Wheel 打包
│   └── spv_to_header.py     # SPIR-V → C 头文件
├── scripts/                 # 开发者脚本
│   ├── build_wheels.py      # 多 Python 版本 wheel 构建
│   ├── upload_wheels.py     # 通过 twine 上传 PyPI
│   └── setup_wsl2.sh        # WSL2 环境配置
├── tutorial/                # MkDocs 源文件（Markdown，中英双语）
└── docs/                    # 生成的网站（GitHub Pages）
```

## 许可证

[MIT](LICENSE)

---

<div align="center">

**[示例](examples/) · [API 文档](https://ChenlizheMe.github.io/Vultorch/tutorial/api/) · [网站](https://ChenlizheMe.github.io/Vultorch/) · [English](README.md)**

</div>
