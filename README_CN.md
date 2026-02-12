<div align="center">

# 🔥 Vultorch

**一行代码可视化任意 CUDA 张量**

基于 Vulkan 的 PyTorch GPU 实时张量查看器，内置 ImGui 界面。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)

[🇬🇧 English](README.md)

</div>

---

## 什么是 Vultorch？

Vultorch 通过 Vulkan 直接在屏幕上渲染 CUDA 张量 — 数据**全程不离开 GPU**。无 CPU 回读、无中转缓冲区、无 OpenGL。只需一行 Python 代码：

```python
vultorch.show(tensor)
```

同时内置 [Dear ImGui](https://github.com/ocornut/imgui)（docking 分支），滑条、折线图、按钮、可停靠窗口布局开箱即用。

## 功能

| 功能 | 说明 |
|------|------|
| **一行显示** | `vultorch.show(tensor)` — 就这么简单 |
| **GPU → GPU** | Vulkan 外部内存互操作，CPU 零参与 |
| **真零拷贝** | `vultorch.create_tensor()` 在 CUDA 与 Vulkan 之间共享显存 |
| **ImGui 内置** | 滑条、按钮、颜色选择器、折线图 — 全部用 Python 调用 |
| **Docking 布局** | 拖拽排列窗口（ImGui docking 分支） |
| **3D 场景** | 将张量映射到带光照的 3D 平面，支持轨道相机 + MSAA |
| **DLPack 互操作** | 标准 `torch.from_dlpack()` 创建共享张量 |

## 快速开始

### 安装

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

### 零拷贝张量

```python
# CUDA 与 Vulkan 共享显存 — 写入即可见
tensor = vultorch.create_tensor(256, 256, channels=4)
tensor[:, :, 0] = torch.linspace(0, 1, 256, device="cuda")  # 立即可见
```

### 3D 场景

```python
scene = vultorch.SceneView("3D", 800, 600, msaa=4)
scene.set_tensor(tensor)
scene.render()  # 轨道相机，Blinn-Phong 光照
```

## 示例

| 示例 | 说明 |
|------|------|
| [`01_hello_tensor.py`](examples/01_hello_tensor.py) | 最简张量显示 |
| [`02_imgui_controls.py`](examples/02_imgui_controls.py) | ImGui 控件展示 |
| [`03_scene_3d.py`](examples/03_scene_3d.py) | 3D 场景 + 光照 + 轨道相机 |
| [`04_docking_layout.py`](examples/04_docking_layout.py) | 可停靠窗口 + DockBuilder 布局 |
| [`05_zero_copy.py`](examples/05_zero_copy.py) | 真零拷贝共享张量 |

运行示例：
```bash
python examples/01_hello_tensor.py
```

## 从源码构建

### 前置条件

- 支持 Vulkan 的 **GPU**（NVIDIA / AMD / Intel 现代显卡）
- **Vulkan SDK** — [lunarg.com/vulkan-sdk](https://vulkan.lunarg.com/sdk/home)
- **CUDA Toolkit**（可选，用于张量显示）
- **Python 3.9+** + pip
- **CMake 3.25+** + **Ninja**

### 克隆

```bash
git clone --recursive https://github.com/vultorch/vultorch.git
cd vultorch
```

### 构建并安装

```powershell
# 为当前 Python 构建 wheel 并安装
.\build.ps1

# 或：快速开发模式（仅 cmake，不打包 wheel）
.\build.ps1 -Dev
```

或手动：

```bash
pip install .
```

### 多版本 wheel（CI）

```powershell
.\build_wheels.ps1 -Versions "3.9","3.10","3.11","3.12"
```

## 项目结构

```
vultorch/
├── src/                    # C++ 核心
│   ├── engine.cpp/h        # Vulkan + SDL3 + ImGui 引擎
│   ├── tensor_texture.*    # CUDA ↔ Vulkan 零拷贝互操作
│   ├── scene_renderer.*    # 离屏 3D 渲染器（MSAA、Blinn-Phong）
│   ├── bindings.cpp        # pybind11 绑定
│   └── shaders/            # GLSL 顶点/片段着色器
├── vultorch/               # Python 包
│   └── __init__.py         # 高层 API（Window、show、SceneView）
├── external/               # Git 子模块
│   ├── pybind11/           # C++ ↔ Python 绑定库
│   ├── SDL/                # 窗口与输入（SDL3）
│   └── imgui/              # Dear ImGui（docking 分支）
└── examples/               # 可直接运行的示例
```

## 系统要求

| 组件 | 必需 | 备注 |
|------|------|------|
| GPU | ✅ | 任何支持 Vulkan 的显卡 |
| Vulkan SDK | 仅构建时 | 运行时不需要 |
| CUDA Toolkit | 可选 | `show()` 和 `create_tensor()` 需要 |
| Python | 3.9+ | |
| PyTorch | 可选 | 张量操作需要 |

## 许可证

[MIT](LICENSE)

---

<div align="center">

**[示例](examples/) · [API 参考](vultorch/__init__.py) · [English](README.md)**

</div>
