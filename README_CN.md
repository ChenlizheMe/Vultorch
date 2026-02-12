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

Vultorch 通过 PyTorch 与 Vulkan 共享 GPU 显存，在原生窗口中显示 CUDA 张量。数据全程留在 GPU 上 — 无需 `tensor.cpu()` 回读，无中转缓冲拷贝。

```python
vultorch.show(tensor)   # 零拷贝，亚毫秒
```

## 核心特性

- **零拷贝显示** — Vulkan 外部内存互操作，数据全程不离开 GPU
- **真正的共享显存** — `vultorch.create_tensor()` 返回由 Vulkan 内存支持的 torch.Tensor（DLPack）
- **一行 API** — `vultorch.show(tensor)` 自动处理格式转换、上传和显示
- **内置 ImGui** — 滑条、按钮、颜色选择器、折线图、停靠布局 — 全部用 Python 调用
- **3D 场景** — 将纹理映射到带光照的 3D 平面，支持轨道相机 + MSAA + Blinn-Phong
- **停靠窗口** — 拖拽排列窗口（ImGui docking 分支）

## 快速开始

```bash
pip install vultorch
```

```python
import torch, vultorch
from vultorch import ui

# 你的神经纹理输出（或任意 CUDA 张量）
texture = torch.rand(512, 512, 4, device="cuda")

win = vultorch.Window("查看器", 800, 600)
while win.poll():
    if not win.begin_frame(): continue
    ui.begin("输出")
    vultorch.show(texture)  # 零拷贝 GPU → 屏幕
    ui.end()
    win.end_frame()
win.destroy()
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
| [`02_imgui_controls.py`](examples/02_imgui_controls.py) | ImGui 控件：滑条、折线图、颜色 |
| [`03_scene_3d.py`](examples/03_scene_3d.py) | 3D 场景 + 光照 + 轨道相机 |
| [`04_docking_layout.py`](examples/04_docking_layout.py) | 拖拽式停靠窗口布局 |
| [`05_zero_copy.py`](examples/05_zero_copy.py) | 真零拷贝共享张量 |

```bash
python examples/01_hello_tensor.py
```

## 从源码构建

### 前置条件

| 组件 | 必需 | 备注 |
|------|------|------|
| **GPU** | ✅ | 任何支持 Vulkan 的显卡（NVIDIA、AMD、Intel） |
| **Vulkan** | 运行时 | 随 GPU 驱动自带 — 无需单独安装 |
| **Vulkan SDK** | 仅构建 | [lunarg.com/vulkan-sdk](https://vulkan.lunarg.com/sdk/home) — 仅从源码构建时需要 |
| **CUDA Toolkit** | 可选 | `show()` 和 `create_tensor()` 需要 |
| **Python 3.8+** | ✅ | |
| **CMake 3.25+** | 仅构建 | + Ninja |

### 克隆与构建

```bash
git clone --recursive https://github.com/ChenlizheMe/Vultorch.git
cd Vultorch
```

**一条命令** — 配置、编译、在 `dist/` 中生成 wheel：

```bash
# Windows
build.bat

# Linux
./build.sh
```

wheel 自动出现在 `dist/` 目录。安装：

```bash
pip install dist/vultorch-*.whl
```

构建脚本自动检测当前激活的 Python 环境，生成的 wheel
与 PATH 上的 `python` 一致（conda / venv / 系统均可）。

## 项目结构

```
vultorch/
├── src/                    # C++ 核心
│   ├── engine.cpp/h        # Vulkan + SDL3 + ImGui 引擎
│   ├── tensor_texture.*    # CUDA ↔ Vulkan 零拷贝互操作
│   ├── scene_renderer.*    # 离屏 3D 渲染器（MSAA、Blinn-Phong）
│   ├── bindings.cpp        # pybind11 Python 绑定
│   └── shaders/            # GLSL 着色器 → SPIR-V
├── vultorch/               # Python 包
│   └── __init__.py         # 高层 API（Window、show、SceneView）
├── external/               # Git 子模块
│   ├── pybind11/           # C++ ↔ Python 绑定库
│   ├── SDL/                # 窗口与输入（SDL3）
│   └── imgui/              # Dear ImGui（docking 分支）
├── examples/               # 可直接运行的示例
├── tools/                  # 构建工具
└── docs/                   # GitHub Pages 网站
```

## 许可证

[MIT](LICENSE)

---

<div align="center">

**[示例](examples/) · [网站](https://ChenlizheMe.github.io/Vultorch/) · [English](README.md)**

</div>
