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

**两条命令** — 配置并构建（在 `dist/` 中生成 wheel）：

```bash
# Windows（需要 Ninja + Vulkan SDK）
cmake --preset release-windows
cmake --build --preset release-windows

# Linux / WSL2（需要 Ninja + Vulkan 头文件）
cmake --preset release-linux
cmake --build --preset release-linux

# Linux 无 Ninja 环境
cmake --preset release-linux-make
cmake --build --preset release-linux-make
```

wheel 自动出现在 `dist/` 目录。安装：

```bash
pip install dist/vultorch-*.whl
```

构建过程自动检测当前激活的 Python 和 CUDA 环境。
如果安装了 `mkdocs`，教程文档也会一并构建。

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
│   └── __init__.py          # 高层 API（Window、show、SceneView）
├── external/                # Git 子模块
│   ├── pybind11/            # C++ ↔ Python 绑定库
│   ├── SDL/                 # 窗口与输入（SDL3）
│   └── imgui/               # Dear ImGui（docking 分支）
├── examples/                # 可直接运行的示例
├── tests/                   # pytest GPU 测试
├── tools/                   # 编译期工具（着色器头文件生成）
├── scripts/                 # 开发者脚本（多版本 wheel、上传、WSL2）
├── tutorial/                # MkDocs 源文件（Markdown）
└── docs/                    # 生成的网站（GitHub Pages）
```

## 许可证

[MIT](LICENSE)

---

<div align="center">

**[示例](examples/) · [网站](https://ChenlizheMe.github.io/Vultorch/) · [English](README.md)**

</div>
