<div align="center">

# VULTORCH

**GPU 原生 PyTorch 张量可视化**

以 GPU 速度可视化 CUDA 张量 — 零 CPU 回读、零中转缓冲。
神经渲染、强化学习、物理仿真 — 只要在张量里，Vultorch 就能显示。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Vulkan](https://img.shields.io/badge/Vulkan-1.2-red.svg)](https://vulkan.org)

**[🇬🇧 English](README.md) · [🌐 网站](https://ChenlizheMe.github.io/Vultorch/) · [📖 教程](https://ChenlizheMe.github.io/Vultorch/tutorial/)**

<br>

<img src="docs/images/example.png" alt="Vultorch 截图" width="720">

</div>

---

## 概述

Vultorch 在原生窗口中显示 CUDA 张量 — 数据全程不离开 GPU。
`show()` 执行快速 GPU-GPU 拷贝；`create_tensor()` 通过 Vulkan 共享显存连这一步都省去。

```python
vultorch.show(tensor)                     # 纯 GPU，无 CPU 回读
tensor = vultorch.create_tensor(...)      # 真零拷贝，无任何 memcpy
```

## 核心特性

- **纯 GPU 显示** — `vultorch.show(tensor)` 执行快速 GPU-GPU 拷贝至 Vulkan，绝不回读 CPU
- **真零拷贝** — `vultorch.create_tensor()` 返回由 Vulkan 共享显存支持的 torch.Tensor — 零 memcpy
- **声明式 API** — `View → Panel → Canvas`，自动布局并支持逐帧回调
- **内置 ImGui** — 滑条、按钮、颜色选择器、折线图、停靠布局 — 全部用 Python 调用
- **3D 场景** — 将纹理映射到带光照的 3D 平面，支持轨道相机 + MSAA + Blinn-Phong
- **停靠窗口** — 拖拽排列窗口（ImGui docking 分支）
- **不只是渲染** — RL 环境、元胞自动机、信号处理 — 一切基于张量的场景

## 快速开始

```bash
pip install vultorch
```

```python
import torch, vultorch

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

### 交互式训练循环

```python
view = vultorch.View("训练", 1000, 700)
ctrl = view.panel("控制", side="left", width=0.25)
output = view.panel("输出")
out_canvas = output.canvas("render")

while view.step():
    lr = 10 ** ctrl.slider("log LR", -5.0, -1.0, default=-2.0)
    loss = train_one_step(lr)
    out_canvas.bind(output_tensor)
    view.end_step()
```

### 3D 场景

```python
scene = vultorch.SceneView("3D", 800, 600, msaa=4)
scene.set_tensor(texture)
scene.render()  # 轨道相机，Blinn-Phong 光照
```

## 示例

| # | 示例 | 说明 |
|---|------|------|
| 01 | [`hello_tensor`](examples/01_hello_tensor.py) | 最简 CUDA 张量显示 |
| 02 | [`imgui_controls`](examples/02_imgui_controls.py) | ImGui 控件 + 多面板布局 |
| 03 | [`training_test`](examples/03_training_test.py) | 实时网络训练与 GT 对比 |
| 04 | [`conway`](examples/04_conway.py) | GPU 上的康威生命游戏 |
| 05 | `image_viewer` | 加载、变换、保存图片 |
| 06 | `pixel_canvas` | 交互式像素级绘画 |
| 07 | `multichannel` | RGB + 深度 + 法线 + Alpha 查看器 |
| 08 | `gt_vs_pred` | 训练对比与误差热力图 |
| 09 | `live_tuning` | 运行时超参数调节 |
| 10 | `gaussian2d` | 可微分二维高斯泼溅 |
| 11 | `3d_inspector` | 轨道相机 + Blinn-Phong 光照 |
| 12 | `neural_workstation` | 完整神经渲染工作站 |
| 13 | `snake_rl` | DQN 学习贪吃蛇 — 强化学习可视化 |

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

```bash
# Windows（MSVC）
cmake --preset release-windows

# Linux / WSL2（GCC + Make）
cmake --preset release-linux
```

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

```bash
sudo bash scripts/setup_wsl2.sh
```

---

## 打包

### 单个 Wheel

```bash
python tools/make_wheel.py
```

### 多版本 Wheel

```bash
python scripts/build_wheels.py            # 所有默认版本（3.8 – 3.12）
python scripts/build_wheels.py 3.10 3.11  # 指定版本
```

### 上传到 PyPI

```bash
python scripts/upload_wheels.py
```

---

## 测试

```bash
pytest                  # 全部测试
pytest -m "not gpu"     # 仅纯 Python 测试
pytest -m gpu           # 仅 GPU 测试
```

| 标记 | 说明 |
|------|------|
| `gpu` | 需要支持 Vulkan 的 GPU 及 CUDA |
| `slow` | 长时间运行的测试 |

---

## 文档

教程与 API 参考使用 **MkDocs Material** + **i18n** 构建（英文 + 中文）。

```bash
mkdocs build --clean    # 构建
mkdocs serve            # 在 http://127.0.0.1:8000 预览
```

---

## 项目结构

```
Vultorch/
├── src/                     # C++ 核心（Vulkan + CUDA + ImGui）
│   ├── engine.cpp/h         # Vulkan + SDL3 + ImGui 引擎
│   ├── tensor_texture.*     # CUDA ↔ Vulkan 零拷贝互操作
│   ├── scene_renderer.*     # 3D 渲染器（MSAA、Blinn-Phong）
│   ├── bindings.cpp         # pybind11 绑定
│   └── shaders/             # GLSL → SPIR-V
├── vultorch/                # Python 包
│   ├── __init__.py          # 高层 API
│   ├── app.py               # 声明式 API（View、Panel、Canvas）
│   └── *.pyi                # 类型存根
├── examples/                # 13 个可运行示例
├── tutorial/                # MkDocs 源文件（中英双语）
├── tests/                   # pytest 测试（GPU + 非 GPU）
├── external/                # pybind11、SDL3、imgui
├── tools/                   # make_wheel.py、spv_to_header.py
└── scripts/                 # build_wheels.py、upload_wheels.py
```

## 许可证

[MIT](LICENSE)

---

<div align="center">

**[示例](examples/) · [教程](https://ChenlizheMe.github.io/Vultorch/tutorial/) · [网站](https://ChenlizheMe.github.io/Vultorch/) · [English](README.md)**

</div>
