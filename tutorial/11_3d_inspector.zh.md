# 11 — 3D 表面检查器

> **示例文件:** `examples/11_3d_inspector.py`

你渲染了一张深度图、法线图或纹理。现在你想从另一个角度看看，
让光打上去，检查掠射角度下有没有瑕疵。在 matplotlib 里你得跟
`plot_surface()` 和 `set_azim()` 搏斗。在这里，你只要拖鼠标。

SceneView 是 Vultorch 的 3D 查看器组件。它接收任何 tensor，
映射到 3D 空间中的一个平面上，加上 Blinn-Phong 光照，
然后让你用鼠标拖拽来绕轨道旋转。MSAA 抗锯齿、可调 FOV、
光照方向、光泽度——全部实时。

## 新朋友

| 新东西 | 它做什么 | 为什么重要 |
|--------|---------|-----------|
| `SceneView` | 带轨道相机的 3D 平面查看器 | 用鼠标交互在 3D 中检查纹理/输出 |
| `Camera` | 方位角、仰角、距离、FOV | 绕场景旋转；可鼠标拖拽或程序控制 |
| `Light` | 方向、强度、环境光、镜面反射、光泽度 | 完全可控的 Blinn-Phong 着色 |
| `.set_tensor()` | 上传任意 tensor 到 3D 场景 | RGB、RGBA、灰度——自动扩展为 RGBA |
| `.render()` | 将场景绘制为 ImGui 图像 | 自动处理鼠标拖拽、相机同步、缩放 |
| `.msaa` | 多重采样抗锯齿（1/2/4/8） | 不同质量/性能权衡下的平滑边缘 |
| `.background` | 背景颜色元组 | 设置 3D 平面后面的清除色 |

## 我们要做什么

在 GPU 上生成四种程序化纹理——棋盘格、径向渐变、正弦图案、法线图——
每次显示一种在 3D 平面上。鼠标拖拽旋转，侧边栏控制相机、光照、
MSAA 和背景颜色。

## 完整代码

```python
import math

import torch
import vultorch
from vultorch import ui

device = "cuda"
H, W = 256, 256

ys = torch.linspace(-1, 1, H, device=device)
xs = torch.linspace(-1, 1, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")


def make_checkerboard(freq=8.0):
    check = ((xx * freq).floor() + (yy * freq).floor()) % 2
    return torch.stack([check*0.9+0.1, check*0.1+0.2, check*0.3+0.4], dim=-1)


def make_radial_gradient():
    r = (xx**2 + yy**2).sqrt()
    val = (1.0 - r).clamp(0, 1)
    return torch.stack([val, val*0.5, val*0.2], dim=-1)


def make_sine_pattern(freq=6.0):
    s1 = (torch.sin(xx * freq * math.pi) * 0.5 + 0.5)
    s2 = (torch.sin(yy * freq * math.pi) * 0.5 + 0.5)
    val = s1 * s2
    return torch.stack([val*0.2+0.1, val*0.8+0.1, val*0.5+0.3], dim=-1)


def make_normal_map():
    nx = xx * 0.5 + 0.5
    ny = yy * 0.5 + 0.5
    nz = (1.0 - xx**2 - yy**2).clamp(0, 1).sqrt() * 0.5 + 0.5
    return torch.stack([nx, ny, nz], dim=-1)


TEXTURE_NAMES = ["Checkerboard", "Radial Gradient", "Sine Pattern", "Normal Map"]
TEXTURE_FNS = [make_checkerboard, make_radial_gradient,
               make_sine_pattern, make_normal_map]

# View + 面板
view = vultorch.View("11 - 3D Surface Inspector", 1200, 800)
ctrl = view.panel("Controls", side="left", width=0.28)
scene_panel = view.panel("3D View")

# SceneView 放在面板里
scene = vultorch.SceneView("Inspector", 800, 600, msaa=4)
current_texture = make_checkerboard()


@ctrl.on_frame
def draw_controls():
    state["texture_idx"] = ctrl.combo("Texture", TEXTURE_NAMES)
    state["fov"] = ctrl.slider("FOV", 10.0, 120.0, default=45.0)
    state["distance"] = ctrl.slider("Distance", 1.0, 10.0, default=3.0)
    state["auto_rotate"] = ctrl.checkbox("Auto Rotate")

    # 光照控制
    state["light_az"] = ctrl.slider("Light Az", -3.14, 3.14)
    state["light_el"] = ctrl.slider("Light El", -3.14, 3.14)
    state["ambient"] = ctrl.slider("Ambient", 0.0, 1.0, default=0.15)
    state["specular"] = ctrl.slider("Specular", 0.0, 2.0, default=0.5)
    state["shininess"] = ctrl.slider("Shininess", 1.0, 128.0, default=32.0)

    state["msaa_idx"] = ctrl.combo("MSAA", ["1", "2", "4", "8"])
    state["bg_color"] = ctrl.color_picker("Background")

    if ctrl.button("Reset Camera"):
        scene.camera.reset()


@scene_panel.on_frame
def draw_scene():
    # 应用设置到相机、光照、背景
    scene.camera.fov = state["fov"]
    scene.camera.distance = state["distance"]
    if state["auto_rotate"]:
        scene.camera.azimuth += 0.02

    scene.light.direction = (cos(el)*sin(az), sin(el), cos(el)*cos(az))
    scene.light.ambient = state["ambient"]
    scene.light.specular = state["specular"]
    scene.light.shininess = state["shininess"]
    scene.msaa = msaa_val
    scene.background = state["bg_color"]

    scene.set_tensor(current_texture)
    scene.render()  # 在这里绘制 3D 视图


view.run()
```

*(精简版——完整代码见 `examples/11_3d_inspector.py`。)*

## 刚才发生了什么？

### SceneView — 面板里的 3D

SceneView 是一个自包含的 3D 查看器组件。创建一次，然后每帧调用
`set_tensor()` 和 `render()`：

```python
scene = vultorch.SceneView("Inspector", 800, 600, msaa=4)

# 在面板回调中：
scene.set_tensor(my_tensor)  # 上传纹理
scene.render()               # 渲染 + 显示 + 处理鼠标
```

`render()` 做所有事情：把相机/光照设置推送到 GPU，离屏渲染场景，
作为 ImGui 图像显示，处理鼠标拖拽的轨道/平移/缩放，
然后把相机状态拉回来让 Python 看到更新后的方位角/仰角。

### Camera — 用鼠标数学旋转

相机由 5 个值定义：

| 属性 | 默认值 | 控制什么 |
|------|--------|---------|
| `azimuth` | 0.0 | 水平旋转（弧度） |
| `elevation` | 0.6 | 垂直旋转（弧度） |
| `distance` | 3.0 | 到目标点的距离 |
| `target` | (0,0,0) | 注视点 |
| `fov` | 45.0 | 视场角（度） |

左键拖拽旋转（方位角 + 仰角），右键拖拽平移（目标点），
中键拖拽/滚轮缩放（距离）。全部内置——不需要写代码。

也可以程序化设置：

```python
scene.camera.fov = 90.0       # 广角镜头
scene.camera.distance = 5.0   # 远离
scene.camera.azimuth += 0.02  # 自动旋转
```

### Light — Blinn-Phong 着色

光源是带 Blinn-Phong 着色的方向光：

```python
scene.light.direction = (0.3, -1.0, 0.5)  # 方向向量
scene.light.intensity = 1.0                # 整体亮度
scene.light.ambient = 0.15                 # 填充光
scene.light.specular = 0.5                 # 高光强度
scene.light.shininess = 32.0               # 高光锐度
```

低环境光 + 高镜面反射 = 戏剧性的、高对比度的外观。
高环境光 + 低镜面反射 = 平坦的、均匀照明的外观。
交互式调整这些参数可以帮你发现只在特定光照角度才显现的表面瑕疵。

### MSAA — 抗锯齿质量

MSAA（多重采样抗锯齿）让锯齿边缘变平滑：

| MSAA | 每像素采样 | 质量 | 性能 |
|------|-----------|------|------|
| 1 | 1 | 有锯齿 | 最快 |
| 2 | 2 | 略微平滑 | 快 |
| 4 | 4 | 平滑 | 中等 |
| 8 | 8 | 非常平滑 | 最慢 |

```python
scene.msaa = 4  # 好的默认值
```

大多数可视化工作中，4× MSAA 是最佳平衡点。需要最大帧率时降到 1，
截图时上到 8。

### GPU 上的程序化纹理

四种纹理全部是纯 PyTorch tensor 运算——没有 CPU，没有 PIL，
没有文件 I/O：

```python
def make_checkerboard(freq=8.0):
    check = ((xx * freq).floor() + (yy * freq).floor()) % 2
    return torch.stack([check*0.9+0.1, check*0.1+0.2, check*0.3+0.4], dim=-1)
```

这很重要，因为在真实的神经渲染管线中，你在这里显示的 tensor
会是你的 NeRF 渲染输出、3DGS 的深度图、或扩散模型生成的纹理。
SceneView 不关心 tensor 从哪来——它只负责显示。

## 核心要点

1. **SceneView** = 3D 平面查看器。上传 tensor，渲染，鼠标拖拽旋转。
   就这么简单。

2. **Camera** 有方位角/仰角/距离/FOV——鼠标拖拽或 Python 代码设置。
   `camera.reset()` 回到默认值。

3. **Light** 是 Blinn-Phong：方向、强度、环境光、镜面反射、光泽度。
   交互式光照揭示表面瑕疵。

4. **MSAA** 从 1（快，有锯齿）到 8（慢，平滑）。日常工作用 4。

5. **任何 tensor 都行** — RGB、RGBA、灰度。SceneView 内部自动扩展为 RGBA。

!!! tip "小贴士"
    用自动旋转（`scene.camera.azimuth += 0.02`）快速扫描表面
    找瑕疵——从默认角度看不见的问题，旋转起来就一目了然。

!!! note "注意"
    SceneView 渲染的是一个平面。要看真正的 3D 几何体（网格、点云），
    需要扩展 C++ 渲染器。但对于检查逐像素输出（深度图、法线图、纹理），
    一个带光照和轨道相机的 3D 平面正好够用。
