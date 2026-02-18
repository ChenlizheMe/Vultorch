# 07 — 多通道查看器

> **示例文件：** `examples/07_multichannel.py`

如果你做神经渲染 —— NeRF、3D Gaussian Splatting、或者下一个缩写词 ——
你的模型不只是产出一张漂亮的图。它同时产出 RGB、depth、normal、alpha。
每一个像素都是。

开发阶段你需要同时看到所有这些。标准流程：存四张 PNG，在四个 matplotlib
窗口里打开，歪着头左右对比，发现深度图上下颠倒，重新存、重新开、
反复直到怀疑人生。

本章用四个零拷贝面板替代上述流程，60 fps 同步刷新。

## 新朋友

| 新东西 | 干什么用 | 为什么重要 |
|--------|----------|-----------|
| 四次 `create_tensor` | 同一个窗口里用四个独立的 GPU 共享纹理 | 每个输出通道都有自己的实时显示 |
| Turbo 色图 | 把 `[0, 1]` 标量 tensor 映射为彩色 `(H, W, 3)` 图像 | 深度等标量字段在灰度下几乎不可见；turbo 让结构一目了然 |
| 法线→RGB 映射 | `normal * 0.5 + 0.5`，把 `[-1, 1]` 法线变成 `[0, 1]` 颜色 | 标准惯例：X→红, Y→绿, Z→蓝 |
| 光线-球体求交 | 用 ~30 行 PyTorch 实现的全 GPU 程序化渲染 | 证明任何 GPU 计算都可以直接接入 Vultorch |

## 这次玩什么

用程序化的光线-球体渲染器产出四个实时输出，加一个控制侧边栏。
所有计算 —— 光线、求交、着色、色图 —— 都跑在 GPU 上。
四个显示 tensor 全部零拷贝，什么都不经过 CPU。

## 完整代码

```python
import math

import torch
import torch.nn.functional as F
import vultorch

if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"

H, W = 256, 256

view = vultorch.View("07 - Multi-Channel Viewer", 512, 1024)
ctrl = view.panel("Controls", side="left", width=0.20)
rgb_panel = view.panel("RGB")
depth_panel = view.panel("Depth")
normal_panel = view.panel("Normal")
alpha_panel = view.panel("Alpha")

# 四个零拷贝显示 tensor
rgb_tensor = vultorch.create_tensor(H, W, 4, device, name="rgb",
                                     window=view.window)
depth_tensor = vultorch.create_tensor(H, W, 4, device, name="depth",
                                       window=view.window)
normal_tensor = vultorch.create_tensor(H, W, 4, device, name="normal",
                                        window=view.window)
alpha_tensor = vultorch.create_tensor(H, W, 4, device, name="alpha",
                                       window=view.window)

rgb_panel.canvas("rgb").bind(rgb_tensor)
depth_panel.canvas("depth").bind(depth_tensor)
normal_panel.canvas("normal").bind(normal_tensor)
alpha_panel.canvas("alpha").bind(alpha_tensor)

# --- Turbo 色图 LUT（256 个条目，只建一次）---
_turbo_data = [
    (0.18995, 0.07176, 0.23217), (0.22500, 0.16354, 0.45096),
    # ...（32 个关键色，插值到 256 条）
]
TURBO_LUT = ...  # 完整 LUT 构建见源文件

def apply_turbo(values):
    """把 [0,1] 浮点 tensor (H,W) 映射为 (H,W,3) turbo 色彩。"""
    idx = (values.clamp(0, 1) * 255).long()
    return TURBO_LUT[idx]

# 预计算光线方向
ys = torch.linspace(1, -1, H, device=device)
xs = torch.linspace(-1, 1, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")

state = {"sphere_r": 0.6, "light_az": 0.5, "light_el": 0.8,
         "ambient": 0.1, "bg_r": 0.12, "bg_g": 0.12, "bg_b": 0.14}


def render_sphere():
    r = state["sphere_r"]
    ray_o = torch.tensor([0.0, 0.0, -2.0], device=device)
    ray_d = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    ray_d = ray_d / ray_d.norm(dim=-1, keepdim=True)

    # 光线-球体求交的二次方程
    b = 2.0 * (ray_d * ray_o).sum(-1)
    c_val = (ray_o * ray_o).sum() - r * r
    disc = b * b - 4.0 * c_val
    hit = disc > 0

    t = (-b - torch.sqrt(disc.clamp(min=0))) / 2.0
    t = t.clamp(min=0)
    point = ray_o + t.unsqueeze(-1) * ray_d
    normal = point / (point.norm(dim=-1, keepdim=True) + 1e-8)

    # Lambert 着色
    az, el = state["light_az"], state["light_el"]
    light_dir = torch.tensor([math.cos(el)*math.sin(az),
                               math.sin(el),
                               math.cos(el)*math.cos(az)], device=device)
    light_dir = light_dir / light_dir.norm()
    shade = state["ambient"] + (1 - state["ambient"]) * \
            (normal * light_dir).sum(-1).clamp(min=0)

    bg = torch.tensor([state["bg_r"], state["bg_g"], state["bg_b"]],
                      device=device)

    # RGB
    rgb = torch.where(hit.unsqueeze(-1),
                      shade.unsqueeze(-1) * torch.ones(1,1,3, device=device), bg)
    rgb_tensor[:,:,:3] = rgb;  rgb_tensor[:,:,3] = 1.0

    # Depth（turbo 色图）
    depth_raw = t * hit.float()
    d_min = depth_raw[hit].min() if hit.any() else torch.tensor(0.0)
    d_max = depth_raw[hit].max() if hit.any() else torch.tensor(1.0)
    depth_norm = ((depth_raw - d_min) / (d_max - d_min + 1e-8)).clamp(0,1)
    depth_color = torch.where(hit.unsqueeze(-1), apply_turbo(depth_norm), bg)
    depth_tensor[:,:,:3] = depth_color;  depth_tensor[:,:,3] = 1.0

    # 法线 ([-1,1] → [0,1])
    nc = torch.where(hit.unsqueeze(-1), normal * 0.5 + 0.5, bg)
    normal_tensor[:,:,:3] = nc;  normal_tensor[:,:,3] = 1.0

    # Alpha
    a = hit.float()
    alpha_tensor[:,:,0] = a; alpha_tensor[:,:,1] = a
    alpha_tensor[:,:,2] = a; alpha_tensor[:,:,3] = 1.0


@ctrl.on_frame
def draw_controls():
    ctrl.text(f"FPS: {view.fps:.1f}")
    ctrl.separator()
    state["sphere_r"] = ctrl.slider("Radius", 0.1, 1.5, default=0.6)
    ctrl.separator()
    state["light_az"] = ctrl.slider("Light Az", -3.14, 3.14, default=0.5)
    state["light_el"] = ctrl.slider("Light El", -1.5, 1.5, default=0.8)
    state["ambient"]  = ctrl.slider("Ambient", 0.0, 1.0, default=0.1)
    ctrl.separator()
    bg = ctrl.color_picker("Background", default=(0.12, 0.12, 0.14))
    state["bg_r"], state["bg_g"], state["bg_b"] = bg


@view.on_frame
def update():
    render_sphere()


view.run()
```

*（以上代码有删节 —— 完整的 turbo 色图 LUT 见 `examples/07_multichannel.py`。）*

## 刚才发生了什么？

### 四个面板，四个 tensor，一个窗口

```python
rgb_tensor    = vultorch.create_tensor(H, W, 4, device, name="rgb", ...)
depth_tensor  = vultorch.create_tensor(H, W, 4, device, name="depth", ...)
normal_tensor = vultorch.create_tensor(H, W, 4, device, name="normal", ...)
alpha_tensor  = vultorch.create_tensor(H, W, 4, device, name="alpha", ...)
```

每次调用分配一个独立的 Vulkan 共享 tensor。每个面板绑定其中一个。
四个 tensor 每帧更新，不经过 CPU —— 数据路径是 CUDA → Vulkan → 屏幕。

这就是神经渲染的工作流：你的模型做一次前向传播，填四个 tensor，
查看器同时把它们全部显示出来。

### Turbo 色图 —— 让深度看得见

原始深度值是个浮点数，值域不确定。直接显示的话你只能看到一张几乎全黑的图，
渐变完全不可见。Turbo 色图把 `[0, 1]` 标量映射到感知均匀的彩虹色，
让你真正**看到**深度结构：

```python
def apply_turbo(values):
    idx = (values.clamp(0, 1) * 255).long()   # 量化到 256 个 bin
    return TURBO_LUT[idx]                       # 查表，返回 (H, W, 3)
```

全部在 GPU 上跑 —— 不需要 numpy，不需要 matplotlib。

### 法线→RGB 惯例

可视化表面法线的标准方法：把每个分量从 `[-1, 1]` 映射到 `[0, 1]`，
分别赋给一个颜色通道：

```python
normal_color = normal * 0.5 + 0.5   # X→R, Y→G, Z→B
```

表面朝右是红色，朝上是绿色，朝摄像机是蓝色。
每篇神经渲染论文都用这个惯例，所以你一眼就能认出来。

### 光线-球体求交

整个渲染器只有 ~30 行 PyTorch 代码。核心是求交的二次公式：

$$t = \frac{-b - \sqrt{b^2 - 4ac}}{2a}$$

其中 $a = \|d\|^2$，$b = 2 \langle o, d \rangle$，$c = \|o\|^2 - r^2$。
对所有 $H \times W$ 条光线并行计算，一次 GPU kernel 调用完成。
把这部分替换成你的神经网络前向传播，你就得到了一个 NeRF 查看器。

## 要点

1. **多个零拷贝 tensor** —— 每个输出通道调一次 `create_tensor`，
   各自绑定到自己的面板。所有更新都在 GPU 上完成。

2. **Turbo 色图** —— 用 `(values * 255).long()` 索引 GPU 上的 LUT。
   对深度、视差、attention 权重、loss 热力图等标量场至关重要 ——
   灰度下会看不见。

3. **法线→RGB** —— `n * 0.5 + 0.5`。三个字符的代码，
   图形学/视觉社区通用的惯例。

4. **程序化→神经** —— 本例用光线-球体求交作为替身。
   把 `render_sphere()` 换成你模型的 forward pass，
   你就得到了一个实时多通道神经渲染查看器。

5. **零拷贝的可扩展性** —— 四个 256×256×4 纹理每帧更新。
   瓶颈是你的计算，不是显示管线。

!!! tip "提示"
    拖动面板边框可以重新排列四个视图 —— 把深度放在 RGB 旁边做对比，
    或者把法线叠在 alpha 上面。布局完全可以在运行时由用户配置。
