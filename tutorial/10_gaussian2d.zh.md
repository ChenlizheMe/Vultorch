# 10 — 二维高斯泼溅

> **示例文件:** `examples/10_gaussian2d.py`

3D Gaussian Splatting（3DGS）席卷了神经渲染领域。但它的核心思想
出奇地简单：撒一堆彩色的椭圆斑点，alpha 混合到一起，和目标图比较，
然后反向传播。

这个例子把 3DGS 蒸馏到二维本质。没有相机投影，没有球谐函数，
没有分块光栅化。就是 N 个可学习的二维高斯——每个有位置、尺度、
旋转、颜色、不透明度——通过可微分 alpha 合成渲染，用普通的
PyTorch autograd 优化。

看着高斯们在画布上游走、缩放、旋转、变色，最终重新组装出目标图像。
这就是那个能以 100+ fps 渲染照片级真实场景的算法。

## 新朋友

| 新东西 | 它做什么 | 为什么重要 |
|--------|---------|-----------|
| 高斯的 `nn.Parameter` | 位置、对数尺度、旋转、原始颜色、原始不透明度 | 每个高斯属性都可学习——autograd 搞定剩下的 |
| 逆协方差 | 从尺度 + 旋转得到 $\Sigma^{-1}$ | 让每个高斯有自己的椭圆形状的数学 |
| Alpha 合成 | $C = \sum_i \alpha_i T_i c_i$，其中 $T_i = \prod_{j<i}(1-\alpha_j)$ | 体渲染的标准前向到后向混合公式 |
| `torch.cumprod` | 累积乘积计算透射率 | 高效并行计算所有高斯的 $T_i$ |
| 高斯中心叠加 | 每个均值位置画红点 | 看到高斯在哪里，不只是它们渲染出什么 |

## 我们要做什么

N 个随机二维高斯，每个由 5 个属性定义：

1. **位置** (μ) — 画布上的位置，在 [0, 1]² 内
2. **尺度** (σ) — x 和 y 方向的宽度（存为对数尺度以保证正数）
3. **旋转** (θ) — 方向角，弧度
4. **颜色** (c) — RGB，通过 sigmoid
5. **不透明度** (α) — 多不透明，通过 sigmoid

每帧：渲染所有高斯 → MSE loss 对比目标 → backward → step。
高斯逐渐收敛以重现目标图像。

## 完整代码

```python
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import vultorch

device = "cuda"

# 加载目标
img_path = Path(__file__).resolve().parents[1] / "docs" / "images" / "pytorch_logo.png"
gt = vultorch.imread(img_path, channels=3, size=(128, 128), device=device)
H, W = gt.shape[0], gt.shape[1]

# 像素坐标 [0, 1]
ys = torch.linspace(0, 1, H, device=device)
xs = torch.linspace(0, 1, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")
pixel_coords = torch.stack([xx, yy], dim=-1)


class GaussianModel2D(nn.Module):
    def __init__(self, n_gaussians=200):
        super().__init__()
        self.n = n_gaussians
        self.means = nn.Parameter(torch.rand(n_gaussians, 2, device=device))
        self.log_scales = nn.Parameter(
            torch.full((n_gaussians, 2), -3.0, device=device))
        self.rotations = nn.Parameter(
            torch.zeros(n_gaussians, device=device))
        self.raw_colors = nn.Parameter(
            torch.randn(n_gaussians, 3, device=device))
        self.raw_opacities = nn.Parameter(
            torch.zeros(n_gaussians, device=device))

    def forward(self, coords):
        means = self.means
        scales = self.log_scales.exp()
        colors = torch.sigmoid(self.raw_colors)
        opacities = torch.sigmoid(self.raw_opacities)

        cos_r = torch.cos(self.rotations)
        sin_r = torch.sin(self.rotations)
        sx2 = scales[:, 0] ** 2
        sy2 = scales[:, 1] ** 2

        # 逆协方差矩阵元素
        a = cos_r**2 / (2*sx2) + sin_r**2 / (2*sy2)
        b = -sin_r*cos_r / (2*sx2) + sin_r*cos_r / (2*sy2)
        c = sin_r**2 / (2*sx2) + cos_r**2 / (2*sy2)

        dx = coords[:,:,0].unsqueeze(-1) - means[:,0]
        dy = coords[:,:,1].unsqueeze(-1) - means[:,1]

        exponent = -(a*dx*dx + 2*b*dx*dy + c*dy*dy)
        alpha = opacities * torch.exp(exponent)
        alpha = alpha.clamp(0, 0.99)

        # 前向-后向 alpha 合成
        T = torch.ones(H, W, 1, device=device)
        transmittance = torch.cumprod(
            torch.cat([T, (1-alpha)[:,:,:-1]], dim=-1), dim=-1)
        weights = alpha * transmittance

        rendered = (weights.unsqueeze(-1) *
                    colors.unsqueeze(0).unsqueeze(0)).sum(dim=2)
        bg = (transmittance[:,:,-1] * (1 - alpha[:,:,-1])).unsqueeze(-1)
        rendered = rendered + bg  # (H,W,1) broadcasts to (H,W,3)

        return rendered.clamp(0, 1)


model = GaussianModel2D(200).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

# View + 面板
view = vultorch.View("10 - 2D Gaussian Splatting", 1100, 900)
ctrl = view.panel("Controls", side="left", width=0.28)
gt_panel = view.panel("Ground Truth")
render_panel = view.panel("Rendered")
metrics_panel = view.panel("Metrics")

# ... 设置画布、状态、回调 ...

try:
    while view.step():
        for _ in range(steps_per_frame):
            rendered = model(pixel_coords)
            loss = F.mse_loss(rendered, gt)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            render_rgba[:, :, :3] = model(pixel_coords).clamp_(0, 1)

        view.end_step()
finally:
    view.close()
```

*(精简版——完整代码见 `examples/10_gaussian2d.py`。)*

## 刚才发生了什么？

### 二维高斯公式

每个高斯是一个椭圆形的色斑。其形状由以下公式定义：

$$G(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

其中 $\Sigma = R \begin{pmatrix} \sigma_x^2 & 0 \\ 0 & \sigma_y^2 \end{pmatrix} R^T$，$R$ 是旋转矩阵。

代码中我们直接计算逆协方差的元素：

```python
a = cos_r**2 / (2*sx2) + sin_r**2 / (2*sy2)
b = -sin_r*cos_r / (2*sx2) + sin_r*cos_r / (2*sy2)
c = sin_r**2 / (2*sx2) + cos_r**2 / (2*sy2)

exponent = -(a*dx*dx + 2*b*dx*dy + c*dy*dy)
```

`log_scales` 技巧确保尺度经 `exp()` 后始终为正。
旋转角无约束——任何实数都行。

### Alpha 合成——体渲染方程

每个像素的颜色是所有高斯的加权和：

$$C = \sum_{i=1}^{N} \alpha_i T_i c_i + T_N \cdot c_{bg}$$

其中透射率 $T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$ 追踪有多少光
穿过了前面所有高斯到达第 $i$ 个。

```python
transmittance = torch.cumprod(
    torch.cat([T, (1-alpha)[:,:,:-1]], dim=-1), dim=-1)
weights = alpha * transmittance
rendered = (weights.unsqueeze(-1) * colors).sum(dim=2)
```

这和 NeRF 的体渲染、3DGS 的光栅化用的是完全相同的方程。
唯一的区别：NeRF 沿光线采样，3DGS 泼溅投影后的高斯，
而我们直接泼溅二维高斯。

### 为什么它能工作

每个高斯有 9 个可学习参数（2 位置 + 2 尺度 + 1 旋转 + 3 颜色
+ 1 不透明度）。200 个高斯就是 1800 个参数——比任何神经网络都小，
但足以近似一张 128×128 的图像。

关键洞察：高斯是图像重建的自然基函数。它们可以重叠，有柔和的边缘，
梯度平滑。Autograd 通过合成方程处理链式法则。

### 高斯中心叠加

```python
means = model.means.detach().clamp(0, 1)
px = (means[:, 0] * (W - 1)).long().clamp(0, W - 1)
py = (means[:, 1] * (H - 1)).long().clamp(0, H - 1)
render_rgba[py, px, 0] = 1.0  # 红点
```

这显示了每个高斯中心当前的位置。训练初期你看到随机的红点。
随着训练推进，它们会聚集在图像细节丰富的地方——
哪里需要重建的信息多，哪里就有更多高斯。

## 核心要点

1. **二维高斯泼溅就是没有相机的 3DGS** — 相同的 alpha 合成，
   相同的可学习椭圆，相同的优化过程。只是更简单。

2. **一切皆 `nn.Parameter`** — 位置、尺度、旋转、颜色、不透明度。
   PyTorch autograd 对所有属性求导。

3. **对数尺度技巧** — 存储 `log_scales`，用 `exp()` 保证正数，
   不需要 clamp。

4. **`torch.cumprod` 算透射率** — 高效并行计算前向-后向合成权重。

5. **从 2D 到 3D** — 要得到 3DGS，加上：3D 位置、相机投影、
   球谐函数实现视角依赖颜色、分块排序。核心合成数学不变。

!!! tip "小贴士"
    试试增加高斯数量（500 或 1000 个），观察优化动态的变化。
    更多高斯 = 更精细的细节，但每个高斯的收敛更慢。

!!! note "注意"
    这是一个教学实现——真正的 3DGS 使用 CUDA 核函数做分块光栅化，
    快 100 倍。但数学是完全一样的。
