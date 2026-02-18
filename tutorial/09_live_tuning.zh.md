# 09 — 实时超参数调优

> **示例文件:** `examples/09_live_tuning.py`

你正在训练一个模型。Loss 卡住了。你想换个学习率试试。
于是你杀掉进程，改脚本，重启，等初始化，等前 500 轮
重新跑完之前的进度，*然后*才能看到新 LR 有没有用。

或者——你可以直接拖一个滑块。

这个例子可以在训练过程中实时修改：学习率、优化器、损失函数、
权重衰减。不用重启，不用存 checkpoint，不用写任何额外代码。

秘密武器是 `step()` / `end_step()` ——它们把主循环的控制权
交给你的训练代码，而 Vultorch 在每一步负责渲染。

## 新朋友

| 新东西 | 它做什么 | 为什么重要 |
|--------|---------|-----------|
| `view.step()` | 处理一帧，窗口关闭时返回 `False` | 你的训练 `while` 循环拥有外层迭代 |
| `view.end_step()` | 结束当前帧 | 与 `step()` 配对使用，替代 `run()` |
| `view.close()` | 显式销毁窗口 | 在 `finally` 中调用，确保干净关闭 |
| 对数 LR 滑块 | slider 范围 -5 到 -1，然后 `10 ** value` | 线性滑块对学习率完全没用——必须用对数尺度 |
| 优化器热切换 | 检测 combo 变化，重建 optimizer | 不重启就能在 Adam ↔ SGD ↔ AdamW 之间切换 |
| `compute_loss()` | MSE / L1 / Huber 由 combo 选择 | 不同损失函数对不同问题，运行时可切换 |

## 我们要做什么

和示例 08 一样的坐标 MLP，但现在有一个控制侧边栏可以让你：

- **拖动 LR** — 对数尺度滑块（1e-5 到 0.1）
- **切换优化器** — Adam、SGD（带动量）、AdamW
- **切换损失函数** — MSE、L1、Huber
- **调整权重衰减** — 实时生效
- **重置模型** — 一键恢复随机权重
- **即时看到效果** — 预测图像和 loss 曲线立刻反映变化

## 完整代码

```python
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import vultorch

device = "cuda"

# 加载目标图像
img_path = Path(__file__).resolve().parents[1] / "docs" / "images" / "pytorch_logo.png"
gt = vultorch.imread(img_path, channels=3, size=(256, 256), device=device)
H, W = gt.shape[0], gt.shape[1]

# 坐标网格
coords = ...  # (H*W, 2) in [-1, 1]
target = gt.reshape(-1, 3)


class CoordMLP(nn.Module):
    def __init__(self, hidden=64, layers=3):
        super().__init__()
        net = [nn.Linear(2, hidden), nn.ReLU(inplace=True)]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
        net += [nn.Linear(hidden, 3), nn.Sigmoid()]
        self.net = nn.Sequential(*net)
    def forward(self, x):
        return self.net(x)


OPTIMIZER_NAMES = ["Adam", "SGD", "AdamW"]
LOSS_NAMES = ["MSE", "L1", "Huber"]


def make_optimizer(model, name, lr, weight_decay):
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
    elif name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr,
                               momentum=0.9, weight_decay=weight_decay)
    elif name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)


def compute_loss(pred, target, loss_name):
    if loss_name == "MSE":  return F.mse_loss(pred, target)
    elif loss_name == "L1": return F.l1_loss(pred, target)
    elif loss_name == "Huber": return F.smooth_l1_loss(pred, target)


model = CoordMLP().to(device)
optimizer = make_optimizer(model, "Adam", 2e-3, 0.0)

# View + 面板
view = vultorch.View("09 - Live Hyperparameter Tuning", 1100, 900)
ctrl = view.panel("Controls", side="left", width=0.30)
pred_panel = view.panel("Prediction")
metrics_panel = view.panel("Metrics")

pred_rgba = vultorch.create_tensor(H, W, 4, device, name="pred",
                                    window=view.window)
pred_rgba[:, :, 3] = 1.0
pred_panel.canvas("pred").bind(pred_rgba)


@ctrl.on_frame
def draw_controls():
    log_lr = ctrl.slider("log10(LR)", -5.0, -1.0, default=-2.7)
    state["lr"] = 10.0 ** log_lr
    for pg in optimizer.param_groups:
        pg["lr"] = state["lr"]

    state["optimizer_idx"] = ctrl.combo("Optimizer", OPTIMIZER_NAMES)
    state["loss_idx"] = ctrl.combo("Loss", LOSS_NAMES)

    if ctrl.button("Reset Model"):
        state["needs_reset"] = True


# 训练循环 — 用 step()/end_step() 而不是 run()
try:
    while view.step():
        if state["needs_reset"]:
            model = CoordMLP().to(device)
            optimizer = make_optimizer(...)
            state["needs_reset"] = False

        if state["optimizer_idx"] != state["prev_optimizer_idx"]:
            optimizer = make_optimizer(...)
            state["prev_optimizer_idx"] = state["optimizer_idx"]

        for _ in range(state["steps_per_frame"]):
            optimizer.zero_grad(set_to_none=True)
            out = model(coords)
            loss = compute_loss(out, target, LOSS_NAMES[state["loss_idx"]])
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = model(coords).reshape(H, W, 3).clamp_(0, 1)
            pred_rgba[:, :, :3] = pred

        view.end_step()
finally:
    view.close()
```

*(精简版——完整代码见 `examples/09_live_tuning.py`。)*

## 刚才发生了什么？

### step() / end_step() — 你的训练循环做主

在示例 01–08 中我们都用 `view.run()`。很方便——Vultorch 拥有主循环然后调你的回调。
但在训练场景里，*你*才应该拥有主循环：

```python
try:
    while view.step():
        # ... 你的训练代码 ...
        view.end_step()
finally:
    view.close()
```

`step()` 处理一帧的输入和渲染，窗口关闭时返回 `False`。
`end_step()` 结束这一帧。你的训练代码放在它们中间。

可以把它想象成 matplotlib 的 `ion()` 模式——图在更新，但你的脚本
还在继续跑。只不过这里是 60fps、零拷贝、还有滑块可以拖。

### 对数尺度 LR 滑块

线性滑块对学习率来说是灾难。1e-4 和 2e-4 的差别很重要，但在
0 到 0.1 的线性滑块上，那只是一个看不见的小格子。对数尺度解决问题：

```python
log_lr = ctrl.slider("log10(LR)", -5.0, -1.0, default=-2.7)
state["lr"] = 10.0 ** log_lr
```

滑块位置 -5.0 = LR 1e-5，位置 -1.0 = LR 0.1。现在你可以精细控制
从微调到激进预热的所有学习率范围。

应用到当前优化器：

```python
for pg in optimizer.param_groups:
    pg["lr"] = state["lr"]
```

PyTorch 的优化器在每次 `step()` 时从 `param_groups` 读取 `lr`。
改了那里，下一次 `optimizer.step()` 就用新值。不需要重建。

### 优化器热切换

当你把 combo 从 Adam 切到 SGD 时，我们必须创建一个新的 optimizer
对象——没有办法把一种变形成另一种：

```python
if state["optimizer_idx"] != state["prev_optimizer_idx"]:
    optimizer = make_optimizer(
        model, OPTIMIZER_NAMES[state["optimizer_idx"]],
        state["lr"], state["weight_decay"])
    state["prev_optimizer_idx"] = state["optimizer_idx"]
```

模型的参数不变——只有优化器状态（动量缓冲区、Adam 的滑动平均）
被重置。这其实很有用：有时候切到 SGD 跑几轮可以帮模型跳出 Adam
陷入的局部最小值。

### 损失函数切换

不同的损失函数强调不同的误差模式：

- **MSE** — 二次惩罚大误差。PSNR 的标准。
- **L1** — 平等惩罚所有误差。对异常值更鲁棒，但零附近梯度恒定
  （可能导致震荡）。
- **Huber** — 零附近用 MSE，远处用 L1。深度估计领域人人都用的
  "两全其美"选择。

运行时切换可以让你直观看到不同损失函数对收敛的影响。
试试先用 MSE 跑 1000 轮，然后切到 Huber——你可能会看到 loss
进一步下降，因为 Huber 更好地处理了离群像素。

### 模型重置

```python
if ctrl.button("Reset Model"):
    state["needs_reset"] = True
```

延迟到训练循环中处理（不在回调内部），这样模型重建发生在正确的时机。
新模型得到全新的随机权重、全新的优化器，计数器归零。

## 核心要点

1. **`step()` / `end_step()`** — 当你需要自定义训练循环时用这对，
   替代 `run()`。你的 `while` 循环掌控迭代。

2. **对数 LR 滑块** — 学习率跨越多个数量级。线性滑块没用，
   用 `10 ** slider_value`。

3. **优化器热切换** — 运行时从 Adam 切到 SGD 再切到 AdamW。
   模型参数保留，只有优化器状态重置。

4. **损失函数切换** — MSE、L1、Huber 运行时可切换。不同损失
   揭示不同的训练动态。

5. **即时反馈** — 每次滑块变化在下一个训练步就生效。不用重启，
   不用存档，不用任何额外代码。

!!! tip "小贴士"
    先用 Adam + LR 2e-3 开始训练，等 loss 趋于平稳后，
    试试切到 SGD with momentum。不同的优化地形有时候
    能改善最终质量。

!!! note "注意"
    PSNR 始终用 MSE 计算，与当前选择的损失函数无关，
    这样 PSNR 数值在不同损失模式下保持可比性。
