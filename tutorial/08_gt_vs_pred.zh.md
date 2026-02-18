# 08 — GT vs 预测

> **示例文件：** `examples/08_gt_vs_pred.py`

每个搞神经渲染的人每天都在做同一件事：训练一个模型，看结果，和 GT 对比，
盯着差异图，琢磨误差藏在哪里。

通常的流程：存一张 PNG，在 matplotlib 里和 GT 并排打开，在另一个 cell 算 PSNR，
在 TensorBoard 里画 loss，在三个窗口间 alt-tab 切来切去，GPU 干等着。
等你拼好对比图的时候，已经忘了刚才改了什么超参数。

本章把这些全部放在一个窗口里：GT、预测、误差热力图、loss 曲线、PSNR ——
全部实时，全部 60 fps，全部零拷贝。

## 新朋友

| 新东西 | 干什么用 | 为什么重要 |
|--------|----------|-----------|
| 误差热力图 | `|GT - pred|` 放大后用 turbo 色图着色 | 看到裸眼对比根本看不见的误差 |
| `panel.plot()` | 用 Python 列表画折线图 | 实时 loss 和 PSNR 曲线，不需要 TensorBoard |
| `panel.progress()` | 一个进度条 | 快速看训练进度 |
| PSNR | $-10 \log_{10}(\text{MSE})$，从实时 loss 计算 | 图像重建质量的标准指标 |
| 误差模式 combo | 在 L1、L2、逐通道最大值之间切换 | 不同的误差范数暴露不同的问题 |
| 误差增益 slider | 放大微小误差使其可见 | 低误差区域乘以 5–20× 后才看得见 |

## 这次玩什么

一个坐标 MLP 拟合目标图像（和例子 03 一样，但升级版）。
三个图像面板 —— GT、预测、误差热力图 —— 加上一个指标侧边栏，
内含实时 loss/PSNR 曲线和控制器。

## 完整代码

```python
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import vultorch

if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"

# 加载目标图像
img_path = Path(__file__).resolve().parents[1] / "docs" / "images" / "pytorch_logo.png"
gt = vultorch.imread(img_path, channels=3, size=(256, 256), device=device)
H, W = gt.shape[0], gt.shape[1]

# MLP 的坐标网格
ys = torch.linspace(-1.0, 1.0, H, device=device)
xs = torch.linspace(-1.0, 1.0, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")
coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
target = gt.reshape(-1, 3)

# 简单的坐标 MLP
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

model = CoordMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

# Turbo 色图 LUT（详见例子 07）
TURBO_LUT = ...  # 256×3, GPU 上

def apply_turbo(values):
    idx = (values.clamp(0, 1) * 255).long()
    return TURBO_LUT[idx]

# 视图 + 面板
view = vultorch.View("08 - GT vs Prediction", 1280, 1000)
metrics_panel = view.panel("Metrics", side="right", width=0.28)
gt_panel = view.panel("Ground Truth")
pred_panel = view.panel("Prediction")
error_panel = view.panel("Error Map")

gt_panel.canvas("gt").bind(gt)

pred_rgba = vultorch.create_tensor(H, W, 4, device, name="pred",
                                    window=view.window)
pred_rgba[:, :, 3] = 1.0
pred_panel.canvas("pred").bind(pred_rgba)

error_rgba = vultorch.create_tensor(H, W, 4, device, name="error",
                                     window=view.window)
error_rgba[:, :, 3] = 1.0
error_panel.canvas("error").bind(error_rgba)

ERROR_MODES = ["L1", "L2", "Per-Channel Max"]
state = {"iter": 0, "loss": 1.0, "psnr": 0.0, "steps_per_frame": 6,
         "error_mode": 0, "error_gain": 5.0,
         "loss_history": [], "psnr_history": []}


def compute_error_map(gt_img, pred_img, mode, gain):
    if mode == 0:    err = (gt_img - pred_img).abs().mean(dim=-1)
    elif mode == 1:  err = ((gt_img - pred_img)**2).mean(dim=-1).sqrt()
    else:            err = (gt_img - pred_img).abs().max(dim=-1).values
    return apply_turbo((err * gain).clamp(0, 1))


def compute_psnr(mse):
    return -10.0 * math.log10(mse) if mse > 0 else 50.0


@view.on_frame
def train():
    for _ in range(state["steps_per_frame"]):
        optimizer.zero_grad(set_to_none=True)
        out = model(coords)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        state["iter"] += 1
        state["loss"] = loss.item()

    with torch.no_grad():
        pred = model(coords).reshape(H, W, 3).clamp_(0, 1)
        pred_rgba[:, :, :3] = pred

        error_color = compute_error_map(gt, pred, state["error_mode"],
                                         state["error_gain"])
        error_rgba[:, :, :3] = error_color
        state["psnr"] = compute_psnr(state["loss"])

    state["loss_history"].append(state["loss"])
    state["psnr_history"].append(state["psnr"])
    if len(state["loss_history"]) > 300:
        state["loss_history"] = state["loss_history"][-300:]
        state["psnr_history"] = state["psnr_history"][-300:]


@metrics_panel.on_frame
def draw_metrics():
    metrics_panel.text(f"FPS: {view.fps:.1f}")
    metrics_panel.text(f"Iteration: {state['iter']}")
    metrics_panel.separator()
    metrics_panel.text(f"MSE Loss: {state['loss']:.6f}")
    metrics_panel.text(f"PSNR: {state['psnr']:.2f} dB")
    metrics_panel.separator()

    metrics_panel.text("Loss 曲线")
    if state["loss_history"]:
        metrics_panel.plot(state["loss_history"], label="##loss",
                           overlay=f"{state['loss']:.5f}", height=80)

    metrics_panel.text("PSNR 曲线")
    if state["psnr_history"]:
        metrics_panel.plot(state["psnr_history"], label="##psnr",
                           overlay=f"{state['psnr']:.1f} dB", height=80)

    metrics_panel.separator()
    state["steps_per_frame"] = metrics_panel.slider_int(
        "Steps/Frame", 1, 32, default=6)
    state["error_mode"] = metrics_panel.combo("Error Mode", ERROR_MODES)
    state["error_gain"] = metrics_panel.slider("Error Gain", 1.0, 20.0,
                                                default=5.0)

    progress = min(1.0, state["iter"] / 5000.0)
    metrics_panel.progress(progress, overlay=f"{progress*100:.0f}%")


view.run()
```

*（有删节 —— 完整代码含 turbo LUT 见 `examples/08_gt_vs_pred.py`。）*

## 刚才发生了什么？

### 误差热力图 —— 看见看不见的东西

原始的 `|GT - pred|` 差值通常是接近零的小浮点数。直接显示的话你只能看到
一张几乎全黑的图，然后得出"一切正常"的结论。这是错误的。

```python
err = (gt_img - pred_img).abs().mean(dim=-1)   # 每像素 L1 误差
err = (err * gain).clamp(0, 1)                  # 放大 5–20×
heatmap = apply_turbo(err)                       # turbo 色图
```

增益滑条让你把微小误差放大到可见。5× 增益下你就能看到网络在哪里吃力 ——
边缘、精细纹理、高频区域。这种洞察力是单个 PSNR 数字永远给不了的。

### PSNR —— 神经渲染的标准指标

$$\text{PSNR} = -10 \log_{10}(\text{MSE})$$

每帧从实时 loss 更新。一个数字就知道你的模型处于什么水平：
20 dB（模糊一片）、30 dB（还行）、还是 40 dB（很锐利）。
实时曲线告诉你训练什么时候开始停滞。

```python
def compute_psnr(mse):
    return -10.0 * math.log10(mse) if mse > 0 else 50.0
```

### panel.plot() —— 不用 TensorBoard 的 loss 曲线

```python
metrics_panel.plot(state["loss_history"],
                   label="##loss",
                   overlay=f"{state['loss']:.5f}",
                   height=80)
```

传一个 Python float 列表进去，画一条折线。`overlay` 文字显示在图表上方。
保留最近 300 个值作为滚动窗口。不需要任何外部日志库。

### 误差模式 —— 不同范数暴露不同问题

- **L1**（`abs().mean(dim=-1)`）—— 平均绝对误差。显示预测整体哪里偏了。
- **L2**（`square().mean(dim=-1).sqrt()`）—— 均方根误差。比 L1 更放大离群值。
- **逐通道最大值**（`abs().max(dim=-1)`）—— 最差的通道。
  暴露颜色通道不匹配（比如蓝通道错了但红绿没问题）。

运行时用 combo 切换。不同的误差范数让不同的问题变得可见。

### 相比例子 03 的升级

例子 03 只有 GT 和预测并排。本例增加了：

- **误差热力图** —— 看到模型**哪里**错了，不只是**错了多少**。
- **PSNR + loss 曲线** —— 实时指标，不只是文本计数器。
- **误差增益** —— 放大微小误差使其可见。
- **误差模式切换** —— 运行时 L1/L2/逐通道。

这就是"我的模型在训练"和"我能在训练过程中调试模型"的区别。

## 要点

1. **三面板对比** —— GT、预测、误差热力图。这是所有重建任务
   最基本的视觉调试布局。

2. **误差热力图 = 放大差值 + 色图** —— 增益滑条是关键。
   不放大的话低误差根本看不见。

3. **实时 PSNR** —— 总结重建质量的一个数字。
   $-10 \log_{10}(\text{MSE})$，直接从已有的 loss 算。

4. **panel.plot()** —— 同一个窗口里的即时 loss/指标曲线。
   不需要 TensorBoard，不需要 wandb，不需要 alt-tab。

5. **运行时切换误差模式** —— 不同范数暴露不同问题。
   L1 看整体误差，L2 看离群值，逐通道看颜色 bug。

!!! tip "提示"
    训练收敛后把"Error Gain"拉到 15–20×。你会看到残余误差的分布 ——
    通常集中在边缘和高频纹理上，这会告诉你是否需要位置编码或更深的网络。

!!! note "说明"
    本例用的是一个简单的 3 层 MLP，为了跑得快。
    把它换成你自己的模型和训练循环 —— 可视化代码完全不用改。
