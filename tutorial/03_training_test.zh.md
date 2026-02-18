# 03 — 训练测试

> **示例文件：** `examples/03_training_test.py`

你有没有经历过这种事：训练跑了半小时，终端里 loss 数字哗哗往下掉，
看起来挺正常 —— 结果一出图发现模型输出全是灰的？

盯着命令行里的数字猜模型状态，跟看股票 K 线猜明天涨跌一样不靠谱。
本章直接把 GT 和预测并排放在屏幕上。网络到底有没有在学，一眼就知道。

## 这次玩什么

一个很小的 MLP（2 → 64 → 64 → 3）去实时拟合一张 256×256 的 PyTorch logo。
窗口分三块：

| 区域 | 内容 |
|------|------|
| 左侧 | **GT** 面板 — 目标图（你要拟合的东西） |
| 右侧 | **Prediction** 面板 — 网络实时输出，每帧刷新 |
| 下方 | **Info** 面板 — FPS、loss、迭代次数、进度条，还能拖滑条 |

所有数值都在画面里，不用再在终端里翻来翻去找。

## 新朋友

前两章都是静态数据 —— `bind()` + `run()`，完事。
这次我们要引入三个新东西：

| 新东西 | 干什么用 | 写法 |
|--------|----------|------|
| **on_frame** | 每帧回调，训练 + 更新在这里搞 | `@view.on_frame` |
| **Panel.on_frame** | 面板回调，控件放这里 | `@info_panel.on_frame` |
| **create_tensor** | 创建 GPU 共享显存 tensor | `vultorch.create_tensor(H, W, ...)` |
| **vultorch.imread** | 加载图片，零依赖 | `vultorch.imread(path, channels=3)` |
| **side="bottom"** | 把面板停靠到窗口底部 | `view.panel("Info", side="bottom")` |

View 回调里写 PyTorch 代码，Panel 回调里画控件，
Vultorch 每帧帮你把 tensor 搬上屏幕。

## 完整代码

```python
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import vultorch


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"

img_path = Path(__file__).resolve().parents[1] / "docs" / "images" / "pytorch_logo.png"
gt = vultorch.imread(img_path, channels=3, size=(256, 256), device=device)

H, W = gt.shape[0], gt.shape[1]
ys = torch.linspace(-1.0, 1.0, H, device=device)
xs = torch.linspace(-1.0, 1.0, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")
coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
target = gt.reshape(-1, 3)

model = TinyMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

# -- 视图 + 面板（高层声明式 API）------------------------------------
view = vultorch.View("03 - Training Test", 1280, 760)
info_panel = view.panel("Info", side="bottom", width=0.28)
gt_panel = view.panel("GT", side="left", width=0.5)
pred_panel = view.panel("Prediction")

gt_panel.canvas("gt").bind(gt)

# 预测用 4 通道 — GPU 零拷贝显示
pred_rgba = vultorch.create_tensor(H, W, channels=4, device=device,
                                   name="pred", window=view.window)
pred_rgba[:, :, 3] = 1.0
pred_panel.canvas("pred").bind(pred_rgba)

state = {
    "iter": 0,
    "loss": 1.0,
    "ema": 1.0,
    "steps_per_frame": 6,
}


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
        state["ema"] = state["ema"] * 0.98 + state["loss"] * 0.02

    with torch.no_grad():
        pred = model(coords).reshape(H, W, 3).clamp_(0, 1)
        pred_rgba[:, :, :3] = pred


@info_panel.on_frame
def draw_info():
    info_panel.text(f"FPS: {view.fps:.1f}")
    info_panel.text(f"Iteration: {state['iter']}")
    info_panel.text(f"Loss (MSE): {state['loss']:.6f}")
    info_panel.text(f"EMA Loss: {state['ema']:.6f}")

    state["steps_per_frame"] = info_panel.slider_int(
        "Steps / Frame", 1, 32, default=6
    )
    progress = min(1.0, state["iter"] / 3000.0)
    info_panel.progress(progress,
                        overlay=f"Training progress {progress * 100:.1f}%")
    info_panel.text_wrapped(
        "左边是 GT，右边是预测。想更快收敛就提高 Steps / Frame。"
    )


view.run()
```

搞定。跑起来之后你会看到右边那坨灰色在几秒内变成 PyTorch logo。

## 刚才发生了什么？

1. **数据** — `vultorch.imread` 直接把图片读成 float32 CUDA tensor（不需要 PIL，不需要 numpy）。
   坐标用 `meshgrid` 展成 `(H*W, 2)`，每个像素的 `(x, y)` 归一化到 `[-1, 1]`。

2. **模型** — 两层 64 宽的 MLP，输入 `(x, y)`，输出 `(r, g, b)`。
   这个网络小到可以跑在回调里不掉帧。

3. **布局** — `side="bottom"` 把 Info 停靠到底部 28%。
   `side="left"` 把 GT 放到剩余空间的左半边。
   Prediction 自动填满剩下的区域。不需要手写任何 docking 代码。

4. **两个回调** — `@view.on_frame` 跑训练。
   `@info_panel.on_frame` 在 Info 面板里画控件（文字、滑条、进度条）。
   Vultorch 帮你管 ImGui 的 begin / end，不用自己操心。

## 要点

1. **`@view.on_frame`** — 回调里可以跑任意 PyTorch 代码。
   每帧结束时 Vultorch 自动把绑定的 tensor 搬到屏幕，不用你管。

2. **`create_tensor`** — 跟普通 `torch.zeros` 一样用，
   但底层是 Vulkan/CUDA 共享显存，显示的时候零拷贝。

3. **声明式布局** — `side="left"` / `"right"` / `"bottom"` / `"top"`
   切窗口，不需要手动写 dock_builder 代码。

4. **面板控件** — `@panel.on_frame` 在面板窗口内部运行。
   用 `panel.text()`、`panel.slider_int()`、`panel.progress()` 代替原始 `ui.*` 调用。

5. **不刷终端** — 所有状态信息都在 Info 面板里，
   你的终端可以留着看 warning 和 traceback，干净多了。

!!! tip "提示"
    `Steps / Frame` 滑条拉到 32 收敛飞快。
    但也别太贪 —— 拉太高帧率会掉下来，因为每帧的训练时间变长了。

!!! note "说明"
    `create_tensor` 只在初始化时调用一次，不在帧循环里。
    之后每帧只需要往这个 tensor 里写数据，开销几乎为零。
