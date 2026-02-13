# 03 — 训练测试

> **示例文件：** `examples/03_training_test.py`

不用再猜“到底学没学到”，直接看。

本章用一个很轻量的 MLP 去实时拟合目标图：

- **左侧面板**：GT（`docs/images/pytorch_logo.png`）
- **右侧面板**：网络预测
- **下方面板**：实时文字信息（FPS、loss、iter、steps/frame）

## 布局

| 区域 | 内容 |
|------|------|
| 左侧 | **GT** 面板（真值图） |
| 右侧 | **Prediction** 面板（网络输出，每帧更新） |
| 下方 | **Info** 面板（FPS、loss、迭代、进度） |

## 完整代码

```python
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vultorch
ui = vultorch.ui

try:
    from PIL import Image
except ImportError as exc:
    raise RuntimeError("Please install pillow: pip install pillow") from exc


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
img = Image.open(img_path).convert("RGB").resize((256, 256), Image.BILINEAR)
gt = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).to(device)

H, W = gt.shape[0], gt.shape[1]
ys = torch.linspace(-1.0, 1.0, H, device=device)
xs = torch.linspace(-1.0, 1.0, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")
coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
target = gt.reshape(-1, 3)

model = TinyMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

view = vultorch.View("03 - Training Test", 1280, 760)
gt_panel = view.panel("GT")
pred_panel = view.panel("Prediction")

gt_panel.canvas("gt").bind(gt)

pred_rgba = vultorch.create_tensor(H, W, channels=4, device=device,
                                   name="pred", window=view.window)
pred_rgba[:, :, 3] = 1.0
pred_panel.canvas("pred").bind(pred_rgba)

state = {
    "iter": 0,
    "loss": 1.0,
    "ema": 1.0,
    "steps_per_frame": 6,
    "layout_done": False,
}


@view.on_frame
def train_and_render():
    if not state["layout_done"]:
        dockspace_id = ui.dock_space_over_viewport(flags=8)
        ui.dock_builder_remove_node(dockspace_id)
        ui.dock_builder_add_node(dockspace_id, 1 << 10)
        ui.dock_builder_set_node_size(dockspace_id, 1280.0, 760.0)

        info_node, top_node = ui.dock_builder_split_node(dockspace_id, 3, 0.28)
        left_node, right_node = ui.dock_builder_split_node(top_node, 0, 0.5)

        ui.dock_builder_dock_window("GT", left_node)
        ui.dock_builder_dock_window("Prediction", right_node)
        ui.dock_builder_dock_window("Info", info_node)
        ui.dock_builder_finish(dockspace_id)
        state["layout_done"] = True

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

    ui.begin("Info", True, 0)
    ui.text(f"FPS: {view.fps:.1f}")
    ui.text(f"Iteration: {state['iter']}")
    ui.text(f"Loss (MSE): {state['loss']:.6f}")
    ui.text(f"EMA Loss: {state['ema']:.6f}")

    state["steps_per_frame"] = ui.slider_int(
        "Steps / Frame", state["steps_per_frame"], 1, 32
    )
    progress = min(1.0, state["iter"] / 3000.0)
    ui.progress_bar(progress, overlay=f"Training progress {progress * 100:.1f}%")
    ui.text_wrapped(
        "左边是 GT，右边是预测。想更快收敛就提高 Steps / Frame。"
    )
    ui.end()


view.run()
```

## 说明

- 不再在控制台刷迭代日志；关键信息全部放在下方 `Info` 面板。
- 预测图通过 `create_tensor(..., channels=4)` 走 GPU 显示路径。
- 想更快看到拟合效果，调大 `Steps / Frame`。
