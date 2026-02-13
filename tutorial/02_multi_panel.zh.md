# 02 — 多面板

> **示例文件：** `examples/02_imgui_controls.py`

一个 View 可以拥有任意多个 Panel。每个 Panel 都是独立的可停靠窗口，
用户可以拖拽、缩放、重新排列。本章展示两种布局模式：

- **每个面板一个画布** — 三个面板垂直堆叠，各显示不同的 tensor。
- **一个面板多个画布** — 单个面板内放三个画布，自动均分空间。

## 布局图示

```
┌──────────────────────────────────────────────────┐
│  Red（面板）             │                       │
│    canvas: red_img       │  Combined（面板）     │
├──────────────────────────┤                       │
│  Green（面板）           │    canvas: c_red      │
│    canvas: green_img     │    canvas: c_green    │
├──────────────────────────┤    canvas: c_blue     │
│  Blue（面板）            │                       │
│    canvas: blue_img      │                       │
└──────────────────────────────────────────────────┘
```

## 完整代码

```python
import torch
import vultorch

# -- 1. 准备三份 tensor ----------------------------------------------------
H, W = 128, 128
device = "cuda"

# 红色水平渐变
x = torch.linspace(0, 1, W, device=device)
red = torch.zeros(H, W, 3, device=device)
red[:, :, 0] = x.unsqueeze(0).expand(H, W)

# 绿色垂直渐变
y = torch.linspace(0, 1, H, device=device)
green = torch.zeros(H, W, 3, device=device)
green[:, :, 1] = y.unsqueeze(1).expand(H, W)

# 蓝色棋盘格
blue = torch.zeros(H, W, 3, device=device)
cx = (torch.arange(W, device=device) // 32) % 2
cy = (torch.arange(H, device=device) // 32) % 2
blue[:, :, 2] = (cx.unsqueeze(0) ^ cy.unsqueeze(1)).float()

# -- 2. 创建 View ----------------------------------------------------------
view = vultorch.View("02 - Multi-Panel", 1200, 600)

# -- 3. 左侧：3 个面板，各自一个画布 --------------------------------------
panel_r = view.panel("Red")
panel_g = view.panel("Green")
panel_b = view.panel("Blue")

panel_r.canvas("red_img").bind(red)
panel_g.canvas("green_img").bind(green)
panel_b.canvas("blue_img").bind(blue)

# -- 4. 右侧：1 个面板，内含 3 个画布 ------------------------------------
combined = view.panel("Combined", side="right", width=0.5)
combined.canvas("c_red").bind(red)
combined.canvas("c_green").bind(green)
combined.canvas("c_blue").bind(blue)

# -- 5. 运行 ---------------------------------------------------------------
view.run()
```

## 重点解析

1. **自动布局** — 不带 `side=` 参数的面板会在主区域垂直堆叠。
   使用 `side="left"` 或 `side="right"` 可将面板停靠到指定侧，
   通过 `width` 指定宽度比例（如 `0.5` = 50%）。

2. **单画布 vs. 多画布** — 每次调用 `panel.canvas()` 都会在该面板内创建新的画布。
   当多个画布的 `fit=True`（默认值）时，它们会自动均分面板的垂直空间。

3. **无需回调** — 对于静态数据，只需 `bind()` 加 `run()`。
   `@view.on_frame` 装饰器仅在需要逐帧更新逻辑时使用（后续章节会讲到）。

4. **用户可以重排** — 所有面板都支持停靠。用户可以拖拽标题栏重新排列、
   拉出为浮动窗口，或拖拽边框调整大小。

!!! note "说明"
    同一个 tensor 可以同时绑定到多个画布。
    本示例中 `red`、`green`、`blue` 各自同时出现在
    左侧面板和右侧组合面板中。
