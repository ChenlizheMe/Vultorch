# 02 — 多面板

> **示例文件：** `examples/02_imgui_controls.py`

一个面板挺好的。但实际做研究的时候，你想同时看 loss map、梯度场、
模型输出 —— 而且最好不用自己写任何布局代码。

好消息：Vultorch 的面板是**可停靠的**。你只管创建，它们自己排好。
用户运行时还能随便拖、随便拉。

本章演示两种模式：

- **每个面板一个画布** — 三个面板垂直堆叠，各自一张图。
- **一个面板多个画布** — 一个面板里塞三个画布，自动均分。

## 布局

窗口长这样：

| 左侧（主区域） | 右侧（`side="right"`） |
|----------------|------------------------|
| **Red** 面板 — `red_img` | **Combined** 面板 |
| **Green** 面板 — `green_img` | `c_red` 画布 |
| **Blue** 面板 — `blue_img` | `c_green` 画布 |
| | `c_blue` 画布 |

左边：3 个独立面板，各一个画布。右边：1 个面板，3 个画布共享空间。

## 完整代码

```python
import torch
import vultorch

H, W = 128, 128
device = "cuda"

# 三张不同的 tensor
x = torch.linspace(0, 1, W, device=device)
red = torch.zeros(H, W, 3, device=device)
red[:, :, 0] = x.unsqueeze(0).expand(H, W)

y = torch.linspace(0, 1, H, device=device)
green = torch.zeros(H, W, 3, device=device)
green[:, :, 1] = y.unsqueeze(1).expand(H, W)

blue = torch.zeros(H, W, 3, device=device)
cx = (torch.arange(W, device=device) // 32) % 2
cy = (torch.arange(H, device=device) // 32) % 2
blue[:, :, 2] = (cx.unsqueeze(0) ^ cy.unsqueeze(1)).float()

view = vultorch.View("02 - Multi-Panel", 1200, 600)

# 左侧：3 个面板，各一个画布
panel_r = view.panel("Red")
panel_g = view.panel("Green")
panel_b = view.panel("Blue")
panel_r.canvas("red_img").bind(red)
panel_g.canvas("green_img").bind(green)
panel_b.canvas("blue_img").bind(blue)

# 右侧：1 个面板，3 个画布（自动垂直均分）
combined = view.panel("Combined", side="right", width=0.5)
combined.canvas("c_red").bind(red)
combined.canvas("c_green").bind(green)
combined.canvas("c_blue").bind(blue)

view.run()
```

## 要点

1. **自动布局** — 不写 `side=` 的面板自动垂直堆叠。
   加 `side="right"` + `width=0.5` 就停靠到右边占一半。

2. **多画布** — 对同一个面板多次调用 `panel.canvas()`。
   多个 `fit=True`（默认）的画布会自动均分垂直空间，不用手算高度。

3. **依然不需要回调** — 静态数据只需 `bind()` + `run()`。
   动态更新后面的章节会讲。

4. **随便拖** — 所有面板都支持停靠。用户可以拖标题栏重排、
   拉出来变浮动窗口、或者拖边框调大小。

!!! note "说明"
    同一个 tensor 可以同时绑定多个画布 ——
    `red` 同时出现在左边的 Red 面板和右边的 Combined 面板里。
