# 01 — Hello Tensor

> **示例文件：** `examples/01_hello_tensor.py`

Vultorch 的最简用法：创建窗口、添加面板、放一个画布、绑定 CUDA 张量、运行。

## 核心层级

```
View          （顶层窗口）
 └── Panel    （可停靠的 ImGui 子窗口）
      └── Canvas   （用于显示 tensor 的 GPU 图像槽位）
           └── bind(tensor)  →  每帧自动渲染
```

这就是 Vultorch 的全部心智模型。

## 完整代码

```python
import torch
import vultorch

# -- 1. 准备数据 -----------------------------------------------------------
# 一张 256×256 的 RGB 渐变色。任何 (H, W, C) float32 CUDA tensor 均可。
H, W = 256, 256
x = torch.linspace(0, 1, W, device="cuda")            # 水平渐变
y = torch.linspace(0, 1, H, device="cuda")            # 垂直渐变
t = torch.stack([
    x.unsqueeze(0).expand(H, W),                       # R 通道：左→右
    y.unsqueeze(1).expand(H, W),                       # G 通道：上→下
    torch.full((H, W), 0.3, device="cuda"),            # B 通道：常量
], dim=-1)                                             # shape: (256, 256, 3)

# -- 2. 创建 View → Panel → Canvas → 绑定 tensor --------------------------
view   = vultorch.View("01 - Hello Tensor", 512, 512)  # 创建 512×512 窗口
panel  = view.panel("Viewer")                          # 添加一个面板
canvas = panel.canvas("gradient")                      # 面板上添加画布
canvas.bind(t)                                         # 把 tensor 绑定到画布

# 上面四行也可以链式写成一行：
# view.panel("Viewer").canvas("gradient").bind(t)

# -- 3. 运行 ---------------------------------------------------------------
# run() 会阻塞，直到用户关闭窗口。
# 画布会自动铺满面板空间，每帧从绑定的 tensor 读取数据并渲染。
view.run()
```

## 逐步解析

1. **准备数据** — 任何 `(H, W)`、`(H, W, 1)`、`(H, W, 3)` 或 `(H, W, 4)`，
   float32 / float16 / uint8 的 CUDA tensor 都可以，Vultorch 会自动处理
   RGBA 通道扩展。

2. **构建对象层级** — `View` 创建操作系统窗口，`panel()` 添加可停靠的
   ImGui 子窗口，`canvas()` 在面板内添加 GPU 图像槽位，`bind()` 将
   tensor 连接上去。

3. **运行** — `view.run()` 进入阻塞事件循环。每帧画布自动重新上传绑定的
   tensor 并渲染，默认铺满面板区域（`fit=True`）。

!!! tip "提示"
    上面四行初始化可以链式一行写完：
    `view.panel("Viewer").canvas("gradient").bind(t)`
