# 01 — Hello Tensor

> **示例文件：** `examples/01_hello_tensor.py`

你是否受够了每个仓库都有一套自己实现的可视化方案 —— 这个用 matplotlib 手动刷新，
那个用 `cv2.imshow` 然后 `waitKey(1)`，还有的直接存 PNG 让你开个图片查看器？

Vultorch 只需 **四行代码** 就把 CUDA tensor 搬到屏幕上。不存图、不过 CPU、
不需要 `plt.pause(0.001)` 这种黑魔法。

## 你需要记住的东西

一共就四个对象：

| 对象 | 是什么 | 写法 |
|------|--------|------|
| **View** | 操作系统窗口 | `vultorch.View("title", w, h)` |
| **Panel** | View 里可停靠的子窗口 | `view.panel("name")` |
| **Canvas** | Panel 里的 GPU 图像槽位 | `panel.canvas("name")` |
| **bind()** | 把 tensor 连到 Canvas 上 | `canvas.bind(t)` |

链起来，调 `run()`，收工。

## 完整代码

```python
import torch
import vultorch

# 一张 256×256 的 RGB 渐变 — 任何 (H,W,C) float32 CUDA tensor 都行
H, W = 256, 256
x = torch.linspace(0, 1, W, device="cuda")
y = torch.linspace(0, 1, H, device="cuda")
t = torch.stack([
    x.unsqueeze(0).expand(H, W),
    y.unsqueeze(1).expand(H, W),
    torch.full((H, W), 0.3, device="cuda"),
], dim=-1)  # (256, 256, 3)

view   = vultorch.View("01 - Hello Tensor", 512, 512)
panel  = view.panel("Viewer")
canvas = panel.canvas("gradient")
canvas.bind(t)
view.run()  # 阻塞，直到你关闭窗口
```

没了。不需要手写事件循环，不需要 `begin_frame()` / `end_frame()`。

## 刚才发生了什么？

1. **数据** — 我们在 CUDA 上造了一张 RGB 渐变。Vultorch 支持
   `(H,W)` / `(H,W,1)` / `(H,W,3)` / `(H,W,4)`，
   float32 / float16 / uint8 都行，RGBA 扩展它自己搞定。

2. **对象树** — `View` → `Panel` → `Canvas` → `bind(tensor)`。
   画布默认铺满整个面板（`fit=True`）。

3. **运行** — `view.run()` 进入阻塞事件循环，每帧重新上传 tensor 并渲染。
   关窗口就退出。

!!! tip "提示"
    四行初始化可以压成一行，如果你喜欢炫技的话：
    `view.panel("Viewer").canvas("gradient").bind(t)`
