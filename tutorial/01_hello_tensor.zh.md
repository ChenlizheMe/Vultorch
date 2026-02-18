# 01 — Hello Tensor

> **示例文件：** `examples/01_hello_tensor.py`

你是否受够了每个仓库都有一套自己实现的可视化方案 —— 这个用 matplotlib 手动刷新，
那个用 `cv2.imshow` 然后 `waitKey(1)`，还有的直接存 PNG 让你开个图片查看器？

Vultorch 只需 **四行代码** 就把 CUDA tensor 搬到屏幕上。不存图、不过 CPU、
不需要 `plt.pause(0.001)` 这种黑魔法。

## 你需要记住的东西

如果你用过 matplotlib，那你已经熟悉这种套娃：figure → axes → plot。
Vultorch 是同样的思路，只不过是给 GPU tensor 用的：

```
View          ← 操作系统窗口（类比 plt.figure）
 └─ Panel     ← 窗口里的一块区域（类比 plt.subplot）
     └─ Canvas  ← 显示 tensor 的槽位（类比 ax.imshow）
         └─ bind(tensor)  ← 把数据接上去
```

一共就四个对象：

| 对象 | 可以理解为…… | 写法 |
|------|------------|------|
| **View** | 屏幕上的那个窗口 | `vultorch.View("title", w, h)` |
| **Panel** | 窗口里的一个矩形区域 | `view.panel("name")` |
| **Canvas** | 区域里的一个“相框” | `panel.canvas("name")` |
| **bind()** | 把你的 tensor 钉到相框里 | `canvas.bind(t)` |

链起来，调 `run()`，收工。

!!! info "面板到底是什么？"
    Panel（面板）就是窗口里一个可以拖动、调整大小的子窗口。
    想象成桌面上的便签纸 —— 你可以随便拖它、把它贴到窗口边缘、
    或者和其他面板叠在一起。你不需要管这些 —— Vultorch 会自动帮你排好。
    下一章会详细讲。

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
   画布默认铺满整个面板（`fit=True`），意思是它会自动拉伸
   填满所有可用空间 —— 就像一个 `imshow` 填满整个 figure 一样。

3. **运行** — `view.run()` 是一个**阻塞循环**（跟 `plt.show()` 类似）。
   它保持窗口打开，每秒重新画 tensor 约 60 次，并帮你处理
   操作系统事件（缩放窗口、点关闭等）。关窗口就退出 ——
   `run()` 返回后你的 Python 脚本继续往下跑。

!!! tip "提示"
    四行初始化可以压成一行，如果你喜欢炫技的话：
    `view.panel("Viewer").canvas("gradient").bind(t)`

!!! info "为什么不用 plt.imshow？"
    matplotlib 会把你的 tensor 拷到 CPU，转成 numpy 数组，用 CPU
    渲染，再绘到窗口上。Vultorch 把整个流程留在 GPU 上 ——
    tensor 通过 Vulkan 从 CUDA 显存直接刻到屏幕。所以即使是大图片，
    也能稳定 60 FPS 刷新。
