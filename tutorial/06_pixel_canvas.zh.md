# 06 — 像素画板

> **示例文件：** `examples/06_pixel_canvas.py`

到目前为止，我们显示的每个 tensor 从观众的角度来看都是"只读"的 ——
GPU 算完, Vultorch 显示，完事。
但如果用户想用鼠标**画**进 tensor 呢？实时地？

这就是本章的内容：把一个零拷贝 GPU tensor 变成交互式画板。
左键画，右键擦，选颜色，调笔刷大小 —— 搞定。

魔法在于根本没有魔法。
`create_tensor` 给你的就是一个普通的 `torch.Tensor`，在 CUDA 上。
你用标准索引往里写像素（`tensor[y, x, :3] = color`）。
因为是零拷贝，显示自动更新，不需要调任何上传函数。

## 新朋友

| 新东西 | 干什么用 | 为什么重要 |
|--------|----------|-----------|
| `ui.get_mouse_pos()` | 返回鼠标光标的屏幕像素坐标 `(x, y)` | 你需要知道用户指向画布的**哪个位置** |
| `ui.is_item_hovered()` | 鼠标在最后一个绘制的控件（画布图片）上方时返回 `True` | 只在光标在画布上时才画，不要画到控制面板上 |
| `ui.is_mouse_clicked(0)` | 左键按下的那一帧返回 `True` | 检测单次点击 |
| `ui.is_mouse_dragging(0)` | 左键按住且鼠标移动期间持续返回 `True` | 持续绘画 —— 拖拽时每帧都触发 |
| 屏幕→像素映射 | 把屏幕坐标转换为 tensor 的 `[y, x]` 索引 | 画布图片被拉伸到面板大小；你需要反向计算才能知道鼠标在哪个像素上 |

## 这次玩什么

一个 128×128 的像素画板：左键画，右键擦，侧边栏控制笔刷大小、颜色、
清除按钮和网格显示开关。

## 完整代码

```python
import torch
import vultorch
from vultorch import ui

if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"

H, W = 128, 128

view = vultorch.View("06 - Pixel Canvas", 900, 700)
ctrl = view.panel("Controls", side="left", width=0.24)
draw_panel = view.panel("Canvas")

# 零拷贝 RGBA tensor
canvas_tensor = vultorch.create_tensor(H, W, channels=4, device=device,
                                        name="canvas", window=view.window)
canvas_tensor[:, :, :3] = 0.1
canvas_tensor[:, :, 3] = 1.0
canvas = draw_panel.canvas("canvas", filter="nearest", fit=True)
canvas.bind(canvas_tensor)

# 持久化的底图（防止网格叠加效果累积）
backing = torch.zeros(H, W, 3, device=device)
backing[:] = 0.1

state = {
    "brush_size": 1,
    "brush_color": (1.0, 0.3, 0.1),
    "show_grid": False,
    "bg_color": (0.1, 0.1, 0.1),
}


def draw_brush(cy, cx, size, r, g, b):
    half = size // 2
    y0, y1 = max(0, cy - half), min(H, cy + half + 1)
    x0, x1 = max(0, cx - half), min(W, cx + half + 1)
    backing[y0:y1, x0:x1, 0] = r
    backing[y0:y1, x0:x1, 1] = g
    backing[y0:y1, x0:x1, 2] = b


def clear_canvas():
    r, g, b = state["bg_color"]
    backing[:, :, 0] = r
    backing[:, :, 1] = g
    backing[:, :, 2] = b


def refresh_display():
    canvas_tensor[:, :, :3] = backing
    if state["show_grid"]:
        for i in range(0, H, 8):
            canvas_tensor[i, :, :3] = canvas_tensor[i, :, :3].clamp(0, 0.85) + 0.15
        for j in range(0, W, 8):
            canvas_tensor[:, j, :3] = canvas_tensor[:, j, :3].clamp(0, 0.85) + 0.15


@ctrl.on_frame
def draw_controls():
    ctrl.text(f"Canvas: {W}×{H}")
    ctrl.text(f"FPS: {view.fps:.1f}")
    ctrl.separator()

    state["brush_size"] = ctrl.slider_int("Brush Size", 1, 16, default=1)
    state["brush_color"] = ctrl.color_picker("Brush Color",
                                              default=(1.0, 0.3, 0.1))
    state["bg_color"] = ctrl.color_picker("Background",
                                           default=(0.1, 0.1, 0.1))
    ctrl.separator()

    if ctrl.button("Clear", width=120):
        clear_canvas()

    state["show_grid"] = ctrl.checkbox("Show Grid (8px)", default=False)

    ctrl.separator()
    ctrl.text_wrapped(
        "左键在画布上画画，右键擦除（涂背景色）。"
        "上方可以调整笔刷大小和颜色。"
    )


@draw_panel.on_frame
def handle_drawing():
    if not ui.is_item_hovered():
        refresh_display()
        return

    mx, my = ui.get_mouse_pos()

    # 屏幕坐标 → tensor 像素坐标
    wp_x, wp_y = ui.get_window_pos()
    win_w, win_h = ui.get_window_size()
    content_x = wp_x + 8
    content_y = wp_y + 26
    content_w = win_w - 16
    content_h = win_h - 34

    u = max(0.0, min(1.0, (mx - content_x) / max(content_w, 1)))
    v = max(0.0, min(1.0, (my - content_y) / max(content_h, 1)))
    px = int(u * (W - 1))
    py = int(v * (H - 1))

    painting = ui.is_mouse_clicked(0) or ui.is_mouse_dragging(0, 0.0)
    erasing  = ui.is_mouse_clicked(1) or ui.is_mouse_dragging(1, 0.0)

    if painting:
        r, g, b = state["brush_color"]
        draw_brush(py, px, state["brush_size"], r, g, b)
    elif erasing:
        r, g, b = state["bg_color"]
        draw_brush(py, px, state["brush_size"], r, g, b)

    refresh_display()


view.run()
```

## 刚才发生了什么？

### 屏幕坐标 → tensor 像素

这是本例的核心技巧。画布图片被拉伸到整个面板，所以一个 128×128 的
tensor 可能显示为 600×500 的屏幕像素。当鼠标在屏幕位置 `(mx, my)` 时，
你需要算出它对应 tensor 的哪个像素：

```python
# 面板窗口位置和大小
wp_x, wp_y = ui.get_window_pos()
win_w, win_h = ui.get_window_size()

# 内容区域（减去标题栏和内边距）
content_x = wp_x + 8
content_y = wp_y + 26

# 归一化到 [0, 1]
u = (mx - content_x) / content_w
v = (my - content_y) / content_h

# 缩放到像素索引
px = int(u * (W - 1))
py = int(v * (H - 1))
```

这和光栅化器里把归一化设备坐标转换为像素坐标是完全相同的思路 —— 只是方向反过来。

### is_item_hovered + is_mouse_dragging

ImGui 会追踪鼠标在哪个控件上方。当画布图片被绘制后（面板自动完成），
`is_item_hovered()` 告诉你光标是否在图片上。

`is_mouse_dragging(button, threshold)` 在按钮按住且鼠标移动超过
`threshold` 像素时返回 `True`。设 `threshold=0.0` 表示立即触发 ——
相当于"按钮是否正在被按住"。

组合起来就实现了连续绘画：

```python
if ui.is_item_hovered():
    if ui.is_mouse_dragging(0, 0.0):
        draw_brush(py, px, ...)
```

### 底图 + 显示刷新

我们保持一个单独的 `backing` tensor（RGB，没有 alpha）来存储实际的像素画。
每帧把它复制到显示 tensor，然后可选地叠加网格线。
这样可以避免网格线随时间"烤进"画作里。

```python
canvas_tensor[:, :, :3] = backing          # 复制画作
if state["show_grid"]:
    canvas_tensor[::8, :, :3] += 0.15      # 提亮网格行
    canvas_tensor[:, ::8, :3] += 0.15      # 提亮网格列
```

## 要点

1. **`create_tensor` 是双向的** —— GPU 可以往里写（模拟），
   Python 也可以往里写（用户交互）。Vultorch 显示 tensor 里面的内容，不管是谁写的。

2. **屏幕→tensor 映射** —— `get_mouse_pos()` 给屏幕像素坐标；
   减去面板原点、除以面板大小、乘以 tensor 尺寸。
   和图形学里的 UV 坐标是一样的数学。

3. **`is_item_hovered` + `is_mouse_dragging`** —— 交互式控件的标准模式。
   先检查悬停，再检查按钮状态。

4. **底图模式** —— 如果要在用户数据上叠加装饰（网格、标记、十字线），
   把原始数据存在单独的 tensor 里，每帧合成。否则装饰会不断累积。

5. **`filter="nearest"`** —— 像素画的必需品。没有它 128×128 的画布
   看起来像模糊的水彩画，而不是清晰的方块。

!!! tip "提示"
    这个屏幕→像素映射技巧可以直接用来做标注工具 —— 分割 mask、
    包围盒、关键点标注。tensor 本身就是标签图。
