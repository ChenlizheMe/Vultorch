# 05 — 图片查看器

> **示例文件：** `examples/05_image_viewer.py`

目前为止我们一直在凭空造 tensor —— 渐变、棋盘格、神经网络的输出。
很酷，但总有那么一天，你想看一张**真正的图片**。就是那种安安静静躺在
硬盘上的 `.png` 文件。像个正常人一样。

"用 PIL 不就行了。" 好，然后你还需要 `torchvision.transforms`，
然后需要 `numpy`，然后需要 `cv2.cvtColor` 因为不知道谁把 RGB 和 BGR
搞反了，然后你打开了三个 Stack Overflow 标签页，试图搞明白为什么图片
上下颠倒而且偏绿。

Vultorch 内置了图片读写。一个函数进，一个函数出。
不需要 PIL，不需要 OpenCV，不需要怀疑人生。

## 新朋友

| 新东西 | 干什么用 | 写法 |
|--------|----------|------|
| **imread** | 把图片文件读成 CUDA tensor | `vultorch.imread("photo.png")` |
| **imwrite** | 把 tensor 保存为图片文件 | `vultorch.imwrite("out.png", t)` |
| **Canvas.save()** | 保存画布绑定的 tensor | `canvas.save("out.png")` |
| **panel.combo()** | 下拉选择菜单 | `panel.combo("选项", ["A","B"])` |
| **panel.input_text()** | 文本输入框 | `panel.input_text("路径")` |
| **canvas.filter** | 采样模式（`"linear"` / `"nearest"`） | `canvas.filter = "nearest"` |

## 这次玩什么

一个迷你图片查看器：加载一张图，从下拉菜单选变换，
用滑条调亮度 / 对比度，然后把结果存盘。

| 左侧 | 右侧（两个画布） |
|------|------------------|
| **Controls** — 变换选择、亮度/对比度滑条、滤波切换、保存 | **原图**（上） |
| | **变换后**（下） |

## 完整代码

```python
from pathlib import Path

import torch
import vultorch

if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"

# ── 加载图片 ──────────────────────────────────────────────────────
img_path = Path(__file__).resolve().parents[1] / "docs" / "images" / "pytorch_logo.png"
original = vultorch.imread(img_path, channels=3, device=device)
H, W, C = original.shape

# 用于变换的工作副本（原图永远不动）
transformed = original.clone()

# ── 视图 + 面板 ──────────────────────────────────────────────────
view = vultorch.View("05 - Image Viewer", 1024, 768)
ctrl = view.panel("Controls", side="left", width=0.28)
img_panel = view.panel("Image")

canvas_orig = img_panel.canvas("Original")
canvas_orig.bind(original)

canvas_xform = img_panel.canvas("Transformed")
canvas_xform.bind(transformed)

# ── 状态 ─────────────────────────────────────────────────────────
TRANSFORMS = [
    "None",
    "Horizontal Flip",
    "Vertical Flip",
    "Grayscale",
    "Invert",
    "Sepia",
]

state = {
    "brightness": 0.0,
    "contrast": 1.0,
    "last_transform": -1,
    "last_brightness": None,
    "last_contrast": None,
}


def apply_transform(img, idx):
    if idx == 0:    return img.clone()
    elif idx == 1:  return img.flip(1)               # 水平翻转
    elif idx == 2:  return img.flip(0)               # 垂直翻转
    elif idx == 3:                                    # 灰度
        gray = img[:,:,0]*0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114
        return gray.unsqueeze(-1).expand_as(img).contiguous()
    elif idx == 4:  return 1.0 - img                 # 反色
    elif idx == 5:                                    # 复古色调
        r = img[:,:,0]*0.393 + img[:,:,1]*0.769 + img[:,:,2]*0.189
        g = img[:,:,0]*0.349 + img[:,:,1]*0.686 + img[:,:,2]*0.168
        b = img[:,:,0]*0.272 + img[:,:,1]*0.534 + img[:,:,2]*0.131
        return torch.stack([r, g, b], dim=-1).clamp(0, 1)
    return img.clone()


def apply_brightness_contrast(img, brightness, contrast):
    return ((img - 0.5) * contrast + 0.5 + brightness).clamp(0, 1)


@ctrl.on_frame
def draw_controls():
    ctrl.text(f"Image: {img_path.name}")
    ctrl.text(f"Size: {W} × {H}")
    ctrl.text(f"FPS: {view.fps:.1f}")
    ctrl.separator()

    # 变换选择器
    ctrl.text("Transform")
    xform_idx = ctrl.combo("##transform", TRANSFORMS, default=0)

    ctrl.separator()

    # 亮度 / 对比度
    ctrl.text("Adjustments")
    brightness = ctrl.slider("Brightness", -1.0, 1.0, default=0.0)
    contrast   = ctrl.slider("Contrast",    0.0, 3.0, default=1.0)

    changed = (xform_idx != state["last_transform"]
               or brightness != state["last_brightness"]
               or contrast   != state["last_contrast"])

    if changed:
        result = apply_transform(original, xform_idx)
        result = apply_brightness_contrast(result, brightness, contrast)
        transformed[:] = result
        state["last_transform"]  = xform_idx
        state["last_brightness"] = brightness
        state["last_contrast"]   = contrast

    ctrl.separator()

    # 滤波切换
    ctrl.text("Sampling Filter")
    filter_idx = ctrl.combo("##filter", ["Linear", "Nearest"], default=0)
    canvas_orig.filter  = "nearest" if filter_idx == 1 else "linear"
    canvas_xform.filter = "nearest" if filter_idx == 1 else "linear"

    ctrl.separator()

    # 保存
    ctrl.text("Save Output")
    save_path = ctrl.input_text("Path", default="output.png")

    if ctrl.button("Save Image", width=140):
        try:
            canvas_xform.save(save_path)
            state["save_msg"] = f"Saved to {save_path}"
        except Exception as e:
            state["save_msg"] = f"Error: {e}"

    if "save_msg" in state:
        ctrl.text_wrapped(state["save_msg"])


view.run()
```

## 刚才发生了什么？

### imread — 告别依赖地狱

```python
original = vultorch.imread(img_path, channels=3, device=device)
```

一行搞定。返回 `(H, W, 3)` 的 float32 CUDA tensor，值域 `[0, 1]`。
支持 PNG、JPEG、BMP、TGA、HDR、PSD、GIF（第一帧）。
底层用的是 `stb_image` —— 不需要任何 Python 图像库。

可选参数：

- `channels=4` —— 强制输出 RGBA。
- `size=(256, 256)` —— 加载后缩放（双线性插值）。
- `device="cpu"` —— 如果你想留在 CPU 上。
- `shared=True` —— 用 `create_tensor` 分配，零拷贝显示。

### combo — 下拉菜单

```python
xform_idx = ctrl.combo("##transform", TRANSFORMS, default=0)
```

显示一个下拉菜单，里面是列表中的选项。返回选中项的**索引**（int）。
状态由面板自动管理 —— 只需要传 `default=` 设初始值就行。

`##` 前缀会隐藏 ImGui 的标签文字（`##` 后面的内容只作为内部 ID）。
当你不想在控件旁边显示文字的时候很有用。

### input_text — 文本输入

```python
save_path = ctrl.input_text("Path", default="output.png")
```

返回当前字符串。输个文件名，点保存就行。
默认 `max_length=256` —— 写个路径绰绰有余。

### Canvas.save() — 一行导出

```python
canvas_xform.save(save_path)
```

把当前画布绑定的 tensor 保存到文件。格式根据扩展名自动判断
（`.png`、`.jpg`、`.bmp`、`.tga`、`.hdr`）。
底层调用的是 `vultorch.imwrite()`。

### filter — nearest vs linear

```python
canvas_orig.filter = "nearest"   # 像素级精确，放大后有锯齿
canvas_orig.filter = "linear"    # 双线性插值，平滑
```

可以随时切换采样滤波器。试试在图片被拉伸的时候切换 ——
`"nearest"` 让你看到原始像素，`"linear"` 把它们糊成平滑渐变。
做科研可视化（分割 mask、attention map）的时候，基本上永远选 `"nearest"`。

## 要点

1. **`imread` / `imwrite`** — 零依赖的图片读写。直接读到 CUDA tensor，
   直接从 tensor 写文件。不需要 PIL，不需要 numpy，
   不需要跟 `cv2.cvtColor` 斗智斗勇。

2. **`combo`** — 下拉选择控件。返回 int 索引。适合模式切换、
   预设选择、枚举类型的选项。

3. **`input_text`** — 自由文本输入。适合文件路径、模型名、
   实验标签这类需要打字的场景。

4. **`Canvas.save()`** — 一行把绑定的 tensor 存为图片。
   扩展名决定格式。

5. **按需重算** — 只在滑条或下拉值真正变化时才重新计算变换。
   通过检查 `changed` 来避免每帧都白白烧 GPU。

!!! tip "提示"
    `imread` 支持 `size=(H, W)` 参数，可以在加载时缩放。
    你的图片是 4K 的，但只需要 256×256 预览？一个参数搞定。

!!! note "说明"
    `imwrite` 接受 `[0, 1]` 的 float32 tensor，也接受 `[0, 255]` 的 uint8 tensor。
    格式转换它自己处理。
