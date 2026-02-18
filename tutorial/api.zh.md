# API 参考文档

`vultorch` 包所有公开类和函数的完整参考。

---

## 模块级属性

### `vultorch.__version__`

```python
__version__: str
```

包版本字符串（例如 `"0.5.0"`）。

### `vultorch.HAS_CUDA`

```python
HAS_CUDA: bool
```

如果原生扩展模块编译时启用了 CUDA 则为 `True`。为 `False` 时，所有张量显示将回退到 CPU 暂存缓冲区（host-visible `memcpy`）。

---

## 核心函数

### `vultorch.show()`

```python
def show(
    tensor: torch.Tensor,
    *,
    name: str = "tensor",
    width: float = 0,
    height: float = 0,
    filter: str = "linear",
    window: Window | None = None,
) -> None
```

在当前 ImGui 上下文中显示张量。

**参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `tensor` | `torch.Tensor` | *(必需)* | CUDA 或 CPU 张量。dtype：`float32`、`float16` 或 `uint8`。形状：`(H, W)` 或 `(H, W, C)`，C ∈ {1, 3, 4}。|
| `name` | `str` | `"tensor"` | 唯一标签，用于同时显示多个张量时的缓存。|
| `width` | `float` | `0` | 显示宽度（像素）。`0` = 自动适应张量大小。|
| `height` | `float` | `0` | 显示高度（像素）。`0` = 自动适应张量大小。|
| `filter` | `str` | `"linear"` | 采样滤波器：`"nearest"` 或 `"linear"`。|
| `window` | `Window \| None` | `None` | 目标窗口。默认使用 `Window._current`。|

**行为：**

- 1 通道和 3 通道张量会自动扩展为 RGBA。
- RGBA 扩展缓冲区按 `name` 缓存，避免每帧分配内存。
- CUDA：使用零拷贝 GPU→GPU 路径。CPU：使用 host-visible 暂存缓冲区。
- `uint8` 张量除以 255；`float16` 张量转换为 `float32`。

**异常：** 如果没有活跃的 `Window` 则抛出 `RuntimeError`。

---

### `vultorch.create_tensor()`

```python
def create_tensor(
    height: int,
    width: int,
    channels: int = 4,
    device: str = "cuda:0",
    *,
    name: str = "tensor",
    window: Window | None = None,
) -> torch.Tensor
```

分配 Vulkan 共享的 CUDA 张量，实现真正的零拷贝显示。

**参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `height` | `int` | *(必需)* | 张量高度（像素）。|
| `width` | `int` | *(必需)* | 张量宽度（像素）。|
| `channels` | `int` | `4` | 通道数：1、3 或 4。|
| `device` | `str` | `"cuda:0"` | CUDA 设备字符串，或 `"cpu"`。|
| `name` | `str` | `"tensor"` | 纹理槽名称（必须与 `show(..., name=...)` 匹配）。|
| `window` | `Window \| None` | `None` | 目标窗口。默认使用 `Window._current`。|

**返回：** 形状为 `(height, width, channels)` 的 `torch.Tensor`。

!!! note
    只有 `channels=4` 才能通过 Vulkan 外部内存实现真正的零拷贝。对于 1 或 3 通道，返回普通 CUDA 张量，`show()` 会处理 RGBA 扩展并进行 GPU→GPU 拷贝。

**异常：** 如果没有活跃的 `Window` 则抛出 `RuntimeError`。

---

### `vultorch.imread()`

```python
def imread(
    path: str,
    *,
    channels: int = 4,
    size: tuple[int, int] | None = None,
    device: str = "cuda",
    shared: bool = False,
    name: str = "tensor",
    window: Window | None = None,
) -> torch.Tensor
```

将图片文件加载为 `float32` 张量。内部使用 stb_image —— 不需要 PIL 或 numpy。

**参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `path` | `str` | *（必需）* | 文件路径（PNG、JPG、BMP、TGA、HDR 等）。|
| `channels` | `int` | `4` | 期望通道数：1（灰度）、3（RGB）或4（RGBA）。|
| `size` | `tuple[int, int] \| None` | `None` | 可选 `(高度, 宽度)`，使用双线性插值缩放。|
| `device` | `str` | `"cuda"` | 目标设备（`"cuda"` 或 `"cpu"`）。|
| `shared` | `bool` | `False` | 为 `True` 时通过 `create_tensor` 分配，实现零拷贝显示。|
| `name` | `str` | `"tensor"` | 纹理槽名称（仅在 `shared=True` 时使用）。|
| `window` | `Window \| None` | `None` | 目标窗口（仅在 `shared=True` 时使用）。|

**返回：** 形状为 `(H, W, C)` 的 `torch.Tensor`，值范围 `[0, 1]`。

**示例：**

```python
import vultorch
gt = vultorch.imread("photo.png", channels=3, device="cuda")
```

---

### `vultorch.imwrite()`

```python
def imwrite(
    path: str,
    tensor: torch.Tensor,
    *,
    channels: int = 0,
    size: tuple[int, int] | None = None,
    quality: int = 95,
) -> None
```

将张量保存为图片文件。格式由扩展名推断。

**参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `path` | `str` | *（必需）* | 输出文件路径。扩展名选择格式：`.png`、`.jpg`、`.bmp`、`.tga`、`.hdr`。|
| `tensor` | `torch.Tensor` | *（必需）* | `(H, W)`、`(H, W, 1)`、`(H, W, 3)` 或 `(H, W, 4)` 张量。|
| `channels` | `int` | `0` | 覆盖输出通道数。`0` = 使用张量自身的通道数。|
| `size` | `tuple[int, int] \| None` | `None` | 可选 `(高度, 宽度)`，保存前缩放。|
| `quality` | `int` | `95` | JPEG 质量（1–100）。其他格式忽略此参数。|

**行为：**

- `.hdr` 写入 32 位浮点数据；其他格式量化为 8 位。
- 如果张量通道数多于 `channels`，多余通道被丢弃。如果少于，缺少的通道会被填充（alpha → 1.0）。
- 张量会被移到 CPU 并转换为 `float32` 后再写入。

**示例：**

```python
vultorch.imwrite("output.png", pred_tensor, channels=3)
vultorch.imwrite("output.jpg", pred_tensor, quality=90)
```

---

## 类

### `vultorch.Window`

```python
class Window:
    _current: Window | None   # 单例引用

    def __init__(self, title: str = "Vultorch",
                 width: int = 1280, height: int = 720) -> None: ...
```

Vulkan + SDL3 + ImGui 引擎的高层封装。创建 `Window` 会自动将其设置为 `show()` 和 `create_tensor()` 的当前目标。

#### 方法

| 方法 | 签名 | 描述 |
|------|------|------|
| `poll()` | `→ bool` | 处理操作系统事件。窗口应关闭时返回 `False`。|
| `begin_frame()` | `→ bool` | 开始新的 ImGui 帧。帧被跳过（最小化）时返回 `False`。|
| `end_frame()` | `→ None` | 提交帧到 GPU 并呈现。|
| `activate()` | `→ None` | 将此窗口设为模块级辅助函数的当前目标。|
| `upload_tensor(tensor, *, name)` | `→ None` | 上传张量用于显示（CUDA 或 CPU）。|
| `get_texture_id(name)` | `→ int` | 指定名称张量的 ImGui 纹理 ID。|
| `get_texture_size(name)` | `→ (int, int)` | 指定名称张量的 `(宽度, 高度)`。|
| `destroy()` | `→ None` | 释放所有 GPU / 窗口资源。可安全多次调用。|

#### 属性

| 属性 | 类型 | 描述 |
|------|------|------|
| `tensor_texture_id` | `int` | 默认 `"tensor"` 槽的 ImGui 纹理 ID。|
| `tensor_size` | `(int, int)` | 默认 `"tensor"` 槽的 `(宽度, 高度)`。|

#### 使用示例

```python
import vultorch
from vultorch import ui

win = vultorch.Window("Demo", 1280, 720)
while win.poll():
    if not win.begin_frame():
        continue
    ui.begin("Panel", True, 0)
    vultorch.show(tensor)
    ui.end()
    win.end_frame()
win.destroy()
```

---

### `vultorch.Camera`

```python
class Camera:
    azimuth: float     # 水平角度（弧度），默认 0.0
    elevation: float   # 垂直角度（弧度），默认 0.6
    distance: float    # 到目标的距离，默认 3.0
    target: tuple      # (x, y, z) 注视点，默认 (0, 0, 0)
    fov: float         # 视场角（度），默认 45.0
```

`SceneView` 使用的轨道相机参数。调用 `reset()` 恢复默认值。

---

### `vultorch.Light`

```python
class Light:
    direction: tuple   # (x, y, z)，默认 (0.3, -1.0, 0.5)
    color: tuple       # (r, g, b)，默认 (1, 1, 1)
    intensity: float   # 默认 1.0
    ambient: float     # 环境光项，默认 0.15
    specular: float    # 高光项，默认 0.5
    shininess: float   # Blinn-Phong 指数，默认 32.0
    enabled: bool      # 默认 True
```

`SceneView` 使用的 Blinn-Phong 方向光参数。

---

### `vultorch.SceneView`

```python
class SceneView:
    def __init__(self, name: str = "SceneView",
                 width: int = 800, height: int = 600,
                 msaa: int = 4) -> None: ...
```

3D 张量查看器 — 在带光照的平面上渲染张量，支持轨道相机和 MSAA。

#### 属性

| 属性 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `name` | `str` | `"SceneView"` | ImGui 窗口标签。|
| `camera` | `Camera` | *(自动)* | 轨道相机（拖拽旋转）。|
| `light` | `Light` | *(自动)* | 方向光源。|
| `background` | `tuple` | `(0.12, 0.12, 0.14)` | 背景颜色 `(r, g, b)`。|
| `msaa` | `int` | `4` | 多重采样抗锯齿级别（1/2/4/8）。|

#### 方法

| 方法 | 描述 |
|------|------|
| `set_tensor(tensor)` | 上传张量到场景纹理。|
| `render()` | 处理鼠标交互，渲染场景，并作为 ImGui 图像显示。|

#### 使用示例

```python
scene = vultorch.SceneView("3D 视图", 800, 600, msaa=4)
# 在帧循环中：
scene.set_tensor(tensor)
scene.render()
```

---

## 声明式 API

声明式 API 提供了更高层的抽象，用于构建多面板可视化应用。

### `vultorch.View`

```python
class View:
    def __init__(self, title: str = "Vultorch",
                 width: int = 1280, height: int = 720) -> None: ...
```

支持自动停靠布局的顶层窗口。

#### 方法

| 方法 | 签名 | 描述 |
|------|------|------|
| `panel(name, *, side, width)` | `→ Panel` | 创建或获取可停靠面板。`side`：`"left"` / `"right"` / `"bottom"` / `"top"` / `None`。|
| `on_frame(fn)` | `→ fn` | 装饰器 — 注册每帧回调函数。|
| `run()` | `→ None` | 阻塞式事件循环。|
| `step()` | `→ bool` | 非阻塞：处理一帧。关闭时返回 `False`。|
| `end_step()` | `→ None` | 结束由 `step()` 开始的帧。|
| `close()` | `→ None` | 销毁窗口。|

#### 属性

| 属性 | 类型 | 描述 |
|------|------|------|
| `fps` | `float` | 当前每秒帧数。|
| `time` | `float` | 已过秒数。|
| `window` | `Window` | 底层 `Window` 实例。|

#### 使用示例 — 阻塞模式

```python
view = vultorch.View("Demo", 1280, 720)
view.panel("Viewer").canvas("img").bind(tensor)

@view.on_frame
def update():
    speed = controls.slider("Speed", 0, 10)
    tensor[:,:,0] = (x + view.time * speed).sin()

view.run()
```

#### 使用示例 — 训练循环

```python
view = vultorch.View("Train", 1024, 768)
output = view.panel("Output").canvas("result")
for epoch in range(100):
    result = model(input)
    output.bind(result)
    if not view.step():
        break
    view.end_step()
view.close()
```

---

### `vultorch.Panel`

```python
class Panel:
    # 通过 View.panel() 创建 — 不直接实例化
```

包含画布和控件的可停靠面板。

#### 画布工厂

| 方法 | 签名 | 描述 |
|------|------|------|
| `canvas(name, *, filter, fit)` | `→ Canvas` | 创建命名画布。`filter`：`"linear"` / `"nearest"`。`fit`：自动填充面板空间。|

#### 面板回调

| 方法 | 签名 | 描述 |
|------|------|------|
| `on_frame(fn)` | `→ fn` | 装饰器 — 注册在面板 ImGui 窗口内运行的每帧回调。|

#### 布局

| 方法 | 描述 |
|------|------|
| `row()` | 上下文管理器 — 将子控件并排放置。|

#### 控件

所有控件方法自动管理跨帧状态。

| 方法 | 签名 | 描述 |
|------|------|------|
| `text(text)` | `→ None` | 静态文本。|
| `text_colored(r, g, b, a, text)` | `→ None` | 有颜色的文本。|
| `text_wrapped(text)` | `→ None` | 自动换行文本。|
| `separator()` | `→ None` | 水平分隔线。|
| `button(label, width, height)` | `→ bool` | 按钮。点击时返回 `True`。`width`/`height` 默认 `0`（自动大小）。|
| `checkbox(label, *, default)` | `→ bool` | 带状态切换的复选框。|
| `slider(label, min, max, *, default)` | `→ float` | 浮点滑块。|
| `slider_int(label, min, max, *, default)` | `→ int` | 整数滑块。|
| `color_picker(label, *, default)` | `→ (r, g, b)` | 颜色选择器（3 浮点元组）。|
| `combo(label, items, *, default)` | `→ int` | 下拉组合框。返回选中索引。|
| `input_text(label, *, default, max_length)` | `→ str` | 文本输入框。|
| `plot(values, *, label, overlay, width, height)` | `→ None` | 浮点列表的折线图。|
| `progress(fraction, *, overlay)` | `→ None` | 进度条（0.0 – 1.0）。|

---

### `vultorch.Canvas`

```python
class Canvas:
    # 通过 Panel.canvas() 创建 — 不直接实例化
```

将绑定张量渲染为 ImGui 图像的显示表面。

#### 方法

| 方法 | 签名 | 描述 |
|------|------|------|
| `bind(tensor)` | `→ Canvas` | 绑定张量用于显示。返回 `self` 以支持链式调用。|
| `alloc(height, width, channels, device)` | `→ torch.Tensor` | 分配 Vulkan 共享内存并自动绑定。返回张量。|
| `save(path, *, channels, size, quality)` | `→ None` | 通过 `imwrite()` 将绑定的张量保存为图片文件。|

#### 属性

| 属性 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `filter` | `str` | `"linear"` | `"linear"` 或 `"nearest"`。|
| `fit` | `bool` | `True` | 自动填充可用面板空间。|

---

## ImGui 绑定 (`vultorch.ui`)

`vultorch.ui` 子模块暴露了 Dear ImGui 函数（docking 分支）。所有函数直接映射到对应的 ImGui C++ 接口。

### 窗口

```python
ui.begin(name: str, opened: bool = True, flags: int = 0) -> tuple[bool, bool]
ui.end() -> None
ui.begin_child(id: str, width=0.0, height=0.0, child_flags=0, window_flags=0) -> bool
ui.end_child() -> None
```

### 文本

```python
ui.text(text: str) -> None
ui.text_colored(r, g, b, a, text: str) -> None
ui.text_disabled(text: str) -> None
ui.text_wrapped(text: str) -> None
ui.label_text(label: str, text: str) -> None
ui.bullet_text(text: str) -> None
```

### 按钮

```python
ui.button(label: str, width=0.0, height=0.0) -> bool
ui.small_button(label: str) -> bool
ui.invisible_button(id: str, width, height) -> bool
ui.arrow_button(id: str, direction: int) -> bool
ui.radio_button(label: str, active: bool) -> bool
```

### 输入

```python
ui.checkbox(label, value: bool) -> bool
ui.slider_float(label, value, min=0.0, max=1.0, format="%.3f") -> float
ui.slider_float2(label, v1, v2, min, max) -> tuple[float, float]
ui.slider_float3(label, v1, v2, v3, min, max) -> tuple[float, float, float]
ui.slider_float4(label, v1, v2, v3, v4, min, max) -> tuple
ui.slider_int(label, value, min=0, max=100) -> int
ui.slider_angle(label, value, min=-360, max=360) -> float
ui.drag_float(label, value, speed=1.0) -> float
ui.drag_float2(label, v1, v2, speed=1.0) -> tuple
ui.drag_float3(label, v1, v2, v3, speed=1.0) -> tuple
ui.drag_int(label, value, speed=1.0) -> int
ui.input_float(label, value) -> float
ui.input_float2(label, v1, v2) -> tuple
ui.input_float3(label, v1, v2, v3) -> tuple
ui.input_float4(label, v1, v2, v3, v4) -> tuple
ui.input_int(label, value) -> int
ui.input_text(label, text, max_length=256) -> str
ui.input_text_multiline(label, text, max_length=1024) -> str
```

### 颜色

```python
ui.color_edit3(label, r, g, b, flags=0) -> tuple[float, float, float]
ui.color_edit4(label, r, g, b, a, flags=0) -> tuple[float, float, float, float]
ui.color_picker3(label, r, g, b, flags=0) -> tuple[float, float, float]
ui.color_picker4(label, r, g, b, a, flags=0) -> tuple
```

### 选择

```python
ui.combo(label, current: int, items: list[str]) -> int
ui.listbox(label, current: int, items: list[str], height_items=-1) -> int
ui.tree_node(label: str) -> bool
ui.tree_pop() -> None
ui.collapsing_header(label: str) -> bool
ui.selectable(label: str, selected: bool = False) -> bool
```

### 标签页

```python
ui.begin_tab_bar(id: str) -> bool
ui.end_tab_bar() -> None
ui.begin_tab_item(label: str) -> bool
ui.end_tab_item() -> None
```

### 显示

```python
ui.progress_bar(fraction, sx=-1.0, sy=0.0, overlay="")
ui.image(texture_id: int, width, height, uv0x=0, uv0y=0, uv1x=1, uv1y=1)
ui.image_button(id: str, texture_id: int, width, height) -> bool
ui.plot_lines(label, values: list[float], offset=0, overlay="", ...)
ui.plot_histogram(label, values: list[float], offset=0, overlay="", ...)
```

### 布局

```python
ui.separator()
ui.same_line(offset=0.0, spacing=-1.0)
ui.new_line()
ui.spacing()
ui.dummy(width, height)
ui.indent(width=0.0)
ui.unindent(width=0.0)
ui.begin_group()
ui.end_group()
ui.push_item_width(width)
ui.pop_item_width()
ui.columns(count=1, id=None, border=True)
ui.next_column()
```

### 表格

```python
ui.begin_table(id: str, columns: int, flags=0) -> bool
ui.end_table()
ui.table_next_row(flags=0, min_row_height=0.0)
ui.table_next_column() -> bool
ui.table_set_column_index(index: int) -> bool
ui.table_setup_column(label: str, flags=0, init_width=0.0)
ui.table_headers_row()
```

### 菜单

```python
ui.begin_main_menu_bar() -> bool
ui.end_main_menu_bar()
ui.begin_menu_bar() -> bool
ui.end_menu_bar()
ui.begin_menu(label: str, enabled=True) -> bool
ui.end_menu()
ui.menu_item(label: str, shortcut="", selected=False, enabled=True) -> bool
```

### 弹出窗口

```python
ui.open_popup(id: str)
ui.begin_popup(id: str) -> bool
ui.begin_popup_modal(name: str, flags=0) -> bool
ui.end_popup()
ui.close_current_popup()
```

### 提示框

```python
ui.begin_tooltip()
ui.end_tooltip()
ui.set_tooltip(text: str)
```

### ID 栈

```python
ui.push_id_str(id: str)
ui.push_id_int(id: int)
ui.pop_id()
ui.get_id(id: str) -> int
```

### 样式

```python
ui.push_style_color(idx: int, r, g, b, a)
ui.pop_style_color(count=1)
ui.push_style_var_float(idx: int, value: float)
ui.push_style_var_vec2(idx: int, x: float, y: float)
ui.pop_style_var(count=1)
ui.style_colors_dark()
ui.style_colors_light()
ui.style_colors_classic()
```

### 光标与窗口信息

```python
ui.get_cursor_pos() -> tuple[float, float]
ui.set_cursor_pos(x, y)
ui.get_content_region_avail() -> tuple[float, float]
ui.get_window_size() -> tuple[float, float]
ui.get_window_pos() -> tuple[float, float]
ui.set_next_window_pos(x, y, cond=0)
ui.set_next_window_size(width, height, cond=0)
```

### 停靠

```python
ui.dock_space_over_viewport(flags=0) -> int
ui.dock_space(id: int, sx=0.0, sy=0.0, flags=0) -> int
ui.set_next_window_dock_id(dock_id: int, cond=0)
ui.dock_builder_add_node(node_id=0, flags=0) -> int
ui.dock_builder_remove_node(node_id: int)
ui.dock_builder_set_node_size(node_id, width, height)
ui.dock_builder_set_node_pos(node_id, x, y)
ui.dock_builder_split_node(node_id, split_dir, ratio) -> tuple[int, int]
ui.dock_builder_dock_window(window_name: str, node_id: int)
ui.dock_builder_finish(node_id: int)
ui.dock_builder_get_node(node_id: int) -> int
```

### 绘图

```python
ui.draw_line(x1, y1, x2, y2, col=0xFFFFFFFF, thickness=1.0)
ui.draw_rect(x1, y1, x2, y2, col=0xFFFFFFFF)
ui.draw_rect_filled(x1, y1, x2, y2, col=0xFFFFFFFF)
ui.draw_circle(cx, cy, radius, col=0xFFFFFFFF)
ui.draw_circle_filled(cx, cy, radius, col=0xFFFFFFFF)
ui.draw_text(x, y, col: int, text: str)
ui.bg_draw_image(texture_id, x1, y1, x2, y2)
```

### 输入状态

```python
ui.is_item_hovered() -> bool
ui.is_item_active() -> bool
ui.is_item_clicked() -> bool
ui.is_item_focused() -> bool
ui.is_item_edited() -> bool
ui.is_item_deactivated_after_edit() -> bool
ui.get_mouse_pos() -> tuple[float, float]
ui.is_mouse_clicked(button: int) -> bool
ui.is_mouse_double_clicked(button: int) -> bool
ui.is_mouse_dragging(button: int, lock_threshold=-1.0) -> bool
ui.get_mouse_drag_delta(button=0, lock_threshold=-1.0) -> tuple[float, float]
ui.is_key_pressed(key: int) -> bool
ui.is_key_down(key: int) -> bool
```

### 工具函数

```python
ui.get_io_framerate() -> float
ui.get_io_delta_time() -> float
ui.get_time() -> float
ui.get_frame_count() -> int
ui.get_display_size() -> tuple[float, float]
ui.col32(r: int, g: int, b: int, a: int = 255) -> int
ui.show_demo_window()
ui.show_metrics_window()
```

---

## 内部辅助函数

### `vultorch._normalize_tensor()`

```python
def _normalize_tensor(tensor) -> tuple[Tensor, int, int, int]
```

规范化张量的 dtype 和形状以用于显示。返回 `(tensor, height, width, channels)`。

- `uint8` → `float32`（÷ 255），`float16` → `float32`。
- 接受 2D `(H, W)` 和 3D `(H, W, C)`，C ∈ {1, 3, 4}。
- 不支持的 dtype、形状或通道数会抛出 `ValueError`。
