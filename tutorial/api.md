# API Reference

Complete reference for all public classes and functions in the `vultorch` package.

---

## Module-level Attributes

### `vultorch.__version__`

```python
__version__: str
```

Package version string (e.g. `"0.5.0"`).

### `vultorch.HAS_CUDA`

```python
HAS_CUDA: bool
```

`True` if the native extension was compiled with CUDA support. When `False`, all tensor display falls back to CPU staging (host-visible `memcpy`).

---

## Core Functions

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

Display a tensor in the current ImGui context.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor` | `torch.Tensor` | *(required)* | CUDA or CPU tensor. dtype: `float32`, `float16`, or `uint8`. Shape: `(H, W)` or `(H, W, C)` with C ∈ {1, 3, 4}. |
| `name` | `str` | `"tensor"` | Unique label for caching when showing multiple tensors. |
| `width` | `float` | `0` | Display width in pixels. `0` = auto-fit to tensor. |
| `height` | `float` | `0` | Display height in pixels. `0` = auto-fit to tensor. |
| `filter` | `str` | `"linear"` | Sampling filter: `"nearest"` or `"linear"`. |
| `window` | `Window \| None` | `None` | Target window. Defaults to `Window._current`. |

**Behavior:**

- 1-channel and 3-channel tensors are automatically expanded to RGBA.
- RGBA expansion buffers are cached per `name` to avoid per-frame allocation.
- On CUDA: uses zero-copy GPU→GPU path. On CPU: uses host-visible staging buffer.
- `uint8` tensors are divided by 255; `float16` tensors are converted to `float32`.

**Raises:** `RuntimeError` if no active `Window` exists.

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

Allocate a Vulkan-shared CUDA tensor for true zero-copy display.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `height` | `int` | *(required)* | Tensor height in pixels. |
| `width` | `int` | *(required)* | Tensor width in pixels. |
| `channels` | `int` | `4` | Number of channels: 1, 3, or 4. |
| `device` | `str` | `"cuda:0"` | CUDA device string, or `"cpu"`. |
| `name` | `str` | `"tensor"` | Texture slot name (must match `show(..., name=...)`). |
| `window` | `Window \| None` | `None` | Target window. Defaults to `Window._current`. |

**Returns:** `torch.Tensor` of shape `(height, width, channels)`.

!!! note
    Only `channels=4` gives true zero-copy via Vulkan external memory. For 1 or 3 channels, a regular CUDA tensor is returned and `show()` handles RGBA expansion with a GPU→GPU copy.

**Raises:** `RuntimeError` if no active `Window` exists.

---

## Classes

### `vultorch.Window`

```python
class Window:
    _current: Window | None   # singleton reference

    def __init__(self, title: str = "Vultorch",
                 width: int = 1280, height: int = 720) -> None: ...
```

High-level wrapper around the Vulkan + SDL3 + ImGui engine. Creating a `Window` automatically makes it the current target for `show()` and `create_tensor()`.

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `poll()` | `→ bool` | Process OS events. Returns `False` when the window should close. |
| `begin_frame()` | `→ bool` | Begin a new ImGui frame. Returns `False` if the frame was skipped (minimized). |
| `end_frame()` | `→ None` | Submit the frame to the GPU and present. |
| `activate()` | `→ None` | Make this window the current target for module-level helpers. |
| `upload_tensor(tensor, *, name)` | `→ None` | Upload a tensor for display (CUDA or CPU). |
| `get_texture_id(name)` | `→ int` | ImGui texture ID for a named tensor. |
| `get_texture_size(name)` | `→ (int, int)` | `(width, height)` for a named tensor. |
| `destroy()` | `→ None` | Release all GPU / window resources. Safe to call multiple times. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `tensor_texture_id` | `int` | ImGui texture ID of the default `"tensor"` slot. |
| `tensor_size` | `(int, int)` | `(width, height)` of the default `"tensor"` slot. |

#### Usage

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
    azimuth: float     # horizontal angle (radians), default 0.0
    elevation: float   # vertical angle (radians), default 0.6
    distance: float    # distance from target, default 3.0
    target: tuple      # (x, y, z) look-at point, default (0, 0, 0)
    fov: float         # field of view (degrees), default 45.0
```

Orbit camera parameters used by `SceneView`. Call `reset()` to restore defaults.

---

### `vultorch.Light`

```python
class Light:
    direction: tuple   # (x, y, z), default (0.3, -1.0, 0.5)
    color: tuple       # (r, g, b), default (1, 1, 1)
    intensity: float   # default 1.0
    ambient: float     # ambient term, default 0.15
    specular: float    # specular term, default 0.5
    shininess: float   # Blinn-Phong exponent, default 32.0
    enabled: bool      # default True
```

Blinn-Phong directional light parameters used by `SceneView`.

---

### `vultorch.SceneView`

```python
class SceneView:
    def __init__(self, name: str = "SceneView",
                 width: int = 800, height: int = 600,
                 msaa: int = 4) -> None: ...
```

3D tensor viewer — renders a tensor on a lit plane with orbit camera and MSAA.

#### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"SceneView"` | ImGui window label. |
| `camera` | `Camera` | *(auto)* | Orbit camera (drag to rotate). |
| `light` | `Light` | *(auto)* | Directional light. |
| `background` | `tuple` | `(0.12, 0.12, 0.14)` | Background color `(r, g, b)`. |
| `msaa` | `int` | `4` | Multi-sample anti-aliasing level (1/2/4/8). |

#### Methods

| Method | Description |
|--------|-------------|
| `set_tensor(tensor)` | Upload a tensor to the scene's texture. |
| `render()` | Process mouse interaction, render the scene, and display as an ImGui image. |

#### Usage

```python
scene = vultorch.SceneView("3D View", 800, 600, msaa=4)
# inside frame loop:
scene.set_tensor(tensor)
scene.render()
```

---

## Declarative API

The declarative API provides a higher-level abstraction for building multi-panel visualization apps.

### `vultorch.View`

```python
class View:
    def __init__(self, title: str = "Vultorch",
                 width: int = 1280, height: int = 720) -> None: ...
```

Top-level window with automatic docking layout.

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `panel(name, *, side, width)` | `→ Panel` | Create or retrieve a dockable panel. `side`: `"left"` / `"right"` / `None`. |
| `on_frame(fn)` | `→ fn` | Decorator — register a per-frame callback. |
| `run()` | `→ None` | Blocking event loop. |
| `step()` | `→ bool` | Non-blocking: process one frame. Returns `False` on close. |
| `end_step()` | `→ None` | Finish the frame started by `step()`. |
| `close()` | `→ None` | Destroy the window. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `fps` | `float` | Current frames per second. |
| `time` | `float` | Elapsed time in seconds. |
| `window` | `Window` | Underlying `Window` instance. |

#### Usage — Blocking

```python
view = vultorch.View("Demo", 1280, 720)
view.panel("Viewer").canvas("img").bind(tensor)

@view.on_frame
def update():
    speed = controls.slider("Speed", 0, 10)
    tensor[:,:,0] = (x + view.time * speed).sin()

view.run()
```

#### Usage — Training Loop

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
    # Created via View.panel() — not instantiated directly
```

A dockable panel containing canvases and widgets.

#### Canvas Factory

| Method | Signature | Description |
|--------|-----------|-------------|
| `canvas(name, *, filter, fit)` | `→ Canvas` | Create a named canvas. `filter`: `"linear"` / `"nearest"`. `fit`: auto-fill panel space. |

#### Layout

| Method | Description |
|--------|-------------|
| `row()` | Context manager — place child widgets side-by-side. |

#### Widgets

All widget methods manage state automatically across frames.

| Method | Signature | Description |
|--------|-----------|-------------|
| `text(text)` | `→ None` | Static text. |
| `text_colored(r, g, b, a, text)` | `→ None` | Colored text. |
| `text_wrapped(text)` | `→ None` | Auto-wrapping text. |
| `separator()` | `→ None` | Horizontal separator line. |
| `button(label)` | `→ bool` | Button. Returns `True` when clicked. |
| `checkbox(label, *, default)` | `→ bool` | Checkbox with stateful toggle. |
| `slider(label, min, max, *, default)` | `→ float` | Float slider. |
| `slider_int(label, min, max, *, default)` | `→ int` | Integer slider. |
| `color_picker(label, *, default)` | `→ (r, g, b)` | Color picker (3-float tuple). |
| `combo(label, items, *, default)` | `→ int` | Dropdown combo box. Returns selected index. |
| `input_text(label, *, default, max_length)` | `→ str` | Text input field. |
| `plot(values, *, label, overlay, width, height)` | `→ None` | Line plot from a list of floats. |
| `progress(fraction, *, overlay)` | `→ None` | Progress bar (0.0 – 1.0). |

---

### `vultorch.Canvas`

```python
class Canvas:
    # Created via Panel.canvas() — not instantiated directly
```

A display surface that renders a bound tensor as an ImGui image.

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `bind(tensor)` | `→ Canvas` | Bind a tensor for display. Returns `self` for chaining. |
| `alloc(height, width, channels, device)` | `→ torch.Tensor` | Allocate Vulkan-shared memory and auto-bind. Returns the tensor. |

#### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `filter` | `str` | `"linear"` | `"linear"` or `"nearest"`. |
| `fit` | `bool` | `True` | Auto-fill available panel space. |

---

## ImGui Bindings (`vultorch.ui`)

The `vultorch.ui` submodule exposes Dear ImGui functions (docking branch). All functions map directly to their ImGui C++ counterparts.

### Windows

```python
ui.begin(name: str, opened: bool = True, flags: int = 0) -> tuple[bool, bool]
ui.end() -> None
ui.begin_child(id: str, width=0.0, height=0.0, child_flags=0, window_flags=0) -> bool
ui.end_child() -> None
```

### Text

```python
ui.text(text: str) -> None
ui.text_colored(r, g, b, a, text: str) -> None
ui.text_disabled(text: str) -> None
ui.text_wrapped(text: str) -> None
ui.label_text(label: str, text: str) -> None
ui.bullet_text(text: str) -> None
```

### Buttons

```python
ui.button(label: str, width=0.0, height=0.0) -> bool
ui.small_button(label: str) -> bool
ui.invisible_button(id: str, width, height) -> bool
ui.arrow_button(id: str, direction: int) -> bool
ui.radio_button(label: str, active: bool) -> bool
```

### Inputs

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

### Colors

```python
ui.color_edit3(label, r, g, b, flags=0) -> tuple[float, float, float]
ui.color_edit4(label, r, g, b, a, flags=0) -> tuple[float, float, float, float]
ui.color_picker3(label, r, g, b, flags=0) -> tuple[float, float, float]
ui.color_picker4(label, r, g, b, a, flags=0) -> tuple
```

### Selection

```python
ui.combo(label, current: int, items: list[str]) -> int
ui.listbox(label, current: int, items: list[str], height_items=-1) -> int
ui.tree_node(label: str) -> bool
ui.tree_pop() -> None
ui.collapsing_header(label: str) -> bool
ui.selectable(label: str, selected: bool = False) -> bool
```

### Tabs

```python
ui.begin_tab_bar(id: str) -> bool
ui.end_tab_bar() -> None
ui.begin_tab_item(label: str) -> bool
ui.end_tab_item() -> None
```

### Display

```python
ui.progress_bar(fraction, sx=-1.0, sy=0.0, overlay="")
ui.image(texture_id: int, width, height, uv0x=0, uv0y=0, uv1x=1, uv1y=1)
ui.image_button(id: str, texture_id: int, width, height) -> bool
ui.plot_lines(label, values: list[float], offset=0, overlay="", ...)
ui.plot_histogram(label, values: list[float], offset=0, overlay="", ...)
```

### Layout

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

### Tables

```python
ui.begin_table(id: str, columns: int, flags=0) -> bool
ui.end_table()
ui.table_next_row(flags=0, min_row_height=0.0)
ui.table_next_column() -> bool
ui.table_set_column_index(index: int) -> bool
ui.table_setup_column(label: str, flags=0, init_width=0.0)
ui.table_headers_row()
```

### Menus

```python
ui.begin_main_menu_bar() -> bool
ui.end_main_menu_bar()
ui.begin_menu_bar() -> bool
ui.end_menu_bar()
ui.begin_menu(label: str, enabled=True) -> bool
ui.end_menu()
ui.menu_item(label: str, shortcut="", selected=False, enabled=True) -> bool
```

### Popups

```python
ui.open_popup(id: str)
ui.begin_popup(id: str) -> bool
ui.begin_popup_modal(name: str, flags=0) -> bool
ui.end_popup()
ui.close_current_popup()
```

### Tooltips

```python
ui.begin_tooltip()
ui.end_tooltip()
ui.set_tooltip(text: str)
```

### ID Stack

```python
ui.push_id_str(id: str)
ui.push_id_int(id: int)
ui.pop_id()
ui.get_id(id: str) -> int
```

### Style

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

### Cursor & Window Info

```python
ui.get_cursor_pos() -> tuple[float, float]
ui.set_cursor_pos(x, y)
ui.get_content_region_avail() -> tuple[float, float]
ui.get_window_size() -> tuple[float, float]
ui.get_window_pos() -> tuple[float, float]
ui.set_next_window_pos(x, y, cond=0)
ui.set_next_window_size(width, height, cond=0)
```

### Docking

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

### Drawing

```python
ui.draw_line(x1, y1, x2, y2, col=0xFFFFFFFF, thickness=1.0)
ui.draw_rect(x1, y1, x2, y2, col=0xFFFFFFFF)
ui.draw_rect_filled(x1, y1, x2, y2, col=0xFFFFFFFF)
ui.draw_circle(cx, cy, radius, col=0xFFFFFFFF)
ui.draw_circle_filled(cx, cy, radius, col=0xFFFFFFFF)
ui.draw_text(x, y, col: int, text: str)
ui.bg_draw_image(texture_id, x1, y1, x2, y2)
```

### Input State

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

### Utility

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

## Internal Helpers

### `vultorch._normalize_tensor()`

```python
def _normalize_tensor(tensor) -> tuple[Tensor, int, int, int]
```

Normalize tensor dtype and shape for display. Returns `(tensor, height, width, channels)`.

- Converts `uint8` → `float32` (÷ 255), `float16` → `float32`.
- Accepts 2D `(H, W)` and 3D `(H, W, C)` with C ∈ {1, 3, 4}.
- Raises `ValueError` for unsupported dtype, shape, or channel count.
