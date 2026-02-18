# 04 — 康威生命游戏

> **示例文件：** `examples/04_conway.py`

Loss 曲线，训练可视化 —— 前面太正经了。
歇一歇，来搞个好玩的：康威生命游戏，全跑在 GPU 上，零拷贝显示，
右边一排按钮和滑条让你当上帝。

更重要的是，这一章要说明一件事：`create_tensor` 不只是给训练用的。
**任何 GPU 计算**都可以通过 Vultorch 的共享显存显示出来 —— 模拟、
程序化生成、物理仿真，只要是 CUDA 上跑的，都行。

## 这次玩什么

一个 256×256 的元胞自动机，带控制面板：

| 区域 | 内容 |
|------|------|
| 左侧 | **Controls** — 播放/暂停、单步、速度滑条、经典图案按钮、颜色选择器 |
| 右侧 | **Grid** — 模拟画面，像素级精确（`filter="nearest"`） |

整个模拟跑在 GPU 上。显示用的 tensor 通过 `create_tensor` 分配，
零拷贝 —— 网格数据不经过 CPU。

## 新朋友

| 新东西 | 干什么用 | 为什么重要 |
|--------|----------|-----------|
| `filter="nearest"` | 每个像素显示为清晰的方块，不做模糊 | 没有它的话双线性插值会把格子边界糊掉。跟 `plt.imshow(data, interpolation='nearest')` 一个道理 |
| `side="left"` 侧边栏 | 把面板放到左边，占窗口 22% | 给你一个永久的控制条 |
| `@panel.on_frame` | 面板级控件回调 | 按钮、滑条、颜色选择器都放这里 |
| `panel.button(标签)` | 一个可点击的按钮 | 被点击的那一帧返回 `True` |
| `with panel.row():` | 把接下来的控件放在**同一行** | 默认控件是一行一个的（跟 `print()` 一样）。用 `with panel.row():` 可以把两个按钮并排。就是个 Python `with` 块 —— 块里的东西都放同一行 |
| `panel.color_picker` | RGB 颜色选择器 | 点色块打开调色板 |
| 循环 padding + conv2d | GPU 并行数邻居 | 整个模拟就是一次卷积 |

## 模拟的核心技巧

数康威生命游戏里的邻居个数，本质上就是一次 2D 卷积，用一个 3×3 的卷积核（中心为 0）：

```
1 1 1
1 0 1
1 1 1
```

PyTorch 的 `F.conv2d` 一个 GPU kernel 调用就搞定 —— 不需要循环，
不需要逐像素的逻辑。circular padding 让边缘环绕，飞行器飞出右边
会从左边冒出来。

```python
kernel = torch.tensor([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], dtype=torch.float32, device=device)
padded = F.pad(inp, (1, 1, 1, 1), mode='circular')
neighbours = F.conv2d(padded, kernel.reshape(1, 1, 3, 3)).squeeze()
```

然后规则就是两个布尔 mask：

```python
survive = (grid == 1) & ((neighbours == 2) | (neighbours == 3))
birth   = (grid == 0) & (neighbours == 3)
grid[:] = (survive | birth).float()
```

## 完整代码

```python
import torch
import vultorch

if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"

# ── 网格参数 ──────────────────────────────────────────────────────
GRID_H, GRID_W = 256, 256

# ── 视图 + 面板 ──────────────────────────────────────────────────
view = vultorch.View("04 - Conway's Game of Life", 1024, 768)
grid_panel = view.panel("Grid")
ctrl_panel = view.panel("Controls", side="left", width=0.22)

# ── 显示 tensor（RGBA，零拷贝）────────────────────────────────────
display = vultorch.create_tensor(GRID_H, GRID_W, channels=4,
                                 device=device, name="grid",
                                 window=view.window)
canvas = grid_panel.canvas("grid", filter="nearest")
canvas.bind(display)

# ── 模拟状态 ─────────────────────────────────────────────────────
grid = torch.zeros(GRID_H, GRID_W, dtype=torch.float32, device=device)

state = {
    "running": False,
    "generation": 0,
    "speed": 1,
    "prob": 0.3,
    "alive_color": (0.0, 1.0, 0.4),
    "dead_color": (0.05, 0.05, 0.08),
}


def randomize():
    grid[:] = (torch.rand(GRID_H, GRID_W, device=device) < state["prob"]).float()
    state["generation"] = 0

def clear():
    grid.zero_()
    state["generation"] = 0

def step_simulation():
    kernel = torch.tensor([[1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]], dtype=torch.float32, device=device)
    inp = grid.unsqueeze(0).unsqueeze(0)
    k = kernel.unsqueeze(0).unsqueeze(0)
    padded = torch.nn.functional.pad(inp, (1, 1, 1, 1), mode='circular')
    neighbours = torch.nn.functional.conv2d(padded, k).squeeze()

    survive = (grid == 1) & ((neighbours == 2) | (neighbours == 3))
    birth = (grid == 0) & (neighbours == 3)
    grid[:] = (survive | birth).float()
    state["generation"] += 1

def grid_to_display():
    alive_r, alive_g, alive_b = state["alive_color"]
    dead_r, dead_g, dead_b = state["dead_color"]
    display[:, :, 0] = dead_r + (alive_r - dead_r) * grid
    display[:, :, 1] = dead_g + (alive_g - dead_g) * grid
    display[:, :, 2] = dead_b + (alive_b - dead_b) * grid
    display[:, :, 3] = 1.0

randomize()


@view.on_frame
def update():
    if state["running"]:
        for _ in range(state["speed"]):
            step_simulation()
    grid_to_display()


@ctrl_panel.on_frame
def draw_controls():
    ctrl_panel.text(f"Generation: {state['generation']}")
    ctrl_panel.text(f"Alive cells: {int(grid.sum().item())}")
    ctrl_panel.text(f"FPS: {view.fps:.1f}")
    ctrl_panel.separator()

    with ctrl_panel.row():
        label = "Pause" if state["running"] else "Play"
        if ctrl_panel.button(label, width=80):
            state["running"] = not state["running"]
        if ctrl_panel.button("Step", width=80):
            step_simulation()

    with ctrl_panel.row():
        if ctrl_panel.button("Randomize", width=80):
            randomize()
        if ctrl_panel.button("Clear", width=80):
            clear()

    ctrl_panel.separator()
    state["speed"] = ctrl_panel.slider_int("Speed", 1, 20, default=1)
    state["prob"] = ctrl_panel.slider("Cell Probability", 0.05, 0.8,
                                       default=0.3)

    ctrl_panel.separator()
    ctrl_panel.text("Colors")
    state["alive_color"] = ctrl_panel.color_picker(
        "Alive", default=(0.0, 1.0, 0.4))
    state["dead_color"] = ctrl_panel.color_picker(
        "Dead", default=(0.05, 0.05, 0.08))

    ctrl_panel.separator()
    ctrl_panel.text("Patterns")
    with ctrl_panel.row():
        if ctrl_panel.button("Glider", width=80):
            clear()
            grid[1, 2] = 1; grid[2, 3] = 1
            grid[3, 1] = 1; grid[3, 2] = 1; grid[3, 3] = 1
        if ctrl_panel.button("Pulsar", width=80):
            clear()
            # ... 放置 Pulsar 图案 ...
        if ctrl_panel.button("Gosper Gun", width=100):
            clear()
            # ... 放置 Gosper 滑翔机枪 ...

    ctrl_panel.separator()
    ctrl_panel.text_wrapped(
        "点 Play 开始，或者 Step 单步。"
        "Randomize 重新随机填充。"
    )


view.run()
```

*（完整的示例文件中包含所有图案放置的辅助函数。）*

## 刚才发生了什么？

1. **网格** — 一个普通的 `(256, 256)` float32 CUDA tensor。
   `1.0` 就是活的，`0.0` 就是死的。不需要类，不需要花哨的数据结构 —— 就一个 tensor。

2. **模拟** — `step_simulation()` 用 `F.conv2d` 配合 circular padding 数邻居，
   然后用布尔 mask 应用出生/存活规则。整代只需要两个 GPU kernel。

3. **显示** — `create_tensor` 分配 Vulkan/CUDA 共享显存。
   `grid_to_display()` 在存活和死亡颜色之间插值，写进去就行。零拷贝上屏。

4. **控制面板** — `@ctrl_panel.on_frame` 把所有控件画在 Controls 面板内部。
   `panel.button()`、`panel.slider_int()`、`panel.color_picker()`、
   `with panel.row()` 让布局紧凑。状态就用一个普通 Python dict 管理。

## 要点

1. **`create_tensor` 不只是给训练用的** — 任何 GPU 计算只要产出
   类似图像的 tensor，都可以零拷贝显示。

2. **`filter="nearest"`** — 像素画 / 网格模拟的必选项。
   没有它的话双线性插值会把格子边界糊成一坨。
   跟 `plt.imshow(data, interpolation='nearest')` 一个意思 ——
   你想看到真实的像素。

3. **卷积 = 数邻居** — 一个小技巧，用一个 GPU kernel 替代嵌套 Python 循环。
   即使网格很大，游戏也能跑到几百 FPS。

4. **面板控件** — 在 `@panel.on_frame` 里调用
   `panel.button()`、`panel.slider_int()`、`panel.color_picker()`。
   每个调用创建一个控件，自上而下排列，跟 `print()` 输出一样。
   不需要写任何定位代码。

5. **`with panel.row():`** — 默认控件每行一个。
   把几个控件调用包在 `with panel.row():` 里，它们就会并排在同一行。
   就是个 Python `with` 块，没什么复杂的。

6. **经典图案** — Glider、Pulsar、Gosper Gun 按钮展示了怎么通过
   直接往 grid tensor 里写值来设初始条件。

!!! tip "提示"
    Speed 滑条拉到 20，看网格以每帧 20 代的速度演化。
    现代 GPU 上帧率依然稳稳 60+。

!!! note "说明"
    网格环绕是因为用了 `mode='circular'` padding。
    滑翔机从右边飞出去会从左边冒出来。
