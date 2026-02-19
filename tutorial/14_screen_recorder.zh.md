# 14 — 屏幕录制

> **示例文件：** `examples/14_screen_recorder.py`

训练了半天的模型，屏幕上跑出漂亮的可视化——然后呢？截图只能抓一帧，屏幕录制
软件又笨重。如果 Canvas 自己就能录呢？

这一章介绍 Vultorch 内置的录制功能。任何 Canvas 都可以直接把画面录成
**GIF**（通过 Pillow）——不需要装别的软件，一个 `pip install Pillow` 搞定。

## 这次玩什么

一个迷幻万花筒动画，左边有控制面板：

| 区域 | 内容 |
|------|------|
| 左侧 | **控制** — 录制/停止按钮、质量滑块、速度滑块 |
| 右侧 | **动画** — GPU 生成的万花筒图案 |

按下录制按钮，随便拨弄速度滑块，再按停止——GIF 就出现在工作目录了。

## 新朋友

| 新东西 | 干什么的 | 为什么重要 |
|--------|---------|-----------|
| `canvas.start_recording(path)` | 开始逐帧录制到 `.gif` | 只需要 Pillow |
| `canvas.stop_recording()` | 停止录制、写入文件 | 返回保存路径 |
| `canvas.is_recording` | 查看是否在录制 | 显示红色 `● REC` 指示 |
| `panel.record_button(canvas, path)` | 一行代码的录制/停止按钮 | 自带红色高亮切换 |
| `quality`（0–1） | 控制每帧颜色数 | 值越低颜色越少、文件越小 |

## 录制原理

录制开启后，每一帧：

1. 把绑定的 tensor 从 GPU 拷贝到 CPU（如果在 CUDA 上）
2. 转换为 uint8 RGB
3. 按 `quality` 做颜色量化（2–256 色）
4. 追加到 PIL Image 列表，停止时一次性写入 GIF

这意味着录制会多一次 GPU→CPU 拷贝。对 256×256 的画布来说几乎不影响帧率；
4K 分辨率可能需要降分再录。

## 质量参数

`quality` 从 0 到 1：

| quality | 颜色数 | 用途 |
|---------|--------|------|
| 1.0 | 256 | 全质量，文件最大 |
| 0.8 | ~205 | 默认，效果好 |
| 0.5 | ~129 | 紧凑，略有色带 |
| 0.1 | ~27 | 很小，明显抖动 |

```python
canvas.start_recording("out.gif", fps=15, quality=0.5)
```

## 依赖

```bash
pip install Pillow
```

## 一行搞定

最快的方法是 `panel.record_button`：

```python
@ctrl.on_frame
def draw():
    ctrl.record_button(my_canvas, "output.gif", fps=30, quality=0.8)
```

搞定。空闲时显示普通按钮写着"Record"，录制中变红写着"Stop Recording"。

## 代码录制

要做自动化或 headless 扫参，可以直接调方法：

```python
canvas.start_recording("sweep.gif", fps=15, quality=0.6)
for lr in learning_rates:
    train(lr)
    canvas.bind(result)
    view.step()
    view.end_step()
canvas.stop_recording()
```

有窗和 headless 模式都能用——特别适合跑 paper 的附图动画。

## 完整代码

```python
import math
import torch
import vultorch

if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"
H, W = 256, 256

# Pre-compute coordinate grids
ys = torch.linspace(-1.0, 1.0, H, device=device)
xs = torch.linspace(-1.0, 1.0, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")
rr = (xx ** 2 + yy ** 2).sqrt()
angle = torch.atan2(yy, xx)

# View + panels
view = vultorch.View("14 - Screen Recorder", 1024, 700)
ctrl = view.panel("Controls", side="left", width=0.30)
anim_panel = view.panel("Animation")

# Shared display tensor
display = vultorch.create_tensor(H, W, channels=4, device=device,
                                 name="anim", window=view.window)
display[:, :, 3] = 1.0
canvas = anim_panel.canvas("anim")
canvas.bind(display)

state = {"speed": 2.0, "quality": 0.8}


@view.on_frame
def animate():
    t = view.time * state["speed"]
    k = 6
    r = rr * 4.0 + t
    a = (angle * k).remainder(math.pi * 2) + t * 0.5
    display[:, :, 0] = (r.sin() * a.cos() * 0.5 + 0.5).clamp(0, 1)
    display[:, :, 1] = ((r + 2.094).sin() * (a + 1.047).cos() * 0.5 + 0.5).clamp(0, 1)
    display[:, :, 2] = ((r + 4.189).sin() * (a + 2.094).cos() * 0.5 + 0.5).clamp(0, 1)


@ctrl.on_frame
def draw_ctrl():
    ctrl.text(f"FPS: {view.fps:.0f}")
    ctrl.separator()
    state["quality"] = ctrl.slider("Quality", 0.0, 1.0, default=0.8)
    state["speed"] = ctrl.slider("Speed", 0.5, 10.0, default=2.0)
    ctrl.separator()

    ctrl.record_button(canvas, "recording.gif", fps=30,
                       quality=state["quality"])

    ctrl.separator()
    if canvas.is_recording:
        ctrl.text_colored(1, 0.3, 0.3, 1, "● RECORDING")
    else:
        ctrl.text("● Idle")


view.run()
```

## 试试看

```bash
python examples/14_screen_recorder.py
```

调质量和速度，点录制，等几秒，点停止。看看工作目录下的 `recording.gif`。


