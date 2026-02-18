# 12 — 神经渲染工作站

!!! tip "压轴大戏"
    这是最后一个示例——一个功能完备的神经渲染 IDE，只用一个 Python
    脚本。用它作为模板，发布你自己研究的精美交互式 demo。

## 本章新朋友

| 名称 | 干什么的 | 类比 |
|------|---------|------|
| **打开图像** | 操作系统文件对话框加载任意 PNG/JPEG/BMP 训练 | `plt.imread()` + 重新训练，但只需一个按钮 |
| **位置编码** | 傅里叶特征 $[\sin(2^l \pi x), \cos(2^l \pi x)]$ | NeRF 学习高频细节的关键技巧 |
| **架构滑块** | 运行时改变隐藏层宽度、深度、PE 级别 | 编辑模型配置不用重启 |
| **误差增益** | 放大误差热力图以看到细微差异 | 给图片拉对比度 |
| **暂停/恢复** | 停止训练但 UI 保持响应 | `Ctrl-C` 但不杀进程 |
| **保存快照** | `Canvas.save()` 把当前 GPU tensor 写成 PNG | 实时 GPU 数据的 `plt.savefig()` |
| **训练速度** | 迭代/秒 计数器 | 内置的 `tqdm` |

---

## 为什么没有 depth 头？

之前的版本有 RGB + depth 双头 MLP。但这个示例重建的是 2D 图像——
没有有意义的深度可以预测。所以我们保持简洁：一个头，三个输出（RGB），
直接用 MSE 对比目标图像。当你做真正的 NeRF 时，再把密度/深度头加回
你自己的模型里。

---

## 位置编码 —— NeRF 核心技巧

原始 `(x, y)` 坐标只能表示低频函数。位置编码把它们提升到高维空间：

```python
def positional_encoding(x, L):
    if L == 0:
        return x
    freqs = 2.0 ** torch.arange(L, device=x.device)
    xf = (x.unsqueeze(-1) * freqs * math.pi).reshape(*x.shape[:-1], -1)
    return torch.cat([x, xf.sin(), xf.cos()], dim=-1)
```

当 $L = 6$ 时，2D 输入变成 $2 + 2 \times 6 \times 2 = 26$ 维。
这让 MLP 可以学到锐利的边缘和精细的纹理。**PE Levels** 滑块让你
实时看到区别——设为 0，观察预测变模糊。

---

## 从操作系统打开图像

点击 **Open Image...**，弹出系统原生文件对话框（通过
`tkinter.filedialog`）。对话框在后台线程运行，渲染循环不会阻塞：

```python
def open_file_dialog():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tga *.hdr"),
                   ("All files", "*.*")])
    root.destroy()
    return path if path else None
```

当对话框返回路径时，主循环接收：

```python
if S["pending_image"]:
    gt, coords, target, H, W = load_image(S["pending_image"], RES)
    S["img_name"] = Path(S["pending_image"]).name
    # 重建显示 tensor，重置模型...
```

这个模式可以在任何 Vultorch demo 中复用：把阻塞的 OS 调用放在线程里，
通过共享变量传回结果。

---

## 实时架构调优

三个滑块控制网络结构：

```python
new_h  = ctrl.slider_int("Hidden",    32, 256, default=128)
new_l  = ctrl.slider_int("Layers",     2,   8, default=4)
new_pe = ctrl.slider_int("PE Levels",  0,  10, default=6)
```

改变任何滑块会设置 `arch_dirty = True` 并显示黄色警告。然后点击
**Apply & Reset** 重建模型：

```python
if S["arch_dirty"]:
    ctrl.text_colored(1, 0.8, 0, 1, "  Architecture changed!")
    if ctrl.button("Apply & Reset", width=170):
        model = make_model(S["hidden"], S["layers"], S["pe"])
        optimizer = make_optimizer(...)
```

这种两步模式（滑块 → 确认按钮）防止你拖动滑块时每帧都重建模型。

---

## 可调增益的误差热力图

```python
err = (gt - pr).abs().mean(dim=-1)       # 逐像素 L1
err_t[:, :, :3] = apply_turbo(
    (err * S["err_gain"]).clamp_(0, 1))
```

**Error Gain** 滑块（1–20 倍）放大细微误差。增益为 1 时大部分像素
看起来是蓝色的；增益为 15 时你能精确看到模型在哪里还在挣扎。

---

## 暂停、快照、速度

| 功能 | 原理 |
|------|------|
| **暂停** | `if not S["paused"]:` 跳过训练循环；窗口、控件、显示继续运行 |
| **保存快照** | `rgb_cv.save("snapshot_pred.png")` 通过 stb_image_write 把 canvas 的 GPU tensor 写到磁盘 |
| **速度计数器** | `(当前迭代 - 上次迭代) / 时间差` 每 0.5 秒测一次 |

---

## 指标面板

```python
@met_pan.on_frame
def draw_met():
    met_pan.text(f"Loss: {S['loss']:.6f}   PSNR: {S['psnr']:.1f} dB   "
                 f"Speed: {S['its_sec']:.0f} it/s")
    met_pan.separator()
    if S["loss_h"]:
        met_pan.plot(S["loss_h"], label="##loss",
                     overlay=f"loss {S['loss']:.5f}", height=70)
    if S["psnr_h"]:
        met_pan.plot(S["psnr_h"], label="##psnr",
                     overlay=f"PSNR {S['psnr']:.1f} dB", height=70)
```

`panel.plot()` 从 Python 列表渲染迷你折线图。保留最近 500 个值，
给你一个滚动的实时图表——内置在训练窗口里的 TensorBoard。

---

## 完整代码

```python title="examples/12_neural_workstation.py"
--8<-- "examples/12_neural_workstation.py"
```

---

## 刚才发生了什么？

用一个 Python 文件，你构建了一个完整的、可发布的 demo：

1. 从操作系统**打开任意图像**
2. 可调频率级别的**位置编码**
3. **实时架构调优** —— 改变隐藏层大小、深度、PE 级别
4. **三种损失函数** —— MSE、L1、Huber —— 运行时可切换
5. **三种优化器** —— Adam、SGD、AdamW —— 热切换
6. 带 turbo 色图和可调增益的**误差热力图**
7. **暂停/恢复** 不杀进程
8. **保存快照** —— 预测和误差导出为 PNG
9. **训练速度**计数器（迭代/秒）
10. **Loss & PSNR 曲线** —— 滚动实时图表

没有 matplotlib。没有 TensorBoard。没有 Jupyter。没有浏览器。
一个窗口，一个脚本，一切以 GPU 速度同步。

**这就是「Vultorch = 你的神经渲染 IDE」的样子。**

---

## 核心要点

| 概念 | 代码 | 用途 |
|------|------|------|
| 位置编码 | `positional_encoding(x, L)` | 傅里叶特征学习高频细节 |
| 文件对话框 | 线程中的 `tkinter.filedialog` | 不阻塞地加载任意图片 |
| 架构滑块 | `slider_int("Hidden", ...)` | 实时拓扑调优 |
| 误差增益 | `(err * gain).clamp_(0, 1)` | 放大细微重建误差 |
| 暂停 | `checkbox("Pause Training")` | 冻结训练，UI 保持活跃 |
| 快照 | `canvas.save("file.png")` | 把 GPU tensor 写到磁盘 |
| 速度 | `(it - it_last) / dt` | 迭代/秒 计数器 |
| step()/end_step() | 训练循环拥有渲染权 | 你控制外层循环 |

!!! success "恭喜！"
    你已经完成了全部 12 个 Vultorch 教程。现在你拥有了将实时、
    GPU 加速的可视化集成到神经渲染研究工作流中所需的一切工具——
    以及发布精美交互式 demo 的完整模板。
