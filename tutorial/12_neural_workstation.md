# 12 — Neural Rendering Workstation

!!! tip "The capstone"
    This is the final example — a fully-featured neural rendering IDE
    in a single Python script.  Use it as a template for publishing
    polished, interactive demos of your own research.

## New friends in this chapter

| Name | What it does | Analogy |
|------|-------------|---------|
| **Open Image** | OS file dialog loads any PNG/JPEG/BMP into training | `plt.imread()` + re-training, but one button |
| **Positional encoding** | Fourier features $[\sin(2^l \pi x), \cos(2^l \pi x)]$ | The NeRF trick for learning high-frequency detail |
| **Architecture sliders** | Change hidden width, depth, PE levels at runtime | Editing your model config without restarting |
| **Error gain** | Amplify the error heatmap to see subtle differences | Cranking contrast on an image |
| **Pause / resume** | Stop training while the UI stays alive | `Ctrl-C` without killing the process |
| **Save snapshots** | `Canvas.save()` dumps current GPU tensor to PNG | `plt.savefig()` for live GPU data |
| **Training speed** | Iterations/sec counter | Like `tqdm` but built-in |

---

## Why no depth head?

Previous versions had a dual-head MLP with RGB + depth.  But this
example reconstructs a 2D image — there's no meaningful depth to
predict.  So we keep it clean: one head, three outputs (RGB), direct
MSE against the target image.  When you build a real NeRF, you'd add
the density/depth head back to your own model.

---

## Positional encoding — the NeRF trick

Raw `(x, y)` coordinates can only represent low-frequency functions.
Positional encoding lifts them into a higher-dimensional space:

```python
def positional_encoding(x, L):
    if L == 0:
        return x
    freqs = 2.0 ** torch.arange(L, device=x.device)
    xf = (x.unsqueeze(-1) * freqs * math.pi).reshape(*x.shape[:-1], -1)
    return torch.cat([x, xf.sin(), xf.cos()], dim=-1)
```

With $L = 6$, a 2D input becomes $2 + 2 \times 6 \times 2 = 26$
dimensions.  This lets the MLP learn sharp edges and fine textures.
The **PE Levels** slider lets you see the difference live — set it to 0
and watch the prediction turn blurry.

---

## Opening images from your OS

Click **Open Image...** and a native file dialog appears (via
`tkinter.filedialog`).  The dialog runs in a background thread so the
render loop never blocks:

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

When the dialog returns a path, the main loop picks it up:

```python
if S["pending_image"]:
    gt, coords, target, H, W = load_image(S["pending_image"], RES)
    S["img_name"] = Path(S["pending_image"]).name
    # Recreate display tensors and reset model...
```

This is a pattern you can reuse in any Vultorch demo: thread the
blocking OS call, pass the result back through a shared variable.

---

## Live architecture tuning

Three sliders control the network structure:

```python
new_h  = ctrl.slider_int("Hidden",    32, 256, default=128)
new_l  = ctrl.slider_int("Layers",     2,   8, default=4)
new_pe = ctrl.slider_int("PE Levels",  0,  10, default=6)
```

Changing any slider sets `arch_dirty = True` and shows a yellow
warning.  You then click **Apply & Reset** to rebuild the model:

```python
if S["arch_dirty"]:
    ctrl.text_colored(1, 0.8, 0, 1, "  Architecture changed!")
    if ctrl.button("Apply & Reset", width=170):
        model = make_model(S["hidden"], S["layers"], S["pe"])
        optimizer = make_optimizer(...)
```

This two-step pattern (slider → confirm button) prevents the model
from being destroyed on every frame while you're dragging the slider.

---

## Error heatmap with adjustable gain

```python
err = (gt - pr).abs().mean(dim=-1)       # per-pixel L1
err_t[:, :, :3] = apply_turbo(
    (err * S["err_gain"]).clamp_(0, 1))
```

The **Error Gain** slider (1–20×) amplifies subtle errors.  At gain 1
most pixels look blue; at gain 15 you'll see exactly where the model
is still struggling.

---

## Pause, snapshot, speed

| Feature | How it works |
|---------|-------------|
| **Pause** | `if not S["paused"]:` skips the training loop; the window, controls, and display keep running |
| **Save Snapshots** | `rgb_cv.save("snapshot_pred.png")` writes the canvas's GPU tensor to disk via stb_image_write |
| **Speed counter** | `(iters_now - iters_then) / elapsed` measured every 0.5 seconds |

---

## Metrics panel

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

`panel.plot()` renders a sparkline from a Python list.  The last 500
values are kept, giving you a scrolling live chart — your TensorBoard,
built into the training window.

---

## Full code

```python title="examples/12_neural_workstation.py"
--8<-- "examples/12_neural_workstation.py"
```

---

## What just happened?

In one Python file you built a complete, publishable demo:

1. **Open any image** from your operating system via file dialog
2. **Positional encoding** with adjustable frequency levels
3. **Live architecture tuning** — change hidden size, depth, PE levels
4. **Three loss functions** — MSE, L1, Huber — switchable at runtime
5. **Three optimizers** — Adam, SGD, AdamW — hot-swappable
6. **Error heatmap** with turbo colormap and adjustable gain
7. **Pause / resume** without killing the process
8. **Save snapshots** of prediction and error to PNG
9. **Training speed** counter (iterations/sec)
10. **Loss & PSNR curves** — scrolling live charts

No matplotlib.  No TensorBoard.  No Jupyter.  No web browser.
One window, one script, and everything synchronized at GPU speed.

**This is what "Vultorch = your neural rendering IDE" looks like.**

---

## Key takeaways

| Concept | Code | Purpose |
|---------|------|---------|
| Positional encoding | `positional_encoding(x, L)` | High-frequency detail via Fourier features |
| File dialog | `tkinter.filedialog` in thread | Load any image without blocking |
| Architecture sliders | `slider_int("Hidden", ...)` | Live topology tuning |
| Error gain | `(err * gain).clamp_(0, 1)` | Amplify subtle reconstruction errors |
| Pause | `checkbox("Pause Training")` | Freeze training, UI stays live |
| Snapshot | `canvas.save("file.png")` | Dump GPU tensor to disk |
| Speed | `(it - it_last) / dt` | Iterations/sec counter |
| step()/end_step() | Training-loop-owned rendering | You control the outer loop |

!!! success "Congratulations!"
    You've completed all 12 Vultorch tutorials.  You now have every
    tool you need to build real-time, GPU-accelerated visualization
    into your neural rendering research workflow — and to publish
    polished interactive demos of your work.
