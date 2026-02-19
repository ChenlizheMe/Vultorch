"""
12 - Neural Rendering Workstation
=================================
The capstone demo — everything Vultorch can do in one script.

Train a coordinate MLP to reconstruct any image, with a full IDE-like
interface: live prediction, error heatmap, loss curves, and a rich
control sidebar that lets you change *everything* at runtime.

Use this as a template for building polished, publishable demos of
your own neural rendering research.

Features
--------
- Open any image          : OS file dialog (tkinter) — load PNG/JPEG/BMP
- Positional encoding     : Fourier features with adjustable frequency
- Network architecture    : Width / depth sliders (rebuild on demand)
- Error heatmap           : Turbo colormap with adjustable gain
- Loss function combo     : MSE / L1 / Huber
- Optimizer hot-swap      : Adam / SGD / AdamW
- Pause / resume          : Freeze training, UI stays responsive
- Save all snapshots      : One button exports prediction + error to PNG
- Training speed          : Iterations/sec counter
- Reset model             : Reinitialize weights, keep everything else

Layout (1400×700)
-----------------
Left sidebar (24%) : All controls
Top-left           : Predicted RGB (live)
Top-right          : Error heatmap (turbo)
Bottom             : Metrics (loss / PSNR curves, stats)
"""

from pathlib import Path
import math, time

import torch
import torch.nn as nn
import torch.nn.functional as F
import vultorch

if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"
RES = 256  # training resolution (square)

# ── Positional encoding ──────────────────────────────────────────
def positional_encoding(x, L):
    """Fourier features: [x, sin(2^0 πx), cos(2^0 πx), ..., sin(2^L πx), cos(2^L πx)]."""
    if L == 0:
        return x
    freqs = 2.0 ** torch.arange(L, dtype=torch.float32, device=x.device)
    xf = (x.unsqueeze(-1) * freqs * math.pi).reshape(*x.shape[:-1], -1)
    return torch.cat([x, xf.sin(), xf.cos()], dim=-1)

# ── Model ─────────────────────────────────────────────────────────
class CoordMLP(nn.Module):
    def __init__(self, hidden=128, layers=4, pe_levels=6, in_dim=2):
        super().__init__()
        self.pe = pe_levels
        d_in = in_dim + in_dim * pe_levels * 2  # raw + sin/cos per level
        net = [nn.Linear(d_in, hidden), nn.ReLU(True)]
        for _ in range(layers - 2):
            net += [nn.Linear(hidden, hidden), nn.ReLU(True)]
        net += [nn.Linear(hidden, 3), nn.Sigmoid()]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(positional_encoding(x, self.pe))

def make_model(hidden, layers, pe):
    return CoordMLP(hidden, layers, pe).to(device)

# ── Optimizer / loss helpers ──────────────────────────────────────
OPT_NAMES  = ["Adam", "SGD", "AdamW"]
LOSS_NAMES = ["MSE", "L1", "Huber"]

def make_optimizer(m, name, lr, wd):
    if name == "SGD":
        return torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9,
                               weight_decay=wd)
    if name == "AdamW":
        return torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    return torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)

def compute_loss(pred, target, name):
    if name == "L1":    return F.l1_loss(pred, target)
    if name == "Huber": return F.smooth_l1_loss(pred, target)
    return F.mse_loss(pred, target)

def psnr(mse):
    return 50.0 if mse <= 0 else -10 * math.log10(mse)

# ── Load image helper ─────────────────────────────────────────────
def load_image(path, res):
    """Load an image and build coordinate grid."""
    img = vultorch.imread(path, channels=3, size=(res, res), device=device)
    H, W = img.shape[:2]
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    target = img.reshape(-1, 3)
    return img, coords, target, H, W

img_path = (Path(__file__).resolve().parents[1]
            / "docs" / "images" / "pytorch_logo.png")
gt, coords, target, H, W = load_image(str(img_path), RES)

# ── State ─────────────────────────────────────────────────────────
S = dict(
    lr=1e-2, opt_idx=0, loss_idx=0, wd=0.01,
    spf=8, paused=False, needs_reset=False,
    hidden=128, layers=4, pe=6, arch_dirty=False,
    err_gain=5.0,
    it=0, loss=1.0, psnr=0.0,
    loss_h=[], psnr_h=[],
    prev_opt=0, its_sec=0.0,
    pending_image=None,
    img_name=Path(img_path).name,
)

model = make_model(S["hidden"], S["layers"], S["pe"])
optimizer = make_optimizer(model, "Adam", S["lr"], S["wd"])
t_last = time.perf_counter()
it_last = 0

# ── View + panels ────────────────────────────────────────────────
view = vultorch.View("12 — Neural Rendering Workstation", 900, 900)
ctrl    = view.panel("Controls",  side="left", width=0.5)
rgb_pan = view.panel("Prediction")
err_pan = view.panel("Error Map")
met_pan = view.panel("Metrics")

# Display tensors
rgb_t = vultorch.create_tensor(H, W, 4, device, name="rgb", window=view.window)
rgb_t[:, :, 3] = 1.0
rgb_cv = rgb_pan.canvas("rgb")
rgb_cv.bind(rgb_t)

err_t = vultorch.create_tensor(H, W, 4, device, name="err", window=view.window)
err_t[:, :, 3] = 1.0
err_cv = err_pan.canvas("error")
err_cv.bind(err_t)

# ── Control sidebar ──────────────────────────────────────────────
@ctrl.on_frame
def draw_ctrl():
    global optimizer, model, gt, coords, target, H, W, t_last, it_last

    ctrl.text(f"FPS {view.fps:.0f}  |  Iter {S['it']}  |  {S['its_sec']:.0f} it/s")
    ctrl.separator()

    # ── Image loading ──
    ctrl.text(f"Image: {S['img_name']}")
    path = ctrl.file_dialog("Open Image...",
                            title="Select an image")
    if path:
        S["pending_image"] = path

    ctrl.separator()

    # ── Learning rate ──
    log_lr = ctrl.slider("log10(LR)", -5.0, -1.0, default=-2.0)
    S["lr"] = 10 ** log_lr
    ctrl.text(f"  LR = {S['lr']:.2e}")
    for pg in optimizer.param_groups:
        pg["lr"] = S["lr"]

    # ── Optimizer ──
    S["opt_idx"] = ctrl.combo("Optimizer", OPT_NAMES, default=0)
    log_wd = ctrl.slider("log10(WD)", -6.0, -2.0, default=-5.0)
    S["wd"] = 10 ** log_wd
    ctrl.text(f"  WD = {S['wd']:.1e}")
    for pg in optimizer.param_groups:
        pg["weight_decay"] = S["wd"]

    # ── Loss ──
    S["loss_idx"] = ctrl.combo("Loss", LOSS_NAMES, default=0)

    ctrl.separator()

    # ── Network architecture ──
    ctrl.text("Network Architecture")
    new_h = ctrl.slider_int("Hidden", 32, 256, default=128)
    new_l = ctrl.slider_int("Layers", 2, 8, default=4)
    new_pe = ctrl.slider_int("PE Levels", 0, 10, default=6)
    if new_h != S["hidden"] or new_l != S["layers"] or new_pe != S["pe"]:
        S["hidden"] = new_h; S["layers"] = new_l; S["pe"] = new_pe
        S["arch_dirty"] = True
    if S["arch_dirty"]:
        ctrl.text_colored(1, 0.8, 0, 1, "  Architecture changed!")
        if ctrl.button("Apply & Reset", width=170):
            model = make_model(S["hidden"], S["layers"], S["pe"])
            optimizer = make_optimizer(model, OPT_NAMES[S["opt_idx"]],
                                       S["lr"], S["wd"])
            S.update(it=0, loss=1.0, psnr=0.0, prev_opt=S["opt_idx"],
                     arch_dirty=False)
            S["loss_h"].clear(); S["psnr_h"].clear()

    ctrl.separator()

    # ── Training controls ──
    S["spf"] = ctrl.slider_int("Steps/Frame", 1, 64, default=8)
    S["paused"] = ctrl.checkbox("Pause Training", default=False)

    ctrl.separator()

    # ── Visualization ──
    ctrl.text("Visualization")
    S["err_gain"] = ctrl.slider("Error Gain", 1.0, 20.0, default=5.0)

    ctrl.separator()

    # ── Actions ──
    if ctrl.button("Save Snapshots", width=170):
        rgb_cv.save("snapshot_pred.png")
        err_cv.save("snapshot_error.png")

    if ctrl.button("Reset Model", width=170):
        S["needs_reset"] = True

    ctrl.separator()
    pct = min(1.0, S["it"] / 10000)
    ctrl.progress(pct, overlay=f"{pct*100:.0f}%")
    ctrl.text(f"MSE  {S['loss']:.6f}")
    ctrl.text(f"PSNR {S['psnr']:.1f} dB")

    n_params = sum(p.numel() for p in model.parameters())
    ctrl.text(f"Params: {n_params:,}")

    ctrl.separator()
    ctrl.text_wrapped(
        "Neural Rendering Workstation — open any image from your OS, "
        "tune the network architecture, optimizer, loss, and LR in "
        "real time. Use this as a template for publishing polished "
        "interactive demos of your research."
    )

# ── Metrics panel ─────────────────────────────────────────────────
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

# ── Training loop ────────────────────────────────────────────────
try:
    while view.step():
        # ── Handle pending image load ──
        if S["pending_image"]:
            try:
                gt, coords, target, H, W = load_image(
                    S["pending_image"], RES)
                S["img_name"] = Path(S["pending_image"]).name
                rgb_t = vultorch.create_tensor(H, W, 4, device,
                    name="rgb", window=view.window)
                rgb_t[:, :, 3] = 1.0
                rgb_cv.bind(rgb_t)
                err_t = vultorch.create_tensor(H, W, 4, device,
                    name="err", window=view.window)
                err_t[:, :, 3] = 1.0
                err_cv.bind(err_t)
                S["needs_reset"] = True
            except Exception:
                pass
            S["pending_image"] = None

        # ── Reset ──
        if S["needs_reset"]:
            model = make_model(S["hidden"], S["layers"], S["pe"])
            optimizer = make_optimizer(model, OPT_NAMES[S["opt_idx"]],
                                       S["lr"], S["wd"])
            S.update(it=0, loss=1.0, psnr=0.0, prev_opt=S["opt_idx"],
                     needs_reset=False, arch_dirty=False)
            S["loss_h"].clear(); S["psnr_h"].clear()
            t_last = time.perf_counter(); it_last = 0

        # ── Optimizer hot-swap ──
        if S["opt_idx"] != S["prev_opt"]:
            optimizer = make_optimizer(model, OPT_NAMES[S["opt_idx"]],
                                       S["lr"], S["wd"])
            S["prev_opt"] = S["opt_idx"]

        # ── Train ──
        if not S["paused"]:
            for _ in range(S["spf"]):
                optimizer.zero_grad(set_to_none=True)
                pred = model(coords)
                loss = compute_loss(pred, target,
                                    LOSS_NAMES[S["loss_idx"]])
                loss.backward()
                optimizer.step()
                S["it"] += 1
                S["loss"] = loss.item()

        # ── Speed counter ──
        now = time.perf_counter()
        dt = now - t_last
        if dt >= 0.5:
            S["its_sec"] = (S["it"] - it_last) / dt
            t_last = now; it_last = S["it"]

        # ── Display ──
        with torch.no_grad():
            pr = model(coords).reshape(H, W, 3).clamp_(0, 1)
            rgb_t[:, :, :3] = pr

            err = (gt - pr).abs().mean(dim=-1)
            err_t[:, :, :3] = vultorch.colormap(
                err, cmap="turbo", vmin=0.0, vmax=1.0 / S["err_gain"])

            mse = F.mse_loss(pr.reshape(-1, 3), target).item()
            S["psnr"] = psnr(mse)

        S["loss_h"].append(S["loss"])
        S["psnr_h"].append(S["psnr"])
        if len(S["loss_h"]) > 500:
            S["loss_h"] = S["loss_h"][-500:]
            S["psnr_h"] = S["psnr_h"][-500:]

        view.end_step()
finally:
    view.close()
