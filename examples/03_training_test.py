"""
03 - Training Test
==================
Fit a tiny MLP to an image in real time.

Layout:
- Left panel : GT image
- Right panel: prediction
- Bottom panel: runtime stats (FPS, loss, iteration)

Key concepts
------------
- on_frame           : Per-frame callback for training logic
- Panel.on_frame     : Per-panel callback for widgets (runs inside the panel)
- create_tensor      : Zero-copy shared GPU memory
- vultorch.imread    : Load image without PIL dependency
- side="bottom"      : Dock a panel to the bottom
- side="left"        : Dock a panel to the left
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import vultorch


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


if not torch.cuda.is_available():
    raise RuntimeError("This example needs CUDA")

device = "cuda"

img_path = Path(__file__).resolve().parents[1] / "docs" / "images" / "pytorch_logo.png"
gt = vultorch.imread(img_path, channels=3, size=(256, 256), device=device)

H, W = gt.shape[0], gt.shape[1]
ys = torch.linspace(-1.0, 1.0, H, device=device)
xs = torch.linspace(-1.0, 1.0, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")
coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
target = gt.reshape(-1, 3)

model = TinyMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

# -- View + panels (high-level declarative API) -------------------------
view = vultorch.View("03 - Training Test", 1280, 760)
info_panel = view.panel("Info", side="bottom", width=0.28)
gt_panel = view.panel("GT", side="left", width=0.5)
pred_panel = view.panel("Prediction")

gt_panel.canvas("gt").bind(gt)

pred_rgba = vultorch.create_tensor(H, W, channels=4, device=device,
                                   name="pred", window=view.window)
pred_rgba[:, :, 3] = 1.0
pred_panel.canvas("pred").bind(pred_rgba)

state = {
    "iter": 0,
    "loss": 1.0,
    "ema": 1.0,
    "steps_per_frame": 6,
}


@view.on_frame
def train():
    for _ in range(state["steps_per_frame"]):
        optimizer.zero_grad(set_to_none=True)
        out = model(coords)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()

        state["iter"] += 1
        state["loss"] = loss.item()
        state["ema"] = state["ema"] * 0.98 + state["loss"] * 0.02

    with torch.no_grad():
        pred = model(coords).reshape(H, W, 3).clamp_(0, 1)
        pred_rgba[:, :, :3] = pred


@info_panel.on_frame
def draw_info():
    info_panel.text(f"FPS: {view.fps:.1f}")
    info_panel.text(f"Iteration: {state['iter']}")
    info_panel.text(f"Loss (MSE): {state['loss']:.6f}")
    info_panel.text(f"EMA Loss: {state['ema']:.6f}")

    state["steps_per_frame"] = info_panel.slider_int(
        "Steps / Frame", 1, 32, default=6
    )
    progress = min(1.0, state["iter"] / 3000.0)
    info_panel.progress(progress,
                        overlay=f"Training progress {progress * 100:.1f}%")
    info_panel.text_wrapped(
        "Left is GT, right is prediction. "
        "Increase 'Steps / Frame' for faster fitting."
    )


view.run()
