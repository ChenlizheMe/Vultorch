"""
03 - Tiny Net Live Training
===========================
Fit a lightweight MLP to an image in real time.

Layout:
- Left panel : GT image (docs/images/pytorch_logo.png)
- Right panel: network prediction, updated every frame
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vultorch

try:
    from PIL import Image
except ImportError as exc:
    raise RuntimeError("Please install pillow: pip install pillow") from exc


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
img = Image.open(img_path).convert("RGB").resize((256, 256), Image.BILINEAR)

gt = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).to(device)

H, W = gt.shape[0], gt.shape[1]

ys = torch.linspace(-1.0, 1.0, H, device=device)
xs = torch.linspace(-1.0, 1.0, W, device=device)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")
coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
target = gt.reshape(-1, 3)

model = TinyMLP().to(device)
optim = torch.optim.Adam(model.parameters(), lr=2e-3)

view = vultorch.View("03 - Tiny Net Live Training", 1200, 700)
left = view.panel("GT")
right = view.panel("Training Output", side="right", width=0.5)

left.canvas("gt").bind(gt)

pred_canvas = right.canvas("pred")
pred_rgba = vultorch.create_tensor(H, W, channels=4, device=device,
                                   name="pred", window=view.window)
pred_rgba[:, :, 3] = 1.0
pred_canvas.bind(pred_rgba)

steps_per_frame = 6
state = {"iter": 0, "ema": 1.0}


@view.on_frame
def train_and_render():
    for _ in range(steps_per_frame):
        optim.zero_grad(set_to_none=True)
        out = model(coords)
        loss = F.mse_loss(out, target)
        loss.backward()
        optim.step()

        state["iter"] += 1
        state["ema"] = state["ema"] * 0.98 + loss.item() * 0.02

    with torch.no_grad():
        pred = model(coords).reshape(H, W, 3).clamp_(0, 1)
        pred_rgba[:, :, :3] = pred

    if state["iter"] % 60 == 0:
        print(f"[03] iter={state['iter']}  ema_mse={state['ema']:.6f}")


view.run()
