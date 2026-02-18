# Vultorch Tutorial

A step-by-step guide through Vultorch, one example at a time.  
Each chapter maps to a runnable script in the `examples/` folder.

| Chapter | Topic | Key concepts |
|---------|-------|-------------|
| [01 — Hello Tensor](01_hello_tensor.md) | Minimal display | View, Panel, Canvas, bind, run |
| [02 — Multi-Panel](02_multi_panel.md) | Multiple panels & canvases | Layout, side, multi-canvas |
| [03 — Training Test](03_training_test.md) | Fit a tiny network to a GT image | custom dock layout, create_tensor, per-pixel optimization |
| [04 — Conway's Game of Life](04_conway.md) | GPU cellular automaton | create_tensor for simulation, filter="nearest", sidebar, buttons, color pickers |
| [05 — Image Viewer](05_image_viewer.md) | Load, transform & save images | imread, imwrite, Canvas.save, combo, input_text, filter toggle |
| [06 — Pixel Canvas](06_pixel_canvas.md) | Interactive drawing on a GPU tensor | mouse interaction, screen→pixel mapping, backing store pattern |
| [07 — Multi-Channel Viewer](07_multichannel.md) | RGB + depth + normal + alpha in one window | multiple zero-copy tensors, turbo colormap, ray-sphere intersection |
| [08 — GT vs Prediction](08_gt_vs_pred.md) | Live training comparison with error heatmap | error heatmap, PSNR, loss curves, error mode switching |
| [09 — Live Hyperparameter Tuning](09_live_tuning.md) | Change LR, optimizer, loss at runtime | step()/end_step(), log-scale LR, optimizer hot-swap |
| [10 — 2D Gaussian Splatting](10_gaussian2d.md) | Differentiable 2D Gaussian rendering | nn.Parameter, alpha compositing, cumprod transmittance |
| [11 — 3D Surface Inspector](11_3d_inspector.md) | Orbit camera with Blinn-Phong lighting | SceneView, Camera, Light, MSAA, procedural textures |
| [12 — Neural Rendering Workstation](12_neural_workstation.md) | Capstone: 6-panel workstation with dual-head MLP | Two-head MLP, six panels, pause/resume, snapshot, optimizer hot-swap |
