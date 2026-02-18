# Vultorch 教程

逐步学习 Vultorch，每个章节对应 `examples/` 目录中的一个可运行脚本。

| 章节 | 主题 | 核心概念 |
|------|------|---------|
| [01 — Hello Tensor](01_hello_tensor.zh.md) | 最小示例 | View, Panel, Canvas, bind, run |
| [02 — 多面板](02_multi_panel.zh.md) | 多面板与多画布 | 布局, side, 多画布 |
| [03 — 训练测试](03_training_test.zh.md) | 拟合 GT 图像 | 自定义停靠布局, create_tensor, 逐像素优化 |
| [04 — 康威生命游戏](04_conway.zh.md) | GPU 元胞自动机 | create_tensor 用于模拟, filter="nearest", 侧边栏, 按钮, 颜色选择器 |
| [05 — 图片查看器](05_image_viewer.zh.md) | 加载、变换、保存图片 | imread, imwrite, Canvas.save, combo, input_text, 滤波切换 |
| [06 — 像素画板](06_pixel_canvas.zh.md) | 在 GPU tensor 上交互式绘画 | 鼠标交互, 屏幕→像素映射, 底图模式 |
| [07 — 多通道查看器](07_multichannel.zh.md) | RGB + depth + normal + alpha 同屏 | 多个零拷贝 tensor, turbo 色图, 光线-球体求交 |
| [08 — GT vs 预测](08_gt_vs_pred.zh.md) | 实时训练对比与误差热力图 | 误差热力图, PSNR, loss 曲线, 误差模式切换 |
| [09 — 实时超参数调优](09_live_tuning.zh.md) | 运行时修改 LR、优化器、损失函数 | step()/end_step(), 对数 LR, 优化器热切换 |
| [10 — 二维高斯泼溅](10_gaussian2d.zh.md) | 可微分二维高斯渲染 | nn.Parameter, alpha 合成, cumprod 透射率 |
| [11 — 3D 表面检查器](11_3d_inspector.zh.md) | 带 Blinn-Phong 光照的轨道相机 | SceneView, Camera, Light, MSAA, 程序化纹理 |
| [12 — 神经渲染工作站](12_neural_workstation.zh.md) | 压轴：双头 MLP 六面板工作站 | 双头 MLP, 六面板, 暂停/恢复, 快照, 优化器热切换 |
