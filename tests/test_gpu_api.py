import pytest
import torch

import vultorch


def _require_gpu():
    assert torch.cuda.is_available(), "CUDA is required for these tests"
    assert vultorch.HAS_CUDA, "Vultorch must be built with CUDA"


@pytest.mark.gpu
def test_import_and_cuda():
    _require_gpu()
    assert isinstance(vultorch.__version__, str)


@pytest.mark.gpu
def test_show_uint8_float16_multi_tensor():
    _require_gpu()

    win = vultorch.Window("pytest-multi-tensor", 256, 256)
    try:
        if win.poll() and win.begin_frame():
            h, w = 64, 64
            t_u8 = (torch.rand(h, w, 3, device="cuda") * 255).to(torch.uint8)
            t_f16 = torch.rand(h, w, 3, device="cuda", dtype=torch.float16)

            vultorch.show(t_u8, name="u8", window=win)
            vultorch.show(t_f16, name="f16", window=win)

            win.end_frame()
    finally:
        win.destroy()


@pytest.mark.gpu
@pytest.mark.slow
def test_scene_view_basic():
    _require_gpu()

    win = vultorch.Window("pytest-scene", 512, 512)
    scene = vultorch.SceneView("Scene", 256, 256, msaa=2)
    try:
        if win.poll() and win.begin_frame():
            t = torch.rand(64, 64, 4, device="cuda")
            scene.set_tensor(t)
            scene.render()
            win.end_frame()
    finally:
        win.destroy()
