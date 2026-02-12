import pytest
import torch

import vultorch


def _require_gpu():
    assert torch.cuda.is_available(), "CUDA is required for these tests"
    assert vultorch.HAS_CUDA, "Vultorch must be built with CUDA"


@pytest.mark.gpu
def test_window_a_single():
    _require_gpu()

    win = vultorch.Window("pytest-win-a", 256, 256)
    try:
        if win.poll() and win.begin_frame():
            t = torch.rand(32, 32, 4, device="cuda")
            vultorch.show(t, name="a", window=win)
            win.end_frame()
    finally:
        win.destroy()
