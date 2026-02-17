import pytest
import torch

import vultorch


def _require_gpu():
    assert torch.cuda.is_available(), "CUDA is required for these tests"
    assert vultorch.HAS_CUDA, "Vultorch must be built with CUDA"


@pytest.mark.gpu
def test_window_a_single(gpu_window):
    _require_gpu()

    if gpu_window.poll() and gpu_window.begin_frame():
        t = torch.rand(32, 32, 4, device="cuda")
        vultorch.show(t, name="a", window=gpu_window)
        gpu_window.end_frame()
