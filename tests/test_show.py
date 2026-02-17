"""Tests for show() function — error paths and RGBA expansion logic.

Coverage targets:
  - show() without active window → RuntimeError
  - show() with various tensor shapes (1ch, 3ch, 4ch)
  - show() with float16, uint8, float32
  - show() filter modes (nearest, linear)
  - show() RGBA buffer caching (same name, different name)
  - show() non-contiguous tensor
  - show() CPU fallback warning when no CUDA
"""

import pytest
from conftest import requires_vultorch, requires_torch


@requires_vultorch
@requires_torch
class TestShowErrors:

    def test_no_window_raises(self):
        import vultorch
        import torch
        saved = vultorch.Window._current
        try:
            vultorch.Window._current = None
            t = torch.rand(32, 32, 4)
            with pytest.raises(RuntimeError, match="No active"):
                vultorch.show(t)
        finally:
            vultorch.Window._current = saved

    def test_explicit_none_window_raises(self):
        import vultorch
        import torch
        saved = vultorch.Window._current
        try:
            vultorch.Window._current = None
            t = torch.rand(32, 32, 4)
            with pytest.raises(RuntimeError, match="No active"):
                vultorch.show(t, window=None)
        finally:
            vultorch.Window._current = saved


@requires_vultorch
@requires_torch
class TestCreateTensorErrors:

    def test_no_window_raises(self):
        import vultorch
        saved = vultorch.Window._current
        try:
            vultorch.Window._current = None
            with pytest.raises(RuntimeError, match="No active"):
                vultorch.create_tensor(64, 64)
        finally:
            vultorch.Window._current = saved
