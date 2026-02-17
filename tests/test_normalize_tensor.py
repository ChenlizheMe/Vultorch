"""Tests for _normalize_tensor helper.

Coverage targets:
  - dtype handling: float32, float16, uint8, invalid
  - shape handling: (H,W), (H,W,1), (H,W,3), (H,W,4), invalid channels, 4D
  - contiguity enforcement
"""

import pytest
from conftest import requires_torch, requires_vultorch


@requires_vultorch
@requires_torch
class TestNormalizeTensor:

    def _normalize(self, tensor):
        from vultorch import _normalize_tensor
        return _normalize_tensor(tensor)

    # ── dtype ──────────────────────────────────────────────────────

    def test_float32_passthrough(self):
        import torch
        t = torch.rand(32, 32, 4)
        out, h, w, c = self._normalize(t)
        assert out.dtype == torch.float32
        assert (h, w, c) == (32, 32, 4)

    def test_float16_conversion(self):
        import torch
        t = torch.rand(16, 16, 3, dtype=torch.float16)
        out, h, w, c = self._normalize(t)
        assert out.dtype == torch.float32
        assert (h, w, c) == (16, 16, 3)

    def test_uint8_conversion(self):
        import torch
        t = (torch.rand(8, 8, 4) * 255).to(torch.uint8)
        out, h, w, c = self._normalize(t)
        assert out.dtype == torch.float32
        assert out.max() <= 1.0
        assert out.min() >= 0.0

    def test_invalid_dtype_raises(self):
        import torch
        t = torch.rand(8, 8).to(torch.float64)
        with pytest.raises(ValueError, match="dtype"):
            self._normalize(t)

    def test_int32_dtype_raises(self):
        import torch
        t = torch.ones(8, 8, dtype=torch.int32)
        with pytest.raises(ValueError, match="dtype"):
            self._normalize(t)

    # ── shape ──────────────────────────────────────────────────────

    def test_2d_grayscale(self):
        import torch
        t = torch.rand(64, 48)
        out, h, w, c = self._normalize(t)
        assert (h, w, c) == (64, 48, 1)

    def test_3d_single_channel(self):
        import torch
        t = torch.rand(32, 24, 1)
        out, h, w, c = self._normalize(t)
        assert (h, w, c) == (32, 24, 1)

    def test_3d_rgb(self):
        import torch
        t = torch.rand(32, 24, 3)
        out, h, w, c = self._normalize(t)
        assert (h, w, c) == (32, 24, 3)

    def test_3d_rgba(self):
        import torch
        t = torch.rand(32, 24, 4)
        out, h, w, c = self._normalize(t)
        assert (h, w, c) == (32, 24, 4)

    def test_invalid_channels_raises(self):
        import torch
        t = torch.rand(32, 24, 2)
        with pytest.raises(ValueError, match="channels"):
            self._normalize(t)

    def test_invalid_channels_5_raises(self):
        import torch
        t = torch.rand(32, 24, 5)
        with pytest.raises(ValueError, match="channels"):
            self._normalize(t)

    def test_4d_raises(self):
        import torch
        t = torch.rand(1, 32, 24, 3)
        with pytest.raises(ValueError, match="2D or 3D"):
            self._normalize(t)

    def test_1d_raises(self):
        import torch
        t = torch.rand(100)
        with pytest.raises(ValueError, match="2D or 3D"):
            self._normalize(t)

    # ── contiguity ─────────────────────────────────────────────────

    def test_non_contiguous_made_contiguous(self):
        import torch
        t = torch.rand(64, 64, 4)[:, ::2, :]  # non-contiguous
        assert not t.is_contiguous()
        out, h, w, c = self._normalize(t)
        assert out.is_contiguous()

    def test_transposed_tensor(self):
        import torch
        t = torch.rand(4, 64, 64).permute(1, 2, 0)  # (64, 64, 4) non-contiguous
        assert not t.is_contiguous()
        out, h, w, c = self._normalize(t)
        assert out.is_contiguous()
        assert (h, w, c) == (64, 64, 4)
