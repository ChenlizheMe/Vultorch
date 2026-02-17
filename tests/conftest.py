"""Shared pytest fixtures and helpers for Vultorch tests.

GPU tests require a Vulkan-capable GPU with CUDA.  They are skipped
automatically when either is missing.
"""

import pytest
import sys

# ---------------------------------------------------------------------------
#  Skip helpers
# ---------------------------------------------------------------------------

def _has_torch():
    try:
        import torch
        return True
    except ImportError:
        return False

def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def _has_vultorch():
    try:
        import vultorch
        return True
    except ImportError:
        return False

def _has_vultorch_cuda():
    try:
        import vultorch
        return vultorch.HAS_CUDA
    except (ImportError, AttributeError):
        return False


requires_torch = pytest.mark.skipif(not _has_torch(), reason="PyTorch not installed")
requires_cuda = pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
requires_vultorch = pytest.mark.skipif(not _has_vultorch(), reason="vultorch not importable")
requires_vultorch_cuda = pytest.mark.skipif(
    not _has_vultorch_cuda(), reason="vultorch built without CUDA"
)


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gpu_window():
    """Create and yield a single vultorch.Window for the whole test session."""
    import vultorch
    win = vultorch.Window("pytest", 256, 256)
    yield win
    win.destroy()


@pytest.fixture
def cpu_tensor_2d():
    """A simple 64x64 float32 CPU tensor (grayscale)."""
    import torch
    return torch.rand(64, 64, dtype=torch.float32)


@pytest.fixture
def cpu_tensor_rgb():
    """A 64x64x3 float32 CPU tensor."""
    import torch
    return torch.rand(64, 64, 3, dtype=torch.float32)


@pytest.fixture
def cpu_tensor_rgba():
    """A 64x64x4 float32 CPU tensor."""
    import torch
    return torch.rand(64, 64, 4, dtype=torch.float32)


@pytest.fixture
def cuda_tensor_rgba():
    """A 64x64x4 float32 CUDA tensor (skipped if no CUDA)."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.rand(64, 64, 4, dtype=torch.float32, device="cuda")
