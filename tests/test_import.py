"""Tests for vultorch package import, version, and module structure.

Coverage targets:
  - vultorch.__version__
  - vultorch.HAS_CUDA flag
  - vultorch.ui submodule existence
  - All public names exported
"""

import pytest
from conftest import requires_torch, requires_vultorch


# ═══════════════════════════════════════════════════════════════════════
#  Basic import
# ═══════════════════════════════════════════════════════════════════════

@requires_vultorch
class TestImport:

    def test_version_string(self):
        import vultorch
        assert isinstance(vultorch.__version__, str)
        parts = vultorch.__version__.split(".")
        assert len(parts) >= 2, "Version should be semver-like"

    def test_has_cuda_flag(self):
        import vultorch
        assert isinstance(vultorch.HAS_CUDA, bool)

    def test_ui_submodule(self):
        from vultorch import ui
        assert hasattr(ui, "text")
        assert hasattr(ui, "button")
        assert hasattr(ui, "begin")
        assert hasattr(ui, "end")

    def test_public_classes_exist(self):
        import vultorch
        assert hasattr(vultorch, "Window")
        assert hasattr(vultorch, "View")
        assert hasattr(vultorch, "Panel")
        assert hasattr(vultorch, "Canvas")
        assert hasattr(vultorch, "Camera")
        assert hasattr(vultorch, "Light")
        assert hasattr(vultorch, "SceneView")

    def test_public_functions_exist(self):
        import vultorch
        assert callable(vultorch.show)
        assert callable(vultorch.create_tensor)

    def test_engine_class_exists(self):
        from vultorch._vultorch import Engine
        assert Engine is not None
