"""Tests for C++ Engine binding coverage (bind_engine.cpp).

Coverage targets:
  - Engine.init / destroy / poll / begin_frame / end_frame
  - Engine.upload_tensor_cpu
  - Engine.tensor_texture_id / tensor_width / tensor_height
  - Engine.set_tensor_filter
  - Engine.allocate_shared_tensor / upload_tensor / sync_tensor (CUDA)
  - Engine.is_shared_ptr (CUDA)
  - SceneRenderer bindings via Engine

All tests share the session-scoped gpu_window (and its Engine) to avoid
creating/destroying multiple ImGui contexts which triggers assertions.
"""

import pytest
from conftest import (
    requires_vultorch, requires_torch, requires_cuda,
    requires_vultorch_cuda,
)


@requires_vultorch
class TestEngineClassExists:

    def test_engine_importable(self):
        from vultorch import _vultorch
        assert hasattr(_vultorch, "Engine")

    def test_engine_methods_exist(self):
        from vultorch import _vultorch
        eng_class = _vultorch.Engine
        for method in [
            "init", "destroy", "poll", "begin_frame", "end_frame",
            "upload_tensor_cpu", "tensor_texture_id",
            "tensor_width", "tensor_height", "set_tensor_filter",
        ]:
            assert hasattr(eng_class, method), f"Engine.{method} missing"

    def test_engine_scene_methods_exist(self):
        from vultorch import _vultorch
        eng_class = _vultorch.Engine
        for method in [
            "init_scene", "scene_render", "scene_texture_id",
            "scene_resize", "scene_set_msaa", "scene_process_input",
            "scene_set_camera", "scene_get_camera",
            "scene_set_light", "scene_set_background",
            "scene_width", "scene_height", "max_msaa",
        ]:
            assert hasattr(eng_class, method), f"Engine.{method} missing"

    def test_engine_cuda_methods_conditional(self):
        """CUDA methods should exist only when HAS_CUDA is True."""
        import vultorch
        from vultorch import _vultorch
        eng_class = _vultorch.Engine
        cuda_methods = [
            "allocate_shared_tensor", "upload_tensor",
            "sync_tensor", "is_shared_ptr",
        ]
        if vultorch.HAS_CUDA:
            for m in cuda_methods:
                assert hasattr(eng_class, m), f"Engine.{m} missing (HAS_CUDA=True)"


@requires_vultorch
@requires_cuda
@pytest.mark.gpu
class TestEngineLifecycle:
    """Use the session gpu_window._engine instead of creating new Engines."""

    def test_engine_poll(self, gpu_window):
        eng = gpu_window._engine
        result = eng.poll()
        assert isinstance(result, bool)

    def test_engine_frame(self, gpu_window):
        eng = gpu_window._engine
        if eng.poll():
            if eng.begin_frame():
                eng.end_frame()

    def test_upload_tensor_cpu(self, gpu_window):
        import torch
        eng = gpu_window._engine
        t = torch.rand(32, 32, 4, dtype=torch.float32)
        ptr = t.data_ptr()
        eng.upload_tensor_cpu("eng_cpu_test", ptr, 32, 32, 4)

    def test_texture_id_and_dimensions(self, gpu_window):
        import torch
        eng = gpu_window._engine
        t = torch.rand(48, 64, 4, dtype=torch.float32)
        eng.upload_tensor_cpu("eng_dim", t.data_ptr(), 64, 48, 4)
        tid = eng.tensor_texture_id("eng_dim")
        assert isinstance(tid, int)
        w = eng.tensor_width("eng_dim")
        h = eng.tensor_height("eng_dim")
        assert w == 64
        assert h == 48

    def test_set_tensor_filter(self, gpu_window):
        import torch
        eng = gpu_window._engine
        t = torch.rand(32, 32, 4, dtype=torch.float32)
        eng.upload_tensor_cpu("eng_flt", t.data_ptr(), 32, 32, 4)
        eng.set_tensor_filter("eng_flt", 1)  # 1 = linear
        eng.set_tensor_filter("eng_flt", 0)  # 0 = nearest


@requires_vultorch
@requires_vultorch_cuda
@requires_cuda
@pytest.mark.gpu
class TestEngineCUDA:

    def test_allocate_shared_tensor(self, gpu_window):
        eng = gpu_window._engine
        ptr = eng.allocate_shared_tensor("eng_shared", 32, 32)
        assert isinstance(ptr, int)
        assert ptr != 0
        assert eng.is_shared_ptr("eng_shared", ptr)

    def test_upload_tensor_cuda(self, gpu_window):
        import torch
        eng = gpu_window._engine
        t = torch.rand(32, 32, 4, device="cuda")
        eng.upload_tensor("eng_upload_t", t.data_ptr(), 32, 32, 4)

    def test_sync_tensor(self, gpu_window):
        eng = gpu_window._engine
        eng.allocate_shared_tensor("eng_sync", 32, 32)
        eng.sync_tensor("eng_sync")


@requires_vultorch
@requires_vultorch_cuda
@requires_cuda
@pytest.mark.gpu
class TestEngineScene:
    """Scene renderer uses the session engine — init_scene only once to avoid
    repeated Vulkan resource creation that leads to VK_ERROR_DEVICE_LOST."""

    _scene_inited = False

    @staticmethod
    def _ensure_scene(eng):
        if not TestEngineScene._scene_inited:
            eng.init_scene(128, 128, 2)
            TestEngineScene._scene_inited = True

    def test_init_scene(self, gpu_window):
        eng = gpu_window._engine
        TestEngineScene._ensure_scene(eng)
        tid = eng.scene_texture_id()
        assert isinstance(tid, int)

    def test_scene_dimensions(self, gpu_window):
        eng = gpu_window._engine
        TestEngineScene._ensure_scene(eng)
        eng.scene_resize(200, 150)
        assert eng.scene_width() == 200
        assert eng.scene_height() == 150

    def test_scene_resize(self, gpu_window):
        eng = gpu_window._engine
        TestEngineScene._ensure_scene(eng)
        eng.scene_resize(256, 256)
        assert eng.scene_width() == 256

    def test_scene_msaa(self, gpu_window):
        eng = gpu_window._engine
        TestEngineScene._ensure_scene(eng)
        eng.scene_set_msaa(4)
        mm = eng.max_msaa()
        assert isinstance(mm, int)
        assert mm >= 1

    def test_scene_camera(self, gpu_window):
        eng = gpu_window._engine
        TestEngineScene._ensure_scene(eng)
        eng.scene_set_camera(0.0, 0.5, 3.0, 0.0, 0.0, 0.0, 45.0)
        cam = eng.scene_get_camera()
        assert len(cam) >= 3

    def test_scene_light(self, gpu_window):
        eng = gpu_window._engine
        TestEngineScene._ensure_scene(eng)
        eng.scene_set_light(0.3, -1.0, 0.5, 1.5)

    def test_scene_background(self, gpu_window):
        eng = gpu_window._engine
        TestEngineScene._ensure_scene(eng)
        eng.scene_set_background(0.2, 0.2, 0.2)

    def test_scene_render(self, gpu_window):
        import torch
        eng = gpu_window._engine
        TestEngineScene._ensure_scene(eng)
        t = torch.rand(32, 32, 4, device="cuda")
        eng.upload_tensor("eng_scene_data", t.data_ptr(), 32, 32, 4)
        if eng.poll() and eng.begin_frame():
            eng.scene_render("eng_scene_data")
            eng.end_frame()
