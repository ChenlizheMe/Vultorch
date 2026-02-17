"""GPU integration tests — require CUDA + Vulkan runtime.

Coverage targets:
  - Window: init, poll, begin_frame, end_frame, destroy, activate
  - Window: upload_tensor (CUDA & CPU), tensor_texture_id, tensor_size
  - Window: get_texture_id, get_texture_size
  - Window: __del__ safety
  - show(): CUDA tensor (4ch, 3ch, 1ch), CPU tensor, uint8, float16
  - show(): filter modes, custom size, named tensors
  - create_tensor(): 4ch (zero-copy), 3ch fallback, 1ch fallback, CPU mode
  - SceneView: init, set_tensor, render, camera/light/background, msaa
  - View + Panel + Canvas: full frame loop (step/end_step)
"""

import pytest
from conftest import (
    requires_vultorch, requires_torch, requires_cuda,
    requires_vultorch_cuda,
)


# ═══════════════════════════════════════════════════════════════════════
#  Window lifecycle
# ═══════════════════════════════════════════════════════════════════════

@requires_vultorch
@requires_cuda
@pytest.mark.gpu
class TestWindowLifecycle:
    """All lifecycle operations happen on the session-scoped gpu_window
    to avoid creating/destroying multiple Windows (ImGui limitation)."""

    def test_window_current_ref(self, gpu_window):
        import vultorch
        assert vultorch.Window._current is gpu_window

    def test_activate_idempotent(self, gpu_window):
        import vultorch
        gpu_window.activate()
        assert vultorch.Window._current is gpu_window

    def test_poll_returns_bool(self, gpu_window):
        result = gpu_window.poll()
        assert isinstance(result, bool)

    def test_begin_end_frame(self, gpu_window):
        if gpu_window.poll():
            result = gpu_window.begin_frame()
            if result:
                gpu_window.end_frame()


# ═══════════════════════════════════════════════════════════════════════
#  show() with GPU tensors
# ═══════════════════════════════════════════════════════════════════════

@requires_vultorch
@requires_vultorch_cuda
@requires_cuda
@pytest.mark.gpu
class TestShowGPU:

    def test_show_rgba_cuda(self, gpu_window):
        import torch, vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            t = torch.rand(64, 64, 4, device="cuda")
            vultorch.show(t, window=gpu_window)
            gpu_window.end_frame()

    def test_show_rgb_cuda(self, gpu_window):
        import torch, vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            t = torch.rand(64, 64, 3, device="cuda")
            vultorch.show(t, name="rgb", window=gpu_window)
            gpu_window.end_frame()

    def test_show_grayscale_cuda(self, gpu_window):
        import torch, vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            t = torch.rand(64, 64, device="cuda")
            vultorch.show(t, name="gray", window=gpu_window)
            gpu_window.end_frame()

    def test_show_single_channel_3d(self, gpu_window):
        import torch, vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            t = torch.rand(64, 64, 1, device="cuda")
            vultorch.show(t, name="1ch", window=gpu_window)
            gpu_window.end_frame()

    def test_show_uint8_cuda(self, gpu_window):
        import torch, vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            t = (torch.rand(64, 64, 3, device="cuda") * 255).to(torch.uint8)
            vultorch.show(t, name="u8", window=gpu_window)
            gpu_window.end_frame()

    def test_show_float16_cuda(self, gpu_window):
        import torch, vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            t = torch.rand(64, 64, 3, device="cuda", dtype=torch.float16)
            vultorch.show(t, name="f16", window=gpu_window)
            gpu_window.end_frame()

    def test_show_filter_nearest(self, gpu_window):
        import torch, vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            t = torch.rand(32, 32, 4, device="cuda")
            vultorch.show(t, name="nearest", filter="nearest", window=gpu_window)
            gpu_window.end_frame()

    def test_show_filter_linear(self, gpu_window):
        import torch, vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            t = torch.rand(32, 32, 4, device="cuda")
            vultorch.show(t, name="linear", filter="linear", window=gpu_window)
            gpu_window.end_frame()

    def test_show_custom_size(self, gpu_window):
        import torch, vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            t = torch.rand(32, 32, 4, device="cuda")
            vultorch.show(t, name="sized", width=128, height=128, window=gpu_window)
            gpu_window.end_frame()

    def test_show_multiple_named_tensors(self, gpu_window):
        import torch, vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            t1 = torch.rand(32, 32, 4, device="cuda")
            t2 = torch.rand(64, 64, 4, device="cuda")
            vultorch.show(t1, name="tex_a", window=gpu_window)
            vultorch.show(t2, name="tex_b", window=gpu_window)
            gpu_window.end_frame()

    def test_show_non_contiguous_cuda(self, gpu_window):
        """Non-contiguous GPU tensors should still work."""
        import torch, vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            t = torch.rand(64, 64, 4, device="cuda")[:, ::2, :]
            vultorch.show(t, name="noncontig", window=gpu_window)
            gpu_window.end_frame()

    def test_rgba_cache_reuse(self, gpu_window):
        """Calling show() twice with the same name should reuse the RGBA cache."""
        import torch, vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            t = torch.rand(32, 32, 3, device="cuda")
            vultorch.show(t, name="cached", window=gpu_window)
            # Second call should reuse cached buffer
            vultorch.show(t, name="cached", window=gpu_window)
            assert "cached" in gpu_window._rgba_bufs
            gpu_window.end_frame()

    def test_rgba_cache_resize(self, gpu_window):
        """Changing tensor size should reallocate the RGBA cache."""
        import torch, vultorch
        # Frame 1: show a 32×32 tensor
        if gpu_window.poll() and gpu_window.begin_frame():
            t1 = torch.rand(32, 32, 3, device="cuda")
            vultorch.show(t1, name="resize", window=gpu_window)
            gpu_window.end_frame()
        # Frame 2: show a 64×64 tensor with the same name — cache reallocates
        if gpu_window.poll() and gpu_window.begin_frame():
            t2 = torch.rand(64, 64, 3, device="cuda")
            vultorch.show(t2, name="resize", window=gpu_window)
            assert gpu_window._rgba_bufs["resize"].shape[0] == 64
            gpu_window.end_frame()


# ═══════════════════════════════════════════════════════════════════════
#  show() with CPU tensors
# ═══════════════════════════════════════════════════════════════════════

@requires_vultorch
@requires_cuda
@pytest.mark.gpu
class TestShowCPU:

    def test_show_cpu_rgba(self, gpu_window):
        import torch, vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            t = torch.rand(32, 32, 4)
            vultorch.show(t, name="cpu_rgba", window=gpu_window)
            gpu_window.end_frame()

    def test_show_cpu_rgb(self, gpu_window):
        import torch, vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            t = torch.rand(32, 32, 3)
            vultorch.show(t, name="cpu_rgb", window=gpu_window)
            gpu_window.end_frame()

    def test_show_cpu_gray(self, gpu_window):
        import torch, vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            t = torch.rand(32, 32)
            vultorch.show(t, name="cpu_gray", window=gpu_window)
            gpu_window.end_frame()


# ═══════════════════════════════════════════════════════════════════════
#  Window.upload_tensor (direct method)
# ═══════════════════════════════════════════════════════════════════════

@requires_vultorch
@requires_vultorch_cuda
@requires_cuda
@pytest.mark.gpu
class TestWindowUpload:

    def test_upload_tensor_cuda(self, gpu_window):
        import torch
        t = torch.rand(32, 32, 4, device="cuda")
        gpu_window.upload_tensor(t, name="direct_upload")

    def test_upload_tensor_cpu(self, gpu_window):
        import torch
        t = torch.rand(32, 32, 4)
        gpu_window.upload_tensor(t, name="cpu_direct")

    def test_texture_id_and_size(self, gpu_window):
        import torch, vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            t = torch.rand(48, 64, 4, device="cuda")
            vultorch.show(t, name="measure", window=gpu_window)
            tid = gpu_window.get_texture_id("measure")
            assert isinstance(tid, int)
            w, h = gpu_window.get_texture_size("measure")
            assert w == 64
            assert h == 48
            gpu_window.end_frame()


# ═══════════════════════════════════════════════════════════════════════
#  create_tensor() — zero-copy and fallback
# ═══════════════════════════════════════════════════════════════════════

@requires_vultorch
@requires_vultorch_cuda
@requires_cuda
@pytest.mark.gpu
class TestCreateTensor:

    def test_create_4ch_cuda(self, gpu_window):
        import vultorch
        t = vultorch.create_tensor(64, 64, channels=4, window=gpu_window)
        assert t.shape == (64, 64, 4)
        assert t.device.type == "cuda"

    def test_create_3ch_fallback(self, gpu_window):
        import vultorch
        t = vultorch.create_tensor(32, 32, channels=3, window=gpu_window)
        assert t.shape == (32, 32, 3)
        assert t.device.type == "cuda"

    def test_create_1ch_fallback(self, gpu_window):
        import vultorch
        t = vultorch.create_tensor(32, 32, channels=1, window=gpu_window)
        assert t.shape == (32, 32, 1)

    def test_create_cpu_device(self, gpu_window):
        import vultorch
        t = vultorch.create_tensor(32, 32, device="cpu", window=gpu_window)
        assert t.device.type == "cpu"
        assert t.shape == (32, 32, 4)

    def test_create_and_show(self, gpu_window):
        """Allocate via create_tensor and then display via show."""
        import vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            t = vultorch.create_tensor(32, 32, channels=4,
                                       name="zc", window=gpu_window)
            t[:] = 0.5
            vultorch.show(t, name="zc", window=gpu_window)
            gpu_window.end_frame()


# ═══════════════════════════════════════════════════════════════════════
#  SceneView
# ═══════════════════════════════════════════════════════════════════════

@requires_vultorch
@requires_vultorch_cuda
@requires_cuda
@pytest.mark.gpu
class TestSceneView:

    def test_init_defaults(self):
        import vultorch
        scene = vultorch.SceneView()
        assert scene.name == "SceneView"
        assert scene._width == 800
        assert scene._height == 600
        assert scene.msaa == 4
        assert isinstance(scene.camera, vultorch.Camera)
        assert isinstance(scene.light, vultorch.Light)
        assert scene.background == (0.12, 0.12, 0.14)

    def test_init_custom_params(self):
        import vultorch
        scene = vultorch.SceneView("MyScene", 400, 300, msaa=2)
        assert scene.name == "MyScene"
        assert scene._width == 400
        assert scene._height == 300
        assert scene.msaa == 2

    def test_msaa_setter(self):
        import vultorch
        scene = vultorch.SceneView()
        scene.msaa = 8
        assert scene._msaa == 8

    def test_scene_set_tensor_no_window_raises(self):
        import vultorch, torch
        saved = vultorch.Window._current
        try:
            vultorch.Window._current = None
            scene = vultorch.SceneView()
            with pytest.raises(RuntimeError, match="No active"):
                scene.set_tensor(torch.rand(32, 32, 4, device="cuda"))
        finally:
            vultorch.Window._current = saved

    def test_scene_render_basic(self, gpu_window):
        import torch, vultorch
        scene = vultorch.SceneView("test_scene", 128, 128, msaa=2)
        if gpu_window.poll() and gpu_window.begin_frame():
            vultorch.ui.begin("Scene", True, 0)
            t = torch.rand(64, 64, 4, device="cuda")
            scene.set_tensor(t)
            scene.render()
            vultorch.ui.end()
            gpu_window.end_frame()

    def test_scene_render_rgb(self, gpu_window):
        import torch, vultorch
        scene = vultorch.SceneView("rgb_scene", 128, 128, msaa=2)
        if gpu_window.poll() and gpu_window.begin_frame():
            vultorch.ui.begin("S2", True, 0)
            t = torch.rand(32, 32, 3, device="cuda")
            scene.set_tensor(t)
            scene.render()
            vultorch.ui.end()
            gpu_window.end_frame()

    def test_scene_render_grayscale(self, gpu_window):
        import torch, vultorch
        scene = vultorch.SceneView("gray_scene", 128, 128, msaa=1)
        if gpu_window.poll() and gpu_window.begin_frame():
            vultorch.ui.begin("S3", True, 0)
            t = torch.rand(32, 32, device="cuda")
            scene.set_tensor(t)
            scene.render()
            vultorch.ui.end()
            gpu_window.end_frame()

    def test_scene_camera_interaction(self, gpu_window):
        import torch, vultorch
        scene = vultorch.SceneView("cam_test", 128, 128, msaa=2)
        scene.camera.azimuth = 1.0
        scene.camera.elevation = 0.3
        scene.camera.distance = 5.0
        scene.light.intensity = 2.0
        scene.background = (0.5, 0.5, 0.5)
        if gpu_window.poll() and gpu_window.begin_frame():
            vultorch.ui.begin("S4", True, 0)
            t = torch.rand(32, 32, 4, device="cuda")
            scene.set_tensor(t)
            scene.render()
            vultorch.ui.end()
            gpu_window.end_frame()


# ═══════════════════════════════════════════════════════════════════════
#  Declarative API (View + Panel + Canvas) end-to-end
# ═══════════════════════════════════════════════════════════════════════

@requires_vultorch
@requires_vultorch_cuda
@requires_cuda
@pytest.mark.gpu
class TestDeclarativeE2E:

    def test_view_single_panel_step(self, gpu_window):
        """Use the session window — View wraps an existing Window."""
        import torch, vultorch
        from vultorch.app import View
        t = torch.rand(32, 32, 4, device="cuda")
        # Build a View that wraps the existing gpu_window
        view = View.__new__(View)
        view._win = gpu_window
        view._frame_fn = None
        view._panels = []
        view._panel_map = {}
        view._first_frame = True
        view._width = 256
        view._height = 256
        view.panel("Main").canvas("img").bind(t)
        ok = view.step()
        if ok:
            view.end_step()

    def test_view_multi_panel_step(self, gpu_window):
        import torch, vultorch
        from vultorch.app import View
        t1 = torch.rand(32, 32, 4, device="cuda")
        t2 = torch.rand(32, 32, 3, device="cuda")
        view = View.__new__(View)
        view._win = gpu_window
        view._frame_fn = None
        view._panels = []
        view._panel_map = {}
        view._first_frame = True
        view._width = 256
        view._height = 256
        view.panel("P1").canvas("c1").bind(t1)
        view.panel("P2").canvas("c2").bind(t2)
        for _ in range(2):
            ok = view.step()
            if ok:
                view.end_step()

    def test_view_on_frame_callback(self, gpu_window):
        import torch, vultorch
        from vultorch.app import View
        t = torch.rand(32, 32, 4, device="cuda")
        view = View.__new__(View)
        view._win = gpu_window
        view._frame_fn = None
        view._panels = []
        view._panel_map = {}
        view._first_frame = True
        view._width = 256
        view._height = 256
        view.panel("P").canvas("c").bind(t)
        called = [False]

        @view.on_frame
        def update():
            called[0] = True

        ok = view.step()
        if ok:
            view.end_step()
        assert called[0]

    def test_view_sidebar_panel(self, gpu_window):
        import torch, vultorch
        from vultorch.app import View
        t = torch.rand(32, 32, 4, device="cuda")
        view = View.__new__(View)
        view._win = gpu_window
        view._frame_fn = None
        view._panels = []
        view._panel_map = {}
        view._first_frame = True
        view._width = 256
        view._height = 256
        view.panel("Main").canvas("c").bind(t)
        view.panel("Sidebar", side="left", width=0.3)
        ok = view.step()
        if ok:
            view.end_step()

    def test_view_panel_retrieve_same(self, gpu_window):
        from vultorch.app import View
        view = View.__new__(View)
        view._win = gpu_window
        view._frame_fn = None
        view._panels = []
        view._panel_map = {}
        view._first_frame = True
        view._width = 256
        view._height = 256
        p1 = view.panel("P")
        p2 = view.panel("P")
        assert p1 is p2

    def test_view_properties(self, gpu_window):
        from vultorch.app import View
        view = View.__new__(View)
        view._win = gpu_window
        view._frame_fn = None
        view._panels = []
        view._panel_map = {}
        view._first_frame = True
        view._width = 256
        view._height = 256
        assert view.window is gpu_window
        ok = view.step()
        if ok:
            assert isinstance(view.fps, float)
            assert isinstance(view.time, float)
            view.end_step()

    def test_canvas_alloc(self, gpu_window):
        from vultorch.app import View
        view = View.__new__(View)
        view._win = gpu_window
        view._frame_fn = None
        view._panels = []
        view._panel_map = {}
        view._first_frame = True
        view._width = 256
        view._height = 256
        c = view.panel("P").canvas("alloc")
        t = c.alloc(32, 32, channels=4)
        assert t.shape == (32, 32, 4)
        ok = view.step()
        if ok:
            view.end_step()


# ═══════════════════════════════════════════════════════════════════════
#  ImGui ui submodule bindings
# ═══════════════════════════════════════════════════════════════════════

@requires_vultorch
@requires_cuda
@pytest.mark.gpu
class TestImGuiBindings:
    """Exercise the vultorch.ui bindings inside a valid frame context."""

    def test_text_widgets(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("widgets_text", True, 0)
            ui.text("Hello")
            ui.text_colored(1.0, 0.0, 0.0, 1.0, "Red text")
            ui.text_disabled("Disabled")
            ui.text_wrapped("Wrapped text that might be long")
            ui.label_text("Label", "Value")
            ui.bullet_text("Bullet")
            ui.end()
            gpu_window.end_frame()

    def test_button_widgets(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("widgets_btn", True, 0)
            ui.button("Click")
            ui.button("Sized", 100.0, 30.0)
            ui.small_button("Small")
            ui.invisible_button("##inv", 32.0, 32.0)
            ui.arrow_button("##arrow", 0)
            ui.radio_button("Radio", False)
            ui.end()
            gpu_window.end_frame()

    def test_checkbox(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("widgets_cb", True, 0)
            result = ui.checkbox("Option", True)
            assert isinstance(result, bool)
            ui.end()
            gpu_window.end_frame()

    def test_sliders(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("widgets_slider", True, 0)
            v = ui.slider_float("Float", 0.5, 0.0, 1.0)
            assert isinstance(v, float)
            v2 = ui.slider_float2("F2", 0.1, 0.2, 0.0, 1.0)
            assert len(v2) == 2
            v3 = ui.slider_float3("F3", 0.1, 0.2, 0.3, 0.0, 1.0)
            assert len(v3) == 3
            v4 = ui.slider_float4("F4", 0.1, 0.2, 0.3, 0.4, 0.0, 1.0)
            assert len(v4) == 4
            vi = ui.slider_int("Int", 50, 0, 100)
            assert isinstance(vi, int)
            va = ui.slider_angle("Angle", 0.0)
            assert isinstance(va, float)
            ui.end()
            gpu_window.end_frame()

    def test_drag_inputs(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("widgets_drag", True, 0)
            ui.drag_float("DF", 1.0)
            ui.drag_float2("DF2", 1.0, 2.0)
            ui.drag_float3("DF3", 1.0, 2.0, 3.0)
            ui.drag_int("DI", 5)
            ui.end()
            gpu_window.end_frame()

    def test_input_fields(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("widgets_input", True, 0)
            ui.input_float("IF", 1.0)
            ui.input_float2("IF2", 1.0, 2.0)
            ui.input_float3("IF3", 1.0, 2.0, 3.0)
            ui.input_float4("IF4", 1.0, 2.0, 3.0, 4.0)
            ui.input_int("II", 42)
            r = ui.input_text("IT", "hello", 256)
            assert isinstance(r, str)
            r2 = ui.input_text_multiline("ITM", "multi", 1024)
            assert isinstance(r2, str)
            ui.end()
            gpu_window.end_frame()

    def test_color_widgets(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("widgets_color", True, 0)
            c3 = ui.color_edit3("CE3", 1.0, 0.0, 0.0)
            assert len(c3) == 3
            c4 = ui.color_edit4("CE4", 1.0, 0.0, 0.0, 1.0)
            assert len(c4) == 4
            cp3 = ui.color_picker3("CP3", 0.5, 0.5, 0.5)
            assert len(cp3) == 3
            cp4 = ui.color_picker4("CP4", 0.5, 0.5, 0.5, 1.0)
            assert len(cp4) == 4
            ui.end()
            gpu_window.end_frame()

    def test_combo_listbox(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("widgets_combo", True, 0)
            idx = ui.combo("Combo", 0, ["A", "B", "C"])
            assert isinstance(idx, int)
            idx2 = ui.listbox("List", 0, ["X", "Y"], 3)
            assert isinstance(idx2, int)
            ui.end()
            gpu_window.end_frame()

    def test_tree_selectable_tabs(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("widgets_tree", True, 0)
            if ui.tree_node("Node"):
                ui.text("Inside tree")
                ui.tree_pop()
            ui.collapsing_header("Header")
            ui.selectable("Item", False)
            if ui.begin_tab_bar("tabs"):
                if ui.begin_tab_item("Tab1"):
                    ui.text("Tab content")
                    ui.end_tab_item()
                ui.end_tab_bar()
            ui.end()
            gpu_window.end_frame()

    def test_progress_bar(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("widgets_progress", True, 0)
            ui.progress_bar(0.7, -1.0, 0.0, "70%")
            ui.end()
            gpu_window.end_frame()

    def test_plot_functions(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("widgets_plot", True, 0)
            ui.plot_lines("Lines", [1.0, 2.0, 3.0, 2.0, 1.0])
            ui.plot_histogram("Hist", [1.0, 3.0, 2.0, 4.0])
            ui.end()
            gpu_window.end_frame()

    def test_image_widget(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("widgets_image", True, 0)
            # tex_id 0 is a valid "no texture" case for ImGui
            ui.image(0, 64.0, 64.0)
            ui.end()
            gpu_window.end_frame()


@requires_vultorch
@requires_cuda
@pytest.mark.gpu
class TestImGuiLayout:
    """Exercise layout, windowing, and docking bindings."""

    def test_layout_functions(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("layout", True, 0)
            ui.separator()
            ui.spacing()
            ui.new_line()
            ui.indent(10.0)
            ui.unindent(10.0)
            ui.begin_group()
            ui.end_group()
            ui.same_line()
            ui.dummy(10.0, 10.0)
            ui.end()
            gpu_window.end_frame()

    def test_child_window(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("child_parent", True, 0)
            ui.begin_child("##child1", 100.0, 100.0)
            ui.text("In child")
            ui.end_child()
            ui.end()
            gpu_window.end_frame()

    def test_columns(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("columns", True, 0)
            ui.columns(2, "col_id", True)
            ui.text("Left")
            ui.next_column()
            ui.text("Right")
            ui.columns(1)
            ui.end()
            gpu_window.end_frame()

    def test_tables(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("tables", True, 0)
            if ui.begin_table("tbl", 2):
                ui.table_setup_column("Col A")
                ui.table_setup_column("Col B")
                ui.table_headers_row()
                ui.table_next_row()
                ui.table_next_column()
                ui.text("A1")
                ui.table_next_column()
                ui.text("B1")
                ui.end_table()
            ui.end()
            gpu_window.end_frame()

    def test_popup(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("popup_test", True, 0)
            ui.open_popup("mypopup")
            if ui.begin_popup("mypopup"):
                ui.text("Popup content")
                ui.close_current_popup()
                ui.end_popup()
            ui.end()
            gpu_window.end_frame()

    def test_tooltip(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("tooltip_test", True, 0)
            ui.button("Hover me")
            ui.set_tooltip("Tooltip text")
            ui.end()
            gpu_window.end_frame()

    def test_id_stack(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("id_stack", True, 0)
            ui.push_id_str("scope1")
            ui.button("Btn")
            ui.pop_id()
            ui.push_id_int(42)
            ui.button("Btn")
            ui.pop_id()
            ui.end()
            gpu_window.end_frame()

    def test_style_functions(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.push_style_color(0, 1.0, 0.0, 0.0, 1.0)
            ui.begin("style_test", True, 0)
            ui.text("Styled")
            ui.end()
            ui.pop_style_color(1)
            gpu_window.end_frame()

    def test_style_var(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.push_style_var_float(0, 5.0)
            ui.begin("stylevar", True, 0)
            ui.end()
            ui.pop_style_var(1)
            gpu_window.end_frame()

    def test_cursor_viewport(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("cursor_vp", True, 0)
            pos = ui.get_cursor_pos()
            assert len(pos) == 2
            ui.set_cursor_pos(10.0, 10.0)
            avail = ui.get_content_region_avail()
            assert len(avail) == 2
            size = ui.get_window_size()
            assert len(size) == 2
            wpos = ui.get_window_pos()
            assert len(wpos) == 2
            ui.end()
            gpu_window.end_frame()

    def test_set_next_window(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.set_next_window_pos(100.0, 100.0)
            ui.set_next_window_size(200.0, 200.0)
            ui.begin("next_win", True, 0)
            ui.end()
            gpu_window.end_frame()

    def test_docking(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ds_id = ui.dock_space_over_viewport(flags=8)
            assert isinstance(ds_id, int)
            ui.begin("dock_test", True, 0)
            ui.end()
            gpu_window.end_frame()

    def test_menu_bar(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            if ui.begin_main_menu_bar():
                if ui.begin_menu("File"):
                    ui.menu_item("Open")
                    ui.end_menu()
                ui.end_main_menu_bar()
            gpu_window.end_frame()

    def test_style_presets(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.style_colors_dark()
            ui.style_colors_light()
            ui.style_colors_classic()
            ui.style_colors_dark()  # restore
            gpu_window.end_frame()

    def test_push_item_width(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("itemw", True, 0)
            ui.push_item_width(100.0)
            ui.slider_float("W", 0.5)
            ui.pop_item_width()
            ui.end()
            gpu_window.end_frame()

    def test_get_id(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("getid", True, 0)
            gid = ui.get_id("test_str")
            assert isinstance(gid, int)
            ui.end()
            gpu_window.end_frame()


@requires_vultorch
@requires_cuda
@pytest.mark.gpu
class TestImGuiDraw:
    """Exercise draw-list and utility bindings."""

    def test_draw_primitives(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("draw", True, 0)
            ui.draw_line(0, 0, 100, 100)
            ui.draw_rect(10, 10, 50, 50)
            ui.draw_rect_filled(60, 10, 90, 50)
            ui.draw_circle(50, 50, 20)
            ui.draw_circle_filled(80, 50, 15)
            ui.draw_text(10, 60, ui.col32(255, 255, 255), "Text")
            ui.end()
            gpu_window.end_frame()

    def test_col32(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            c = ui.col32(255, 0, 0, 128)
            assert isinstance(c, int)
            gpu_window.end_frame()

    def test_mouse_state(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            pos = ui.get_mouse_pos()
            assert len(pos) == 2
            ui.is_mouse_clicked(0)
            ui.is_mouse_double_clicked(0)
            ui.is_mouse_dragging(0)
            delta = ui.get_mouse_drag_delta()
            assert len(delta) == 2
            gpu_window.end_frame()

    def test_keyboard_state(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            # Key 0 = ImGuiKey_None, safe to query
            ui.is_key_pressed(0)
            ui.is_key_down(0)
            gpu_window.end_frame()

    def test_utility_queries(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            fps = ui.get_io_framerate()
            assert isinstance(fps, float)
            dt = ui.get_io_delta_time()
            assert isinstance(dt, float)
            t = ui.get_time()
            assert isinstance(t, float)
            fc = ui.get_frame_count()
            assert isinstance(fc, int)
            ds = ui.get_display_size()
            assert len(ds) == 2
            gpu_window.end_frame()

    def test_item_state(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.begin("item_state", True, 0)
            ui.button("B")
            ui.is_item_hovered()
            ui.is_item_active()
            ui.is_item_clicked()
            ui.is_item_focused()
            ui.is_item_edited()
            ui.is_item_deactivated_after_edit()
            ui.end()
            gpu_window.end_frame()

    def test_bg_draw_image(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.bg_draw_image(0, 0, 0, 100, 100)
            gpu_window.end_frame()

    def test_demo_metrics(self, gpu_window):
        from vultorch import ui
        if gpu_window.poll() and gpu_window.begin_frame():
            ui.show_demo_window()
            ui.show_metrics_window()
            gpu_window.end_frame()
