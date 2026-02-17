"""Tests for Panel widget wrappers in vultorch/app.py — GPU required.

Coverage targets:
  - Panel.text / text_colored / text_wrapped
  - Panel.separator / button / checkbox
  - Panel.slider / slider_int / color_picker
  - Panel.combo / input_text / plot / progress
  - Panel._auto_state management
  - Panel.row() context manager
  - Canvas.bind / Canvas.alloc / Canvas.filter / Canvas.fit

Strategy:  We drive frames *manually* via the shared gpu_window rather than
via ``View.step()`` because ``step()`` calls ``_render_all()`` (which opens
and closes every panel ImGui window) before returning, so widget calls placed
*after* ``step()`` would happen outside any ImGui window — which causes
undefined behaviour / hangs.

Instead each test:
  1. ``gpu_window.poll()`` + ``gpu_window.begin_frame()``
  2. ``ui.begin(...)`` — opens an ImGui window
  3. Panel widget calls
  4. ``ui.end()`` — closes the ImGui window
  5. ``gpu_window.end_frame()``
"""

import pytest
from conftest import requires_vultorch, requires_cuda, requires_vultorch_cuda


def _make_panel(name="TestPanel"):
    """Create a standalone Panel (no View needed for widget testing)."""
    from vultorch.app import Panel
    panel = Panel.__new__(Panel)
    panel._name = name
    panel._view = None
    panel._side = None
    panel._width = 0.0
    panel._canvases = []
    panel._state = {}
    panel._row_stack = []
    return panel


@requires_vultorch
@requires_vultorch_cuda
@requires_cuda
@pytest.mark.gpu
class TestPanelWidgetsGPU:
    """Exercise Panel widget methods inside a real ImGui frame."""

    def test_text(self, gpu_window):
        import vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            vultorch.ui.begin("W_text", True, 0)
            p = _make_panel()
            p.text("Hello")
            vultorch.ui.end()
            gpu_window.end_frame()

    def test_text_colored(self, gpu_window):
        import vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            vultorch.ui.begin("W_tc", True, 0)
            p = _make_panel()
            p.text_colored(1.0, 0.0, 0.0, 1.0, "Red")
            vultorch.ui.end()
            gpu_window.end_frame()

    def test_text_wrapped(self, gpu_window):
        import vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            vultorch.ui.begin("W_tw", True, 0)
            p = _make_panel()
            p.text_wrapped("Long text " * 20)
            vultorch.ui.end()
            gpu_window.end_frame()

    def test_separator(self, gpu_window):
        import vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            vultorch.ui.begin("W_sep", True, 0)
            p = _make_panel()
            p.separator()
            vultorch.ui.end()
            gpu_window.end_frame()

    def test_button(self, gpu_window):
        import vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            vultorch.ui.begin("W_btn", True, 0)
            p = _make_panel()
            result = p.button("Click")
            assert isinstance(result, bool)
            vultorch.ui.end()
            gpu_window.end_frame()

    def test_checkbox(self, gpu_window):
        import vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            vultorch.ui.begin("W_cb", True, 0)
            p = _make_panel()
            val = p.checkbox("Check", default=True)
            assert isinstance(val, bool)
            vultorch.ui.end()
            gpu_window.end_frame()

    def test_slider(self, gpu_window):
        import vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            vultorch.ui.begin("W_sl", True, 0)
            p = _make_panel()
            val = p.slider("Slider", 0.0, 1.0, default=0.5)
            assert isinstance(val, float)
            vultorch.ui.end()
            gpu_window.end_frame()

    def test_slider_int(self, gpu_window):
        import vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            vultorch.ui.begin("W_si", True, 0)
            p = _make_panel()
            val = p.slider_int("SI", 0, 100, default=50)
            assert isinstance(val, int)
            vultorch.ui.end()
            gpu_window.end_frame()

    def test_color_picker(self, gpu_window):
        import vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            vultorch.ui.begin("W_cp", True, 0)
            p = _make_panel()
            val = p.color_picker("Color", default=(0.5, 0.5, 0.5))
            assert len(val) == 3
            vultorch.ui.end()
            gpu_window.end_frame()

    def test_combo(self, gpu_window):
        import vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            vultorch.ui.begin("W_co", True, 0)
            p = _make_panel()
            idx = p.combo("Combo", ["A", "B", "C"], default=0)
            assert isinstance(idx, int)
            vultorch.ui.end()
            gpu_window.end_frame()

    def test_input_text(self, gpu_window):
        import vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            vultorch.ui.begin("W_it", True, 0)
            p = _make_panel()
            txt = p.input_text("Input", default="hello")
            assert isinstance(txt, str)
            vultorch.ui.end()
            gpu_window.end_frame()

    def test_plot(self, gpu_window):
        import vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            vultorch.ui.begin("W_plt", True, 0)
            p = _make_panel()
            p.plot([1.0, 2.0, 3.0, 2.0])
            vultorch.ui.end()
            gpu_window.end_frame()

    def test_progress(self, gpu_window):
        import vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            vultorch.ui.begin("W_prog", True, 0)
            p = _make_panel()
            p.progress(0.75)
            vultorch.ui.end()
            gpu_window.end_frame()

    def test_row_context(self, gpu_window):
        import vultorch
        if gpu_window.poll() and gpu_window.begin_frame():
            vultorch.ui.begin("W_row", True, 0)
            p = _make_panel()
            with p.row():
                p.text("Left")
                p.text("Right")
            vultorch.ui.end()
            gpu_window.end_frame()

    def test_auto_state_persistence(self, gpu_window):
        """Widget state should persist across frames via _auto_state."""
        import vultorch
        p = _make_panel()
        # Frame 1
        if gpu_window.poll() and gpu_window.begin_frame():
            vultorch.ui.begin("W_state1", True, 0)
            p.slider("persist", 0.0, 1.0, default=0.5)
            vultorch.ui.end()
            gpu_window.end_frame()
        # Frame 2
        if gpu_window.poll() and gpu_window.begin_frame():
            vultorch.ui.begin("W_state2", True, 0)
            p.slider("persist", 0.0, 1.0, default=0.5)
            vultorch.ui.end()
            gpu_window.end_frame()
        # State key should exist
        assert "_sf:persist" in p._state


@requires_vultorch
@requires_vultorch_cuda
@requires_cuda
@pytest.mark.gpu
class TestCanvasGPU:
    """Exercise Canvas methods with actual GPU tensors."""

    def _make_view(self, gpu_window):
        from vultorch.app import View
        view = View.__new__(View)
        view._win = gpu_window
        view._frame_fn = None
        view._panels = []
        view._panel_map = {}
        view._first_frame = True
        view._width = 256
        view._height = 256
        return view

    def test_bind_and_render(self, gpu_window):
        import torch
        v = self._make_view(gpu_window)
        c = v.panel("P").canvas("img")
        t = torch.rand(32, 32, 4, device="cuda")
        c.bind(t)
        if v.step():
            v.end_step()

    def test_alloc_and_write(self, gpu_window):
        v = self._make_view(gpu_window)
        c = v.panel("P").canvas("alloc")
        t = c.alloc(64, 64, channels=4)
        t[:] = 0.5
        if v.step():
            v.end_step()

    def test_filter_nearest(self, gpu_window):
        import torch
        v = self._make_view(gpu_window)
        c = v.panel("P").canvas("flt")
        t = torch.rand(32, 32, 4, device="cuda")
        c.bind(t)
        c.filter = "nearest"
        if v.step():
            v.end_step()

    def test_filter_linear(self, gpu_window):
        import torch
        v = self._make_view(gpu_window)
        c = v.panel("P").canvas("flt2")
        t = torch.rand(32, 32, 4, device="cuda")
        c.bind(t)
        c.filter = "linear"
        if v.step():
            v.end_step()

    def test_fit_mode(self, gpu_window):
        import torch
        v = self._make_view(gpu_window)
        c = v.panel("P").canvas("fit")
        t = torch.rand(32, 32, 4, device="cuda")
        c.bind(t)
        c.fit = False
        if v.step():
            v.end_step()

    def test_multiple_canvases(self, gpu_window):
        import torch
        v = self._make_view(gpu_window)
        p = v.panel("P")
        t1 = torch.rand(32, 32, 4, device="cuda")
        t2 = torch.rand(64, 64, 4, device="cuda")
        p.canvas("c1").bind(t1)
        p.canvas("c2").bind(t2)
        if v.step():
            v.end_step()
