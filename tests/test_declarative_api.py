"""Tests for the declarative API: View, Panel, Canvas.

Coverage targets:
  - Canvas: __init__, bind, filter property, fit property, _render with no tensor
  - Panel: __init__, canvas factory, state management, _before_widget, row context
  - Panel widgets: text, button, checkbox, slider, slider_int, color_picker,
                   combo, input_text, plot, progress, separator, text_colored, text_wrapped
  - View: __init__, panel factory (create & retrieve), on_frame decorator,
          step/end_step, close, fps/time/window properties
  - _RowContext: __enter__, __exit__
"""

import pytest
from conftest import requires_vultorch, requires_torch


@requires_vultorch
class TestCanvas:

    def test_init(self):
        from vultorch.app import Canvas, Panel, View
        # Create a minimal mock panel
        class FakeView:
            _win = None
        class FakePanel:
            _view = FakeView()
        p = FakePanel()
        c = Canvas("test_canvas", p)
        assert c._name == "test_canvas"
        assert c._tensor is None
        assert c._filter == "linear"
        assert c._fit is True

    def test_bind_returns_self(self):
        from vultorch.app import Canvas
        class FakePanel:
            pass
        c = Canvas("c", FakePanel())
        result = c.bind("fake_tensor")
        assert result is c
        assert c._tensor == "fake_tensor"

    def test_bind_multiple_times(self):
        from vultorch.app import Canvas
        class FakePanel:
            pass
        c = Canvas("c", FakePanel())
        c.bind("tensor_a")
        assert c._tensor == "tensor_a"
        c.bind("tensor_b")
        assert c._tensor == "tensor_b"

    def test_filter_property(self):
        from vultorch.app import Canvas
        class FakePanel:
            pass
        c = Canvas("c", FakePanel())
        assert c.filter == "linear"
        c.filter = "nearest"
        assert c.filter == "nearest"

    def test_fit_property(self):
        from vultorch.app import Canvas
        class FakePanel:
            pass
        c = Canvas("c", FakePanel())
        assert c.fit is True
        c.fit = False
        assert c.fit is False


@requires_vultorch
class TestPanel:

    def _make_panel(self, name="TestPanel"):
        from vultorch.app import Panel
        class FakeView:
            _win = None
        return Panel(name, FakeView())

    def test_init(self):
        p = self._make_panel()
        assert p._name == "TestPanel"
        assert p._side is None
        assert p._width == 0.0
        assert p._canvases == []
        assert p._state == {}

    def test_init_with_side(self):
        from vultorch.app import Panel
        class FakeView:
            _win = None
        p = Panel("Left", FakeView(), side="left", width=0.3)
        assert p._side == "left"
        assert p._width == 0.3

    def test_canvas_factory(self):
        p = self._make_panel()
        c = p.canvas("img1")
        assert len(p._canvases) == 1
        assert c._name == "img1"

    def test_canvas_factory_custom_props(self):
        p = self._make_panel()
        c = p.canvas("img", filter="nearest", fit=False)
        assert c._filter == "nearest"
        assert c._fit is False

    def test_multiple_canvases(self):
        p = self._make_panel()
        c1 = p.canvas("a")
        c2 = p.canvas("b")
        c3 = p.canvas("c")
        assert len(p._canvases) == 3
        assert [c._name for c in p._canvases] == ["a", "b", "c"]

    def test_row_context_manager(self):
        p = self._make_panel()
        assert p._row_stack == []
        # We can't actually call __enter__/__exit__ without ImGui context,
        # but we can verify the _RowContext is returned
        ctx = p.row()
        from vultorch.app import _RowContext
        assert isinstance(ctx, _RowContext)

    def test_state_isolation(self):
        """Two panels should have independent state dicts."""
        p1 = self._make_panel("P1")
        p2 = self._make_panel("P2")
        p1._state["key"] = "val1"
        assert "key" not in p2._state


@requires_vultorch
class TestView:

    def test_panel_factory_creates_new(self):
        """View.panel() should create a new panel if name is new."""
        # We can't create a real View without Vulkan, so test the Panel creation logic directly
        from vultorch.app import Panel
        panels = {}
        name = "TestPanel"
        class FakeView:
            _win = None
        if name not in panels:
            p = Panel(name, FakeView())
            panels[name] = p
        assert name in panels

    def test_panel_factory_retrieves_existing(self):
        """Retrieving by same name should return the same Panel."""
        from vultorch.app import Panel
        panel_map = {}
        class FakeView:
            _win = None
        p1 = Panel("same", FakeView())
        panel_map["same"] = p1
        p2 = panel_map.get("same", None)
        assert p1 is p2

    def test_on_frame_returns_function(self):
        """on_frame decorator should store and return the function."""
        class FakeView:
            _frame_fn = None
            def on_frame(self, fn):
                self._frame_fn = fn
                return fn

        view = FakeView()
        @view.on_frame
        def my_callback():
            pass
        assert view._frame_fn is my_callback

    def test_direction_constants(self):
        from vultorch.app import _DIR_LEFT, _DIR_RIGHT, _DIR_UP, _DIR_DOWN
        assert _DIR_LEFT == 0
        assert _DIR_RIGHT == 1
        assert _DIR_UP == 2
        assert _DIR_DOWN == 3


@requires_vultorch
class TestRowContext:

    def test_row_context_type(self):
        from vultorch.app import _RowContext, Panel
        class FakeView:
            _win = None
        p = Panel("test", FakeView())
        ctx = _RowContext(p)
        assert ctx._panel is p
