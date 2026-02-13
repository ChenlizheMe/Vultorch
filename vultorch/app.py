"""Vultorch declarative API — View / Panel / Canvas hierarchy.

Usage — pure display::

    view = vultorch.View("Hello", 512, 512)
    view.panel("Viewer").canvas("img").bind(tensor)
    view.run()

Usage — with controls::

    view = vultorch.View("Demo", 1280, 720)
    preview  = view.panel("Preview")
    controls = view.panel("Controls", side="left", width=0.22)
    rgb = preview.canvas("RGB")
    rgb.bind(t)

    @view.on_frame
    def update():
        speed = controls.slider("Speed", 0, 10)
        t[:,:,0] = (x + view.time * speed).sin()

    view.run()

Usage — training loop::

    view = vultorch.View("Train", 1024, 768)
    output = view.panel("Output").canvas("Result")
    for epoch in range(100):
        result = model(input)
        output.bind(result)
        if not view.step():
            break
        view.end_step()
    view.close()
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from ._vultorch import ui
from . import Window, show, HAS_CUDA

__all__ = ["View"]

# ImGui direction constants
_DIR_LEFT = 0
_DIR_RIGHT = 1
_DIR_UP = 2
_DIR_DOWN = 3


# ═══════════════════════════════════════════════════════════════════════
#  Canvas — a named GPU texture slot bound to a tensor
# ═══════════════════════════════════════════════════════════════════════

class Canvas:
    """A display surface that renders a bound tensor as an ImGui image.

    Created via :meth:`Panel.canvas`.  Not instantiated directly.
    """

    __slots__ = ("_name", "_panel", "_tensor", "_filter", "_fit")

    def __init__(self, name: str, panel: "Panel"):
        self._name = name
        self._panel = panel
        self._tensor = None     # bound tensor reference
        self._filter = "linear"
        self._fit = True        # auto-fill available region

    def bind(self, tensor) -> "Canvas":
        """Bind a CUDA tensor.  The canvas will display it every frame.

        Can be called multiple times to switch the data source.
        Returns *self* for chaining: ``canvas.bind(t)``
        """
        self._tensor = tensor
        return self

    def alloc(self, height: int, width: int, channels: int = 4,
              device: str = "cuda:0") -> Any:
        """Allocate Vulkan-shared memory and auto-bind.

        Returns a ``torch.Tensor`` with zero-copy Vulkan interop.
        """
        from . import create_tensor
        win = self._panel._view._win
        t = create_tensor(height, width, channels, device,
                          name=self._name, window=win)
        self._tensor = t
        return t

    @property
    def filter(self) -> str:
        """Sampling filter: ``'linear'`` (interpolation) or ``'nearest'`` (no interpolation)."""
        return self._filter

    @filter.setter
    def filter(self, value: str):
        self._filter = value

    @property
    def fit(self) -> bool:
        """Whether the canvas auto-fills available panel space (default ``True``).

        When ``False`` the image is displayed at its native tensor resolution."""
        return self._fit

    @fit.setter
    def fit(self, value: bool):
        self._fit = value

    def _render(self, *, fit_height: float = 0.0):
        """Internal: upload and render the bound tensor.

        Args:
            fit_height: When > 0, use this as the display height instead
                        of querying the remaining region (set by the panel
                        when multiple fit-canvases share space).
        """
        if self._tensor is None:
            ui.text(f"[Canvas '{self._name}': no tensor bound]")
            return
        kw: dict[str, Any] = dict(
            name=self._name, filter=self._filter,
            window=self._panel._view._win,
        )
        if self._fit:
            avail_w, avail_h = ui.get_content_region_avail()
            kw["width"] = avail_w
            kw["height"] = fit_height if fit_height > 0 else avail_h
        show(self._tensor, **kw)


# ═══════════════════════════════════════════════════════════════════════
#  Panel — a dockable ImGui window owning canvases and widgets
# ═══════════════════════════════════════════════════════════════════════

class Panel:
    """A dockable panel containing canvases and widgets.

    Created via :meth:`View.panel`.  Not instantiated directly.
    """

    def __init__(self, name: str, view: "View", *,
                 side: Optional[str] = None, width: float = 0.0):
        self._name = name
        self._view = view
        self._side = side          # "left" / "right" / None
        self._width = width        # ratio for sidebar
        self._canvases: list[Canvas] = []
        self._state: dict[str, Any] = {}
        self._row_stack: list[bool] = []

    # ── Canvas factory ──────────────────────────────────────────────

    def canvas(self, name: str, *, filter: str = "linear",
               fit: bool = True) -> Canvas:
        """Create a named canvas in this panel.

        Args:
            name:   Unique label for the canvas.
            filter: ``'linear'`` (bilinear interpolation) or ``'nearest'``.
            fit:    Auto-fill available panel space (default ``True``).
                    Set ``False`` to display at native tensor resolution.
        """
        c = Canvas(name, self)
        c._filter = filter
        c._fit = fit
        self._canvases.append(c)
        return c

    # ── Row context manager ─────────────────────────────────────────

    def row(self):
        """Context manager — place child widgets side-by-side."""
        return _RowContext(self)

    # ── Widgets (with automatic state) ──────────────────────────────

    def text(self, text: str):
        self._before_widget()
        ui.text(str(text))

    def text_colored(self, r: float, g: float, b: float, a: float,
                     text: str):
        self._before_widget()
        ui.text_colored(r, g, b, a, str(text))

    def text_wrapped(self, text: str):
        self._before_widget()
        ui.text_wrapped(str(text))

    def separator(self):
        ui.separator()

    def button(self, label: str) -> bool:
        self._before_widget()
        return ui.button(label)

    def checkbox(self, label: str, *, default: bool = False) -> bool:
        key = "_cb:" + label
        if key not in self._state:
            self._state[key] = default
        self._before_widget()
        self._state[key] = ui.checkbox(label, self._state[key])
        return self._state[key]

    def slider(self, label: str, min: float = 0.0, max: float = 1.0, *,
               default: Optional[float] = None) -> float:
        key = "_sf:" + label
        if key not in self._state:
            self._state[key] = default if default is not None else min
        self._before_widget()
        self._state[key] = ui.slider_float(label, self._state[key], min, max)
        return self._state[key]

    def slider_int(self, label: str, min: int = 0, max: int = 100, *,
                   default: int = 0) -> int:
        key = "_si:" + label
        if key not in self._state:
            self._state[key] = default
        self._before_widget()
        self._state[key] = ui.slider_int(label, self._state[key], min, max)
        return self._state[key]

    def color_picker(self, label: str, *,
                     default: Tuple[float, float, float] = (1.0, 1.0, 1.0)
                     ) -> Tuple[float, float, float]:
        key = "_cp:" + label
        if key not in self._state:
            self._state[key] = default
        self._before_widget()
        c = self._state[key]
        self._state[key] = ui.color_edit3(label, c[0], c[1], c[2])
        return self._state[key]

    def combo(self, label: str, items: List[str], *,
              default: int = 0) -> int:
        key = "_co:" + label
        if key not in self._state:
            self._state[key] = default
        self._before_widget()
        self._state[key] = ui.combo(label, self._state[key], items)
        return self._state[key]

    def input_text(self, label: str, *, default: str = "",
                   max_length: int = 256) -> str:
        key = "_it:" + label
        if key not in self._state:
            self._state[key] = default
        self._before_widget()
        self._state[key] = ui.input_text(label, self._state[key], max_length)
        return self._state[key]

    def plot(self, values: Sequence[float], *, label: str = "##plot",
             overlay: str = "", width: float = 0.0, height: float = 80.0):
        self._before_widget()
        ui.plot_lines(label, list(values), 0, overlay,
                      3.4028235e+38, 3.4028235e+38, width, height)

    def progress(self, fraction: float, *, overlay: str = ""):
        self._before_widget()
        ui.progress_bar(fraction, overlay=overlay)

    # ── Internal ────────────────────────────────────────────────────

    def _before_widget(self):
        if self._row_stack:
            if self._row_stack[-1]:
                self._row_stack[-1] = False
            else:
                ui.same_line()

    def _render_canvases(self):
        """Render all bound canvases.

        When multiple canvases have ``fit=True`` the available vertical
        space is divided equally using ImGui child regions (no guessing
        about item spacing).
        """
        n_fit = sum(1 for c in self._canvases if c._fit)
        if n_fit > 1:
            avail_w, avail_h = ui.get_content_region_avail()
            per_h = max(1.0, avail_h / n_fit)

        for c in self._canvases:
            if c._fit and n_fit > 1:
                # Wrap each fit-canvas in a fixed-height child region
                # so it gets exactly 1/N of the panel height.
                ui.begin_child(f"##cv_{c._name}", 0.0, per_h, 0, 0)
                c._render(fit_height=0.0)   # 0 => fill the child region
                ui.end_child()
            else:
                c._render(fit_height=0.0)


# ═══════════════════════════════════════════════════════════════════════
#  Row context manager
# ═══════════════════════════════════════════════════════════════════════

class _RowContext:
    __slots__ = ("_panel",)

    def __init__(self, panel: Panel):
        self._panel = panel

    def __enter__(self):
        self._panel._row_stack.append(True)
        ui.begin_group()
        return self

    def __exit__(self, *exc):
        ui.end_group()
        self._panel._row_stack.pop()
        return False


# ═══════════════════════════════════════════════════════════════════════
#  View — the top-level window
# ═══════════════════════════════════════════════════════════════════════

class View:
    """Top-level Vultorch window.

    Example::

        view = vultorch.View("Hello", 512, 512)
        view.panel("Viewer").canvas("img").bind(tensor)
        view.run()
    """

    def __init__(self, title: str = "Vultorch", width: int = 1280,
                 height: int = 720):
        self._win = Window(title, width, height)
        self._frame_fn = None       # type: Any
        self._panels: list[Panel] = []
        self._panel_map: dict[str, Panel] = {}
        self._first_frame = True
        self._width = width
        self._height = height

    # ── Panel factory ───────────────────────────────────────────────

    def panel(self, name: str, *, side: Optional[str] = None,
              width: float = 0.0) -> Panel:
        """Create (or retrieve) a dockable panel.

        Args:
            name:   Panel title (shown in the tab/title bar).
            side:   ``"left"`` or ``"right"`` to dock as a sidebar.
            width:  Sidebar width ratio (e.g. 0.22).  Ignored if *side* is None.
        """
        if name in self._panel_map:
            return self._panel_map[name]
        p = Panel(name, self, side=side, width=width)
        self._panels.append(p)
        self._panel_map[name] = p
        return p

    # ── Frame callback ──────────────────────────────────────────────

    def on_frame(self, fn):
        """Decorator — register an optional per-frame callback.

        Used for dynamic updates (widgets, rebinding, tensor mutation).
        Canvas rendering is automatic — no need to call show().
        """
        self._frame_fn = fn
        return fn

    # ── Event loops ─────────────────────────────────────────────────

    def run(self):
        """Blocking event loop.  Renders all panels and their canvases
        every frame.  If a ``@on_frame`` callback is registered it runs
        first (for widget logic / data updates)."""
        try:
            while self._win.poll():
                if not self._win.begin_frame():
                    continue

                dockspace_id = ui.dock_space_over_viewport(flags=8)

                if self._first_frame:
                    self._setup_layout(dockspace_id)
                    self._first_frame = False

                # User callback (widgets, rebind, data mutation)
                if self._frame_fn is not None:
                    self._frame_fn()

                # Render all panels and their canvases
                self._render_all()

                self._win.end_frame()
        finally:
            self._win.destroy()

    def step(self) -> bool:
        """Process one frame (training-loop integration).

        Returns ``False`` when the window should close.
        Pair with :meth:`end_step`.
        """
        if not self._win.poll():
            return False
        if not self._win.begin_frame():
            return True

        dockspace_id = ui.dock_space_over_viewport(flags=8)
        if self._first_frame:
            self._setup_layout(dockspace_id)
            self._first_frame = False

        if self._frame_fn is not None:
            self._frame_fn()

        self._render_all()
        return True

    def end_step(self):
        """Finish the current step frame."""
        self._win.end_frame()

    def close(self):
        """Explicitly close the window."""
        self._win.destroy()

    # ── Properties ──────────────────────────────────────────────────

    @property
    def fps(self) -> float:
        return ui.get_io_framerate()

    @property
    def time(self) -> float:
        return ui.get_time()

    @property
    def window(self) -> Window:
        return self._win

    # ── Internal ────────────────────────────────────────────────────

    def _render_all(self):
        """Render every panel: open ImGui window, draw canvases."""
        for p in self._panels:
            ui.begin(p._name, True, 0)
            p._render_canvases()
            ui.end()

    def _setup_layout(self, dockspace_id: int):
        """Auto-generate docking layout from declared panels."""
        ui.dock_builder_remove_node(dockspace_id)
        ui.dock_builder_add_node(dockspace_id, 1 << 10)
        ui.dock_builder_set_node_size(
            dockspace_id, float(self._width), float(self._height))

        remaining = dockspace_id

        # Sidebars first
        sidebars = [p for p in self._panels if p._side]
        mains = [p for p in self._panels if not p._side]

        for p in sidebars:
            direction = _DIR_LEFT if p._side == "left" else _DIR_RIGHT
            ratio = p._width if p._width > 0 else 0.2
            id_side, remaining = ui.dock_builder_split_node(
                remaining, direction, ratio)
            ui.dock_builder_dock_window(p._name, id_side)

        # Remaining panels distributed equally
        n = len(mains)
        if n == 0:
            pass
        elif n == 1:
            ui.dock_builder_dock_window(mains[0]._name, remaining)
        else:
            current = remaining
            for i in range(n - 1):
                ratio = 1.0 / (n - i)
                id_top, current = ui.dock_builder_split_node(
                    current, _DIR_UP, ratio)
                ui.dock_builder_dock_window(mains[i]._name, id_top)
            ui.dock_builder_dock_window(mains[-1]._name, current)

        ui.dock_builder_finish(dockspace_id)
