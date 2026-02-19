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
        t[:,:,0] = (x + view.time * speed).sin()

    @controls.on_frame
    def draw_controls():
        speed = controls.slider("Speed", 0, 10)

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

Usage — headless (no window / display)::

    view = vultorch.View("Train", headless=True)
    output = view.panel("Output").canvas("Result")
    ctrl = view.panel("Controls")

    @view.on_frame
    def train():
        result = model(input)
        output.bind(result)

    @ctrl.on_frame
    def draw():
        ctrl.text(f"Loss: {loss:.4f}")   # no-op in headless

    view.run(max_frames=1000)
    output.canvas("Result").save("final.png")
"""

from __future__ import annotations

import time as _time
from typing import Any, List, Optional, Sequence, Tuple

# Lazy imports — ui and Window are only needed in non-headless mode.
# We import them at module level but guard their use behind _headless checks.
try:
    from ._vultorch import ui as _ui
except ImportError:
    _ui = None  # type: ignore[assignment]

try:
    from . import Window as _Window, show as _show, HAS_CUDA as _HAS_CUDA
except ImportError:
    _Window = None  # type: ignore[assignment,misc]
    _show = None    # type: ignore[assignment]
    _HAS_CUDA = False

# Keep backward-compatible re-exports for non-headless code that does
# ``from vultorch.app import View`` — ui / Window / show were previously
# importable from this module.
ui = _ui
Window = _Window
show = _show
HAS_CUDA = _HAS_CUDA

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

    __slots__ = ("_name", "_panel", "_tensor", "_filter", "_fit",
                 "_recorder")

    def __init__(self, name: str, panel: "Panel"):
        self._name = name
        self._panel = panel
        self._tensor = None     # bound tensor reference
        self._filter = "linear"
        self._fit = True        # auto-fill available region
        self._recorder = None   # type: ignore[assignment]

    def bind(self, tensor) -> "Canvas":
        """Bind a tensor (CUDA or CPU).  The canvas will display it every frame.

        Can be called multiple times to switch the data source.
        Returns *self* for chaining: ``canvas.bind(t)``
        """
        self._tensor = tensor
        return self

    def alloc(self, height: int, width: int, channels: int = 4,
              device: str = "cuda:0") -> Any:
        """Allocate Vulkan-shared memory and auto-bind.

        Returns a ``torch.Tensor``.  On CUDA this uses zero-copy Vulkan
        interop; on CPU a regular tensor is returned and `show()` uses
        host staging.

        In headless mode a regular tensor is always returned (no Vulkan
        interop available).
        """
        if self._panel._view._headless:
            import torch
            t = torch.zeros(height, width, channels, dtype=torch.float32,
                            device=device)
            self._tensor = t
            return t
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

    def save(self, path: str, *, channels: int = 0,
             size: "tuple[int, int] | None" = None,
             quality: int = 95) -> None:
        """Save the currently bound tensor to an image file.

        Convenience wrapper around ``vultorch.imwrite()``.

        Args:
            path:     Output file path.
            channels: Output channels (0 = same as tensor).
            size:     Optional ``(height, width)`` to resize before saving.
            quality:  JPEG quality (1–100).
        """
        if self._tensor is None:
            raise RuntimeError(f"Canvas '{self._name}' has no tensor bound")
        from . import imwrite
        imwrite(path, self._tensor, channels=channels, size=size,
                quality=quality)

    # ── Recording ───────────────────────────────────────────────────

    def start_recording(self, path: str, *, fps: int = 30,
                        quality: float = 0.8) -> "Canvas":
        """Start recording this canvas to an animated GIF.

        Every frame the bound tensor is captured and encoded.
        Call :meth:`stop_recording` to finalize the file.

        Args:
            path:    Output file path (must end with ``.gif``).
                     Requires ``Pillow`` (``pip install Pillow``).
            fps:     Frames per second for the output (default 30).
            quality: Image quality, 0–1.  Controls the number of
                     colours per frame (0 → 2, 1 → 256).  Lower
                     values produce smaller files.  Default 0.8.

        Returns:
            *self* for chaining.

        Example::

            canvas.start_recording("demo.gif", fps=15, quality=0.6)
        """
        from .recorder import Recorder
        if self._recorder is not None and self._recorder.recording:
            self._recorder.stop()
        self._recorder = Recorder(path, fps=fps, quality=quality)
        return self

    def stop_recording(self) -> "str | None":
        """Stop recording and finalize the output file.

        Returns:
            The absolute path of the saved file, or ``None`` if not
            recording.
        """
        if self._recorder is None or not self._recorder.recording:
            return None
        return self._recorder.stop()

    @property
    def is_recording(self) -> bool:
        """``True`` while the canvas is actively recording."""
        return self._recorder is not None and self._recorder.recording

    @property
    def is_saving(self) -> bool:
        """``True`` while a recorded GIF is being written to disk."""
        return self._recorder is not None and self._recorder.saving

    def _feed_recorder(self):
        """Internal: send the current tensor frame to the recorder."""
        if (self._recorder is not None
                and self._recorder.recording
                and self._tensor is not None):
            self._recorder.feed(self._tensor)

    def _render(self, *, fit_height: float = 0.0):
        """Internal: upload and render the bound tensor.

        Args:
            fit_height: When > 0, use this as the display height instead
                        of querying the remaining region (set by the panel
                        when multiple fit-canvases share space).
        """
        # Feed the recorder regardless of headless mode
        self._feed_recorder()

        if self._panel._view._headless:
            return  # no display in headless mode
        if self._tensor is None:
            _ui.text(f"[Canvas '{self._name}': no tensor bound]")
            return
        kw: dict[str, Any] = dict(
            name=self._name, filter=self._filter,
            window=self._panel._view._win,
        )
        if self._fit:
            avail_w, avail_h = _ui.get_content_region_avail()
            kw["width"] = avail_w
            kw["height"] = fit_height if fit_height > 0 else avail_h
        _show(self._tensor, **kw)


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
        self._side = side          # "left" / "right" / "bottom" / "top" / None
        self._width = width        # ratio for sidebar
        self._canvases: list[Canvas] = []
        self._state: dict[str, Any] = {}
        self._row_stack: list[bool] = []
        self._frame_fn: Any = None

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

    # ── Per-panel frame callback ──────────────────────────────────

    def on_frame(self, fn):
        """Decorator — register a per-frame callback for this panel.

        The callback runs inside the panel's ImGui window context,
        after canvases are rendered.  Use panel widget methods
        (``text``, ``button``, ``slider``, …) inside the callback.
        """
        self._frame_fn = fn
        return fn

    # ── Row context manager ─────────────────────────────────────────

    def row(self):
        """Context manager — place child widgets side-by-side."""
        if self._view._headless:
            return _HeadlessRowContext()
        return _RowContext(self)

    # ── Programmatic state access (useful in headless mode) ─────────

    def set(self, label: str, value: Any) -> None:
        """Programmatically set the value of a widget by its label.

        In headless mode this is the only way to change slider /
        checkbox / combo values from outside the ``@on_frame`` callback.
        Works in windowed mode too (takes effect on the next frame).

        Args:
            label: The widget label (the same string passed to
                   ``slider``, ``checkbox``, ``combo``, etc.).
            value: The new value.

        Example::

            view = vultorch.View("Sweep", headless=True)
            ctrl = view.panel("ctrl")

            @ctrl.on_frame
            def draw():
                lr = ctrl.slider("LR", 0.0, 1.0, default=0.5)

            for lr in [0.01, 0.001, 0.0001]:
                ctrl.set("LR", lr)        # override slider value
                view.run(max_frames=100)
        """
        # Scan all prefix variants used by widget methods
        for prefix in ("_sf:", "_si:", "_cb:", "_cp:", "_co:", "_it:"):
            key = prefix + label
            if key in self._state:
                self._state[key] = value
                return
        # If the key doesn't exist yet, seed it for every numeric-ish prefix
        # so the next call to the widget will find it.
        if isinstance(value, bool):
            self._state["_cb:" + label] = value
        elif isinstance(value, int):
            self._state["_si:" + label] = value
        elif isinstance(value, float):
            self._state["_sf:" + label] = value
        elif isinstance(value, str):
            self._state["_it:" + label] = value
        elif isinstance(value, tuple):
            self._state["_cp:" + label] = value
        else:
            # Fallback: store with a generic prefix
            self._state["_sf:" + label] = value

    def get(self, label: str, default: Any = None) -> Any:
        """Read the current value of a widget by its label.

        Args:
            label:   The widget label.
            default: Returned if the widget has not been created yet.

        Returns:
            The widget's current value, or *default*.
        """
        for prefix in ("_sf:", "_si:", "_cb:", "_cp:", "_co:", "_it:"):
            key = prefix + label
            if key in self._state:
                return self._state[key]
        return default

    # ── Widgets (with automatic state) ──────────────────────────────

    def text(self, text: str):
        if self._view._headless:
            return
        self._before_widget()
        _ui.text(str(text))

    def text_colored(self, r: float, g: float, b: float, a: float,
                     text: str):
        if self._view._headless:
            return
        self._before_widget()
        _ui.text_colored(r, g, b, a, str(text))

    def text_wrapped(self, text: str):
        if self._view._headless:
            return
        self._before_widget()
        _ui.text_wrapped(str(text))

    def separator(self):
        if self._view._headless:
            return
        _ui.separator()

    def button(self, label: str, width: float = 0, height: float = 0) -> bool:
        if self._view._headless:
            return False
        self._before_widget()
        return _ui.button(label, width, height)

    def checkbox(self, label: str, *, default: bool = False) -> bool:
        key = "_cb:" + label
        if key not in self._state:
            self._state[key] = default
        if self._view._headless:
            return self._state[key]
        self._before_widget()
        self._state[key] = _ui.checkbox(label, self._state[key])
        return self._state[key]

    def slider(self, label: str, min: float = 0.0, max: float = 1.0, *,
               default: Optional[float] = None) -> float:
        key = "_sf:" + label
        if key not in self._state:
            self._state[key] = default if default is not None else min
        if self._view._headless:
            return self._state[key]
        self._before_widget()
        self._state[key] = _ui.slider_float(label, self._state[key], min, max)
        return self._state[key]

    def slider_int(self, label: str, min: int = 0, max: int = 100, *,
                   default: int = 0) -> int:
        key = "_si:" + label
        if key not in self._state:
            self._state[key] = default
        if self._view._headless:
            return self._state[key]
        self._before_widget()
        self._state[key] = _ui.slider_int(label, self._state[key], min, max)
        return self._state[key]

    def color_picker(self, label: str, *,
                     default: Tuple[float, float, float] = (1.0, 1.0, 1.0)
                     ) -> Tuple[float, float, float]:
        key = "_cp:" + label
        if key not in self._state:
            self._state[key] = default
        if self._view._headless:
            return self._state[key]
        self._before_widget()
        c = self._state[key]
        self._state[key] = _ui.color_edit3(label, c[0], c[1], c[2])
        return self._state[key]

    def combo(self, label: str, items: List[str], *,
              default: int = 0) -> int:
        key = "_co:" + label
        if key not in self._state:
            self._state[key] = default
        if self._view._headless:
            return self._state[key]
        self._before_widget()
        self._state[key] = _ui.combo(label, self._state[key], items)
        return self._state[key]

    def input_text(self, label: str, *, default: str = "",
                   max_length: int = 256) -> str:
        key = "_it:" + label
        if key not in self._state:
            self._state[key] = default
        if self._view._headless:
            return self._state[key]
        self._before_widget()
        self._state[key] = _ui.input_text(label, self._state[key], max_length)
        return self._state[key]

    def plot(self, values: Sequence[float], *, label: str = "##plot",
             overlay: str = "", width: float = 0.0, height: float = 80.0):
        if self._view._headless:
            return
        self._before_widget()
        _ui.plot_lines(label, list(values), 0, overlay,
                      3.4028235e+38, 3.4028235e+38, width, height)

    def progress(self, fraction: float, *, overlay: str = ""):
        if self._view._headless:
            return
        self._before_widget()
        _ui.progress_bar(fraction, overlay=overlay)

    def file_dialog(
        self,
        label: str,
        *,
        title: str = "Open File",
        filters: "list[tuple[str, str]] | None" = None,
        initial_dir: "str | None" = None,
    ) -> "str | None":
        """Show a button; when clicked, open a native file-selection dialog.

        Combines :meth:`button` with :func:`vultorch.open_file_dialog` into a
        single convenience widget.  Suitable for use inside an
        ``@panel.on_frame`` callback.

        Args:
            label:       Button label text.
            title:       File dialog window title.
            filters:     List of ``(description, glob_pattern)`` tuples, e.g.
                         ``[("Images", "*.png *.jpg"), ("All files", "*.*")]``.
                         Defaults to common image formats.
            initial_dir: Starting directory for the dialog.

        Returns:
            The selected file path (``str``) when the user picks a file,
            ``None`` when the dialog is cancelled or the button was not clicked.

        Example::

            @controls.on_frame
            def draw():
                path = controls.file_dialog("📂 Open image…")
                if path:
                    canvas.bind(vultorch.imread(path, channels=3))
        """
        from . import open_file_dialog as _open
        if self.button(label):
            return _open(title=title, filters=filters, initial_dir=initial_dir)
        return None

    def save_file_dialog(
        self,
        label: str,
        *,
        title: str = "Save File",
        filters: "list[tuple[str, str]] | None" = None,
        initial_dir: "str | None" = None,
        default_extension: str = ".png",
    ) -> "str | None":
        """Show a button; when clicked, open a native save-file dialog.

        Combines :meth:`button` with :func:`vultorch.save_file_dialog`.

        Args:
            label:              Button label text.
            title:              Dialog window title.
            filters:            File type filters.
            initial_dir:        Starting directory.
            default_extension:  Extension appended when the user omits one.

        Returns:
            The chosen save path (``str``), or ``None`` if cancelled.

        Example::

            @controls.on_frame
            def draw():
                path = controls.save_file_dialog("💾 Save snapshot")
                if path:
                    canvas.save(path)
        """
        from . import save_file_dialog as _save
        if self.button(label):
            return _save(title=title, filters=filters, initial_dir=initial_dir,
                         default_extension=default_extension)
        return None

    def record_button(
        self,
        canvas: Canvas,
        path: str = "recording.gif",
        *,
        fps: int = 30,
        quality: float = 0.8,
        start_label: str = "Record",
        stop_label: str = "Stop Recording",
    ) -> bool:
        """Toggle button that starts / stops recording a canvas.

        When not recording, shows *start_label*.  When recording, shows
        *stop_label* with a red tint.  Returns ``True`` when recording
        is active.

        Args:
            canvas:       The :class:`Canvas` to record.
            path:         Output GIF path (must end with ``.gif``).
            fps:          Recording FPS (default 30).
            quality:      GIF quality 0–1 (default 0.8).  Controls
                          colours per frame; lower → smaller file.
            start_label:  Button text when idle.
            stop_label:   Button text when recording.

        Returns:
            ``True`` while the canvas is recording, ``False`` otherwise.

        Example::

            @ctrl.on_frame
            def draw():
                ctrl.record_button(my_canvas, "demo.gif")
        """
        if canvas.is_saving:
            # Show "Saving..." while background thread writes the GIF
            if not self._view._headless:
                self.text_colored(1.0, 0.8, 0.2, 1.0, "Saving GIF...")
            return False
        if canvas.is_recording:
            # Red-tinted stop button
            if not self._view._headless:
                _ui.push_style_color(21, 0.8, 0.2, 0.2, 1.0)  # ImGuiCol_Button
                _ui.push_style_color(22, 0.9, 0.3, 0.3, 1.0)  # ImGuiCol_ButtonHovered
                _ui.push_style_color(23, 1.0, 0.1, 0.1, 1.0)  # ImGuiCol_ButtonActive
            clicked = self.button(stop_label)
            if not self._view._headless:
                _ui.pop_style_color(3)
            if clicked:
                canvas.stop_recording()
            return canvas.is_recording
        else:
            if self.button(start_label):
                canvas.start_recording(path, fps=fps, quality=quality)
            return canvas.is_recording

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

        In headless mode, canvases are still iterated so that any active
        recorder can capture frames.
        """
        if self._view._headless:
            # Feed recorders even in headless mode
            for c in self._canvases:
                c._feed_recorder()
            return

        n_fit = sum(1 for c in self._canvases if c._fit)
        if n_fit > 1:
            avail_w, avail_h = _ui.get_content_region_avail()
            per_h = max(1.0, avail_h / n_fit)

        for c in self._canvases:
            if c._fit and n_fit > 1:
                _ui.begin_child(f"##cv_{c._name}", 0.0, per_h, 0, 0)
                c._render(fit_height=0.0)
                _ui.end_child()
            else:
                c._render(fit_height=0.0)


# ═══════════════════════════════════════════════════════════════════════
#  Row context managers
# ═══════════════════════════════════════════════════════════════════════

class _RowContext:
    __slots__ = ("_panel",)

    def __init__(self, panel: Panel):
        self._panel = panel

    def __enter__(self):
        self._panel._row_stack.append(True)
        _ui.begin_group()
        return self

    def __exit__(self, *exc):
        _ui.end_group()
        self._panel._row_stack.pop()
        return False


class _HeadlessRowContext:
    """No-op context manager used in headless mode."""
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ═══════════════════════════════════════════════════════════════════════
#  View — the top-level window
# ═══════════════════════════════════════════════════════════════════════

class View:
    """Top-level Vultorch window.

    Example (windowed)::

        view = vultorch.View("Hello", 512, 512)
        view.panel("Viewer").canvas("img").bind(tensor)
        view.run()

    Example (headless — no window / display required)::

        view = vultorch.View("Train", headless=True)
        out = view.panel("Output").canvas("img")

        @view.on_frame
        def train():
            out.bind(model(x))

        view.run(max_frames=500)
        out.save("result.png")
    """

    def __init__(self, title: str = "Vultorch", width: int = 1280,
                 height: int = 720, *, headless: bool = False):
        self._headless = headless
        self._frame_fn = None       # type: Any
        self._panels: list[Panel] = []
        self._panel_map: dict[str, Panel] = {}
        self._first_frame = True
        self._width = width
        self._height = height

        if headless:
            self._win = None        # type: ignore[assignment]
            self._closed = False
            self._frame_count = 0
            self._t0 = _time.perf_counter()
            self._fps_t = self._t0
            self._fps_count = 0
            self._fps = 0.0
        else:
            self._win = _Window(title, width, height)

    # ── Panel factory ───────────────────────────────────────────────

    def panel(self, name: str, *, side: Optional[str] = None,
              width: float = 0.0) -> Panel:
        """Create (or retrieve) a dockable panel.

        Args:
            name:   Panel title (shown in the tab/title bar).
            side:   ``"left"``, ``"right"``, ``"bottom"``, or ``"top"``.
            width:  Sidebar split ratio (e.g. 0.22).  Ignored if *side* is None.
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

    def run(self, *, max_frames: int = 0):
        """Blocking event loop.

        In **windowed** mode: renders all panels and their canvases every
        frame.  If a ``@on_frame`` callback is registered it runs first.

        In **headless** mode: runs callbacks without any rendering.
        *max_frames* limits how many iterations to execute (0 = unlimited).
        When *max_frames* is 0 in headless mode the loop runs until
        :meth:`close` is called from within a callback.

        Args:
            max_frames: Maximum number of frames to execute (0 = unlimited).
                        In windowed mode this is also respected.
        """
        if self._headless:
            self._run_headless(max_frames)
            return

        try:
            frame = 0
            while self._win.poll():
                if 0 < max_frames <= frame:
                    break
                if not self._win.begin_frame():
                    continue

                dockspace_id = _ui.dock_space_over_viewport(flags=8)

                if self._first_frame:
                    self._setup_layout(dockspace_id)
                    self._first_frame = False

                if self._frame_fn is not None:
                    self._frame_fn()

                self._render_all()
                self._win.end_frame()
                frame += 1
        finally:
            self._win.destroy()

    def step(self) -> bool:
        """Process one frame (training-loop integration).

        Returns ``False`` when the window should close (or headless
        iteration is done).  Pair with :meth:`end_step`.
        """
        if self._headless:
            return self._step_headless()

        if not self._win.poll():
            return False
        if not self._win.begin_frame():
            return True

        dockspace_id = _ui.dock_space_over_viewport(flags=8)
        if self._first_frame:
            self._setup_layout(dockspace_id)
            self._first_frame = False

        if self._frame_fn is not None:
            self._frame_fn()

        self._render_all()
        return True

    def end_step(self):
        """Finish the current step frame."""
        if self._headless:
            self._end_step_headless()
            return
        self._win.end_frame()

    def close(self):
        """Explicitly close the window / stop the headless loop."""
        if self._headless:
            self._closed = True
        else:
            self._win.destroy()

    # ── Properties ──────────────────────────────────────────────────

    @property
    def fps(self) -> float:
        """Frames per second.  In headless mode this measures callback throughput."""
        if self._headless:
            return self._fps
        return _ui.get_io_framerate()

    @property
    def time(self) -> float:
        """Seconds since the view was created."""
        if self._headless:
            return _time.perf_counter() - self._t0
        return _ui.get_time()

    @property
    def frame_count(self) -> int:
        """Number of frames executed so far (headless & windowed)."""
        if self._headless:
            return self._frame_count
        try:
            return _ui.get_frame_count()
        except Exception:
            return 0

    @property
    def headless(self) -> bool:
        """``True`` when running without a window."""
        return self._headless

    @property
    def window(self) -> "Any":
        """The underlying :class:`Window`, or ``None`` in headless mode."""
        return self._win

    # ── Headless internals ──────────────────────────────────────────

    def _run_headless(self, max_frames: int):
        """Run callbacks in a tight loop without any rendering."""
        frame = 0
        while not self._closed:
            if 0 < max_frames <= frame:
                break

            self._frame_count += 1
            frame += 1

            # User callback
            if self._frame_fn is not None:
                self._frame_fn()

            # Panel callbacks
            for p in self._panels:
                if p._frame_fn is not None:
                    p._frame_fn()

            # FPS bookkeeping
            self._update_headless_fps()

    def _step_headless(self) -> bool:
        """Headless step(): always returns True unless closed."""
        if self._closed:
            return False

        self._frame_count += 1

        if self._frame_fn is not None:
            self._frame_fn()

        for p in self._panels:
            if p._frame_fn is not None:
                p._frame_fn()
            # Feed canvas recorders even in headless mode
            for c in p._canvases:
                c._feed_recorder()

        return True

    def _end_step_headless(self):
        """Headless end_step(): update FPS counter."""
        self._update_headless_fps()

    def _update_headless_fps(self):
        """Update the headless FPS estimate (every 0.5 s)."""
        self._fps_count += 1
        now = _time.perf_counter()
        dt = now - self._fps_t
        if dt >= 0.5:
            self._fps = self._fps_count / dt
            self._fps_t = now
            self._fps_count = 0

    # ── Windowed internals ──────────────────────────────────────────

    def _render_all(self):
        """Render every panel: open ImGui window, draw canvases, run panel callback."""
        for p in self._panels:
            _ui.begin(p._name, True, 0)
            p._render_canvases()
            if p._frame_fn is not None:
                p._frame_fn()
            _ui.end()

    def _setup_layout(self, dockspace_id: int):
        """Auto-generate docking layout from declared panels."""
        _ui.dock_builder_remove_node(dockspace_id)
        _ui.dock_builder_add_node(dockspace_id, 1 << 10)
        _ui.dock_builder_set_node_size(
            dockspace_id, float(self._width), float(self._height))

        remaining = dockspace_id

        sidebars = [p for p in self._panels if p._side]
        mains = [p for p in self._panels if not p._side]

        _side_dir = {"left": _DIR_LEFT, "right": _DIR_RIGHT,
                     "top": _DIR_UP, "bottom": _DIR_DOWN}
        for p in sidebars:
            direction = _side_dir.get(p._side, _DIR_LEFT)
            ratio = p._width if p._width > 0 else 0.2
            id_side, remaining = _ui.dock_builder_split_node(
                remaining, direction, ratio)
            _ui.dock_builder_dock_window(p._name, id_side)

        n = len(mains)
        if n == 0:
            pass
        elif n == 1:
            _ui.dock_builder_dock_window(mains[0]._name, remaining)
        else:
            current = remaining
            for i in range(n - 1):
                ratio = 1.0 / (n - i)
                id_top, current = _ui.dock_builder_split_node(
                    current, _DIR_UP, ratio)
                _ui.dock_builder_dock_window(mains[i]._name, id_top)
            _ui.dock_builder_dock_window(mains[-1]._name, current)

        _ui.dock_builder_finish(dockspace_id)
