from __future__ import annotations

from typing import Optional, Tuple

import torch

HAS_CUDA: bool
__version__: str


def set_log_level(level: str) -> None:
    """Set vultorch log verbosity: 'quiet', 'error', 'warn', 'info', 'debug'."""
    ...


class Camera:
    azimuth: float
    elevation: float
    distance: float
    target: Tuple[float, float, float]
    fov: float

    def reset(self) -> None: ...


class Light:
    direction: Tuple[float, float, float]
    color: Tuple[float, float, float]
    intensity: float
    ambient: float
    specular: float
    shininess: float
    enabled: bool


class Window:
    _current: Optional["Window"]

    def __init__(self, title: str = "Vultorch", width: int = 1280, height: int = 720, vsync: bool = True) -> None: ...
    def activate(self) -> None: ...
    def poll(self) -> bool: ...
    def begin_frame(self) -> bool: ...
    def end_frame(self) -> None: ...
    def upload_tensor(self, tensor: torch.Tensor, *, name: str = "tensor") -> None: ...

    @property
    def tensor_texture_id(self) -> int: ...
    def get_texture_id(self, name: str = "tensor") -> int: ...

    @property
    def tensor_size(self) -> Tuple[int, int]: ...
    def get_texture_size(self, name: str = "tensor") -> Tuple[int, int]: ...

    def destroy(self) -> None: ...


def show(
    tensor: torch.Tensor,
    *,
    name: str = "tensor",
    width: float = 0,
    height: float = 0,
    filter: str = "linear",
    window: Optional[Window] = None,
) -> None: ...


def create_tensor(
    height: int,
    width: int,
    channels: int = 4,
    device: str = "cuda:0",
    *,
    name: str = "tensor",
    window: Optional[Window] = None,
) -> torch.Tensor: ...


def imread(
    path: str,
    *,
    channels: int = 4,
    size: Optional[Tuple[int, int]] = None,
    device: str = "cuda",
    shared: bool = False,
    name: str = "tensor",
    window: Optional[Window] = None,
) -> torch.Tensor: ...


def imwrite(
    path: str,
    tensor: torch.Tensor,
    *,
    channels: int = 0,
    size: Optional[Tuple[int, int]] = None,
    quality: int = 95,
) -> None: ...


COLORMAPS: tuple

def colormap(
    tensor: torch.Tensor,
    cmap: str = "turbo",
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> torch.Tensor: ...
"""Apply a colormap LUT to a (H,W) or (H,W,1) scalar tensor.
Returns (H,W,3) float32 on the same device."""


def open_file_dialog(
    title: str = "Open File",
    filters: Optional[list] = None,
    initial_dir: Optional[str] = None,
) -> Optional[str]: ...
"""Open a native file-picker dialog.  Returns the path or None."""


def save_file_dialog(
    title: str = "Save File",
    filters: Optional[list] = None,
    initial_dir: Optional[str] = None,
    default_extension: str = ".png",
) -> Optional[str]: ...
"""Open a native save-file dialog.  Returns the path or None."""


class SceneView:
    name: str
    camera: Camera
    light: Light
    background: Tuple[float, float, float]

    def __init__(self, name: str = "SceneView", width: int = 800, height: int = 600, msaa: int = 4) -> None: ...
    def set_tensor(self, tensor: torch.Tensor) -> None: ...
    def render(self) -> None: ...

    @property
    def msaa(self) -> int: ...
    @msaa.setter
    def msaa(self, value: int) -> None: ...


from . import ui as ui


# ── Declarative API ─────────────────────────────────────────────────

class Canvas:
    def bind(self, tensor: torch.Tensor) -> "Canvas": ...
    def alloc(self, height: int, width: int, channels: int = 4, device: str = "cuda:0") -> torch.Tensor: ...
    @property
    def filter(self) -> str: ...
    @filter.setter
    def filter(self, value: str) -> None: ...
    @property
    def fit(self) -> bool: ...
    @fit.setter
    def fit(self, value: bool) -> None: ...
    def save(self, path: str, *, channels: int = 0, size: Optional[Tuple[int, int]] = None, quality: int = 95) -> None: ...
    def start_recording(self, path: str, *, fps: int = 30, quality: float = 0.8) -> "Canvas": ...
    def stop_recording(self) -> Optional[str]: ...
    @property
    def is_recording(self) -> bool: ...
    @property
    def is_saving(self) -> bool: ...


class Panel:
    def canvas(self, name: str, *, filter: str = "linear", fit: bool = True) -> Canvas: ...
    def on_frame(self, fn: object) -> object: ...
    def row(self) -> object: ...
    def text(self, text: str) -> None: ...
    def text_colored(self, r: float, g: float, b: float, a: float, text: str) -> None: ...
    def text_wrapped(self, text: str) -> None: ...
    def separator(self) -> None: ...
    def button(self, label: str, width: float = 0, height: float = 0) -> bool: ...
    def checkbox(self, label: str, *, default: bool = False) -> bool: ...
    def slider(self, label: str, min: float = 0.0, max: float = 1.0, *, default: Optional[float] = None) -> float: ...
    def slider_int(self, label: str, min: int = 0, max: int = 100, *, default: int = 0) -> int: ...
    def color_picker(self, label: str, *, default: Tuple[float, float, float] = ...) -> Tuple[float, float, float]: ...
    def combo(self, label: str, items: list, *, default: int = 0) -> int: ...
    def input_text(self, label: str, *, default: str = "", max_length: int = 256) -> str: ...
    def plot(self, values: object, *, label: str = "##plot", overlay: str = "", width: float = 0.0, height: float = 80.0) -> None: ...
    def progress(self, fraction: float, *, overlay: str = "") -> None: ...
    def set(self, label: str, value: object) -> None: ...
    def get(self, label: str, default: object = None) -> object: ...
    def file_dialog(
        self,
        label: str,
        *,
        title: str = "Open File",
        filters: Optional[list] = None,
        initial_dir: Optional[str] = None,
    ) -> Optional[str]: ...
    def save_file_dialog(
        self,
        label: str,
        *,
        title: str = "Save File",
        filters: Optional[list] = None,
        initial_dir: Optional[str] = None,
        default_extension: str = ".png",
    ) -> Optional[str]: ...
    def record_button(
        self,
        canvas: Canvas,
        path: str = "recording.gif",
        *,
        fps: int = 30,
        quality: float = 0.8,
        start_label: str = "Record",
        stop_label: str = "Stop Recording",
    ) -> bool: ...


class View:
    def __init__(self, title: str = "Vultorch", width: int = 1280, height: int = 720, *, headless: bool = False) -> None: ...
    def panel(self, name: str, *, side: Optional[str] = None, width: float = 0.0) -> Panel: ...
    def on_frame(self, fn: object) -> object: ...
    def run(self, *, max_frames: int = 0) -> None: ...
    def step(self) -> bool: ...
    def end_step(self) -> None: ...
    def close(self) -> None: ...

    @property
    def fps(self) -> float: ...
    @property
    def time(self) -> float: ...
    @property
    def frame_count(self) -> int: ...
    @property
    def headless(self) -> bool: ...
    @property
    def window(self) -> Optional[Window]: ...
