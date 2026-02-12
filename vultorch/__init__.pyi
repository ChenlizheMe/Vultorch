from __future__ import annotations

from typing import Optional, Tuple

import torch

HAS_CUDA: bool
__version__: str


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

    def __init__(self, title: str = "Vultorch", width: int = 1280, height: int = 720) -> None: ...
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
