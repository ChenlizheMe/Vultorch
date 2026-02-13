"""Vultorch – Vulkan + PyTorch zero-copy interop with ImGui.

High-level API:
    vultorch.show(tensor)        — display a 2D/3D tensor in ImGui (one line!)
    vultorch.create_tensor(...)  — allocate a GPU-shared tensor (true zero-copy)
    vultorch.SceneView(...)      — 3D plane viewer with MSAA, lighting, orbit camera
"""

from ._vultorch import Engine
from ._vultorch import ui  # noqa: F401  (re-export)

# Re-export CUDA availability flag
try:
    from ._vultorch import HAS_CUDA
except ImportError:
    HAS_CUDA = False

__version__ = "0.3.2"


# ═══════════════════════════════════════════════════════════════════════
#  Camera / Light  (Python-side data holders)
# ═══════════════════════════════════════════════════════════════════════
class Camera:
    """Orbit camera parameters."""

    __slots__ = ("azimuth", "elevation", "distance", "target", "fov")

    def __init__(self):
        self.azimuth: float = 0.0
        self.elevation: float = 0.6
        self.distance: float = 3.0
        self.target: tuple = (0.0, 0.0, 0.0)
        self.fov: float = 45.0

    def reset(self):
        self.__init__()

    def _push(self, engine: Engine):
        engine.scene_set_camera(
            self.azimuth, self.elevation, self.distance,
            *self.target, self.fov)

    def _pull(self, engine: Engine):
        az, el, dist, tx, ty, tz, fov = engine.scene_get_camera()
        self.azimuth   = az
        self.elevation = el
        self.distance  = dist
        self.target    = (tx, ty, tz)
        self.fov       = fov


class Light:
    """Blinn-Phong directional light."""

    __slots__ = ("direction", "color", "intensity", "ambient",
                 "specular", "shininess", "enabled")

    def __init__(self):
        self.direction: tuple = (0.3, -1.0, 0.5)
        self.color: tuple = (1.0, 1.0, 1.0)
        self.intensity: float = 1.0
        self.ambient: float = 0.15
        self.specular: float = 0.5
        self.shininess: float = 32.0
        self.enabled: bool = True

    def _push(self, engine: Engine):
        engine.scene_set_light(
            *self.direction,
            self.intensity,
            *self.color,
            self.ambient, self.specular, self.shininess,
            self.enabled)


# ═══════════════════════════════════════════════════════════════════════
#  Window — high-level wrapper around Engine
# ═══════════════════════════════════════════════════════════════════════
class Window:
    """High-level wrapper around the Vultorch engine.

    Usage::

        import vultorch
        from vultorch import ui

        win = vultorch.Window("Demo", 1280, 720)
        while win.poll():
            if not win.begin_frame():
                continue
            ui.begin("Panel")
            vultorch.show(tensor)
            ui.end()
            win.end_frame()
        win.destroy()
    """

    # Singleton reference (for module-level show() helper)
    _current: "Window | None" = None

    def __init__(self, title: str = "Vultorch", width: int = 1280, height: int = 720):
        self._engine = Engine()
        self._engine.init(title, width, height)
        self._rgba_bufs = {}   # cached RGBA buffers per name for show()
        Window._current = self

    # -- frame loop ----------------------------------------------------------

    def poll(self) -> bool:
        """Process events.  Returns *False* when the window should close."""
        return self._engine.poll()

    def begin_frame(self) -> bool:
        """Begin a new frame.  Returns *False* if the frame was skipped."""
        return self._engine.begin_frame()

    def end_frame(self) -> None:
        """Finish the current frame and present."""
        self._engine.end_frame()

    def activate(self) -> None:
        """Make this window the current target for module-level helpers."""
        Window._current = self

    # -- tensor display (requires CUDA) --------------------------------------

    def upload_tensor(self, tensor, *, name: str = "tensor") -> None:
        """Upload a CUDA PyTorch tensor for display.

        Supports shape ``(H, W)``, ``(H, W, C)`` with C = 1, 3, or 4.
        dtype must be ``torch.float32``, ``torch.float16``, or ``torch.uint8``.
        """
        tensor, h, w, channels = _normalize_tensor(tensor)
        self._engine.upload_tensor(name, tensor.data_ptr(), w, h, channels,
                                   tensor.device.index or 0)

    @property
    def tensor_texture_id(self) -> int:
        """ImGui texture ID of the last uploaded tensor (0 if none)."""
        if not HAS_CUDA:
            return 0
        return self._engine.tensor_texture_id("tensor")

    def get_texture_id(self, name: str = "tensor") -> int:
        """ImGui texture ID of a named tensor (0 if none)."""
        if not HAS_CUDA:
            return 0
        return self._engine.tensor_texture_id(name)

    @property
    def tensor_size(self) -> tuple:
        """(width, height) of the last uploaded tensor."""
        if not HAS_CUDA:
            return (0, 0)
        return (self._engine.tensor_width("tensor"), self._engine.tensor_height("tensor"))

    def get_texture_size(self, name: str = "tensor") -> tuple:
        """(width, height) of a named tensor."""
        if not HAS_CUDA:
            return (0, 0)
        return (self._engine.tensor_width(name), self._engine.tensor_height(name))

    # -- lifecycle -----------------------------------------------------------

    def destroy(self) -> None:
        """Explicitly release all GPU / window resources."""
        self._engine.destroy()
        if Window._current is self:
            Window._current = None

    def __del__(self) -> None:
        try:
            self._engine.destroy()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════
#  show() — one-line tensor display
# ═══════════════════════════════════════════════════════════════════════
def show(tensor, *, name: str = "tensor", width: float = 0,
         height: float = 0, filter: str = "linear", window: "Window | None" = None) -> None:
    """Display a CUDA tensor in the current ImGui context.

    Args:
        tensor:  torch.Tensor on CUDA, float32/float16/uint8, shape (H,W) or (H,W,C).
        name:    Unique label (for caching when showing multiple tensors).
        width:   Display width in pixels (0 = auto-fit to tensor).
        height:  Display height in pixels (0 = auto-fit to tensor).
        filter:  ``"nearest"`` or ``"linear"``.
        window:  Target window (defaults to the current window).
    """
    win = window or Window._current
    if win is None:
        raise RuntimeError("No active vultorch.Window — create one first")
    if not HAS_CUDA:
        ui.text("[Vultorch: CUDA not available]")
        return

    tensor, h, w, c = _normalize_tensor(tensor)

    if c != 4:
        # RGBA expansion using a cached buffer (avoids per-frame allocation)
        import torch
        rgba = win._rgba_bufs.get(name)
        if (rgba is None or rgba.shape[0] != h or rgba.shape[1] != w
                or rgba.device != tensor.device):
            rgba = torch.empty(h, w, 4, dtype=torch.float32, device=tensor.device)
            win._rgba_bufs[name] = rgba
        if c == 1:
            src = tensor.view(h, w, 1) if tensor.ndim == 2 else tensor
            rgba[:, :, 0:3] = src.expand(h, w, 3)
        else:  # c == 3
            rgba[:, :, 0:3] = tensor
        rgba[:, :, 3] = 1.0
        tensor = rgba
        c = 4

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # Set filter mode
    fmode = 0 if filter == "nearest" else 1
    win._engine.set_tensor_filter(name, fmode)

    # Upload (GPU-GPU or zero-copy if shared)
    win._engine.upload_tensor(name, tensor.data_ptr(), w, h, c,
                              tensor.device.index or 0)

    # Auto-size
    tex_id = win._engine.tensor_texture_id(name)
    tw = win._engine.tensor_width(name)
    th = win._engine.tensor_height(name)
    if tex_id:
        dw = width  if width  > 0 else float(tw)
        dh = height if height > 0 else float(th)
        ui.image(tex_id, dw, dh)


# ═══════════════════════════════════════════════════════════════════════
#  create_tensor() — allocate GPU-shared tensor (true zero-copy)
# ═══════════════════════════════════════════════════════════════════════
def create_tensor(height: int, width: int, channels: int = 4,
                  device: str = "cuda:0", *, name: str = "tensor",
                  window: "Window | None" = None):
    """Allocate a Vulkan-shared CUDA tensor for true zero-copy display.

    The returned object is a standard ``torch.Tensor`` on the given CUDA
    device.  Any writes to it are immediately visible to Vulkan — call
    ``show(tensor)`` each frame; no GPU-GPU memcpy is needed.

    .. note::

        Currently only ``channels=4`` gives true zero-copy.  For 1 or 3
        channels a regular tensor is returned and upload handles expansion.
        Zero-copy allocation always uses float32.

    Args:
        height:    Tensor height.
        width:     Tensor width.
        channels:  1, 3, or 4.
        device:    CUDA device string, e.g. ``"cuda:0"``.
        name:      Texture slot name (must match ``show(..., name=...)``).
        window:    Target window (defaults to the current window).

    Returns:
        torch.Tensor of shape ``(height, width, channels)`` on CUDA.
    """
    win = window or Window._current
    if win is None:
        raise RuntimeError("No active vultorch.Window — create one first")
    if not HAS_CUDA:
        raise RuntimeError("Vultorch was built without CUDA support")

    import torch

    dev_idx = 0
    if ":" in device:
        dev_idx = int(device.split(":")[1])

    if channels == 4:
        try:
            # Allocate Vulkan-shared memory and wrap via __cuda_array_interface__
            ptr = win._engine.allocate_shared_tensor(
                name, width, height, channels, dev_idx)
            if ptr == 0:
                raise RuntimeError("allocate_shared_tensor returned null")

            class _CUDAMem:
                """Tiny wrapper exposing __cuda_array_interface__ for torch.as_tensor."""
                __slots__ = ("__cuda_array_interface__",)
                def __init__(self, p, shape):
                    self.__cuda_array_interface__ = {
                        "shape": shape,
                        "typestr": "<f4",
                        "data": (p, False),
                        "version": 3,
                        "strides": None,
                    }

            result = torch.as_tensor(
                _CUDAMem(ptr, (height, width, channels)),
                device=f"cuda:{dev_idx}"
            )
            # Some PyTorch builds leave a stale ValueError on the error
            # indicator after internal DLPack capsule-name probing.
            # Clear it so the caller does not see a spurious SystemError.
            import ctypes
            ctypes.pythonapi.PyErr_Clear()
            return result
        except Exception as exc:
            import warnings
            warnings.warn(
                f"[vultorch] Failed to allocate shared GPU memory "
                f"({exc!r}); falling back to regular CUDA tensor "
                f"(GPU-GPU copy will be used instead of zero-copy)",
                RuntimeWarning,
                stacklevel=2,
            )
            # Clear any residual error state left by the failed attempt
            import ctypes
            ctypes.pythonapi.PyErr_Clear()

    # Non-4ch or shared-alloc failed: return a regular CUDA tensor.
    # show() / upload_tensor() will pad to RGBA and do GPU-GPU copy.
    return torch.zeros(height, width, channels, dtype=torch.float32,
                       device=f"cuda:{dev_idx}")


# ═══════════════════════════════════════════════════════════════════════
#  SceneView — 3D tensor viewer widget
# ═══════════════════════════════════════════════════════════════════════
class SceneView:
    """3D tensor viewer — renders a tensor on a plane in 3D space
    with Blinn-Phong lighting, orbit camera, and MSAA.

    Usage::

        scene = vultorch.SceneView("3D View", 800, 600, msaa=4)
        # in frame loop:
        scene.set_tensor(tensor)
        scene.render()
    """

    def __init__(self, name: str = "SceneView", width: int = 800,
                 height: int = 600, msaa: int = 4):
        self.name = name
        self._width = width
        self._height = height
        self._msaa = msaa
        self._initialized = False

        self._texture_name = f"scene:{id(self)}"

        self.camera = Camera()
        self.light = Light()
        self.background: tuple = (0.12, 0.12, 0.14)

        self._prev_mouse = None
        self._rgba_buf = None   # cached RGBA buffer for set_tensor()

    def _ensure_init(self):
        win = Window._current
        if win is None:
            raise RuntimeError("No active vultorch.Window")
        if not self._initialized:
            win._engine.init_scene(self._width, self._height, self._msaa)
            self.camera._push(win._engine)
            self.light._push(win._engine)
            win._engine.scene_set_background(*self.background)
            self._initialized = True

    def set_tensor(self, tensor) -> None:
        """Upload tensor to the scene's texture."""
        win = Window._current
        if win is None:
            raise RuntimeError("No active vultorch.Window")
        self._ensure_init()

        tensor, h, w, c = _normalize_tensor(tensor)

        if c != 4:
            # RGBA expansion using cached buffer (avoids per-frame allocation)
            import torch
            rgba = self._rgba_buf
            if (rgba is None or rgba.shape[0] != h or rgba.shape[1] != w
                    or rgba.device != tensor.device):
                rgba = torch.empty(h, w, 4, dtype=torch.float32, device=tensor.device)
                self._rgba_buf = rgba
            if c == 1:
                src = tensor.view(h, w, 1) if tensor.ndim == 2 else tensor
                rgba[:, :, 0:3] = src.expand(h, w, 3)
            elif c == 3:
                rgba[:, :, 0:3] = tensor
            else:
                raise ValueError(f"Expected 1, 3, or 4 channels, got {c}")
            rgba[:, :, 3] = 1.0
            tensor = rgba

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        h, w, c = tensor.shape

        win._engine.upload_tensor(self._texture_name, tensor.data_ptr(), w, h, c,
                                  tensor.device.index or 0)

    def render(self) -> None:
        """Process interaction, render the 3D scene, and display in ImGui."""
        win = Window._current
        if win is None:
            return
        self._ensure_init()

        # Push Python-side parameters to C++
        self.camera._push(win._engine)
        self.light._push(win._engine)
        win._engine.scene_set_background(*self.background)

        # ── Resize to fit available space ─────────────────────────
        avail = ui.get_content_region_avail()
        sw = min(self._width, int(avail[0])) if avail[0] > 0 else self._width
        sh = min(self._height, int(avail[1])) if avail[1] > 0 else self._height

        if sw > 0 and sh > 0:
            cur_w = win._engine.scene_width()
            cur_h = win._engine.scene_height()
            if cur_w != sw or cur_h != sh:
                win._engine.scene_resize(sw, sh)

        # Render offscreen
        win._engine.scene_render(self._texture_name)

        # Display result as ImGui image
        tex_id = win._engine.scene_texture_id()
        if tex_id:
            ui.image(tex_id, float(sw), float(sh))

            # ── Mouse interaction ────────────────────────────────
            if ui.is_item_hovered():
                mx, my = ui.get_mouse_pos()
                if self._prev_mouse is not None:
                    dx = mx - self._prev_mouse[0]
                    dy = my - self._prev_mouse[1]
                else:
                    dx = dy = 0.0

                left = ui.is_mouse_dragging(0)
                right = ui.is_mouse_dragging(1)
                middle = ui.is_mouse_dragging(2)

                if left or right or middle:
                    win._engine.scene_process_input(
                        dx, dy, 0.0, left, right, middle)

                self._prev_mouse = (mx, my)
            else:
                self._prev_mouse = None

        # Pull back camera state
        self.camera._pull(win._engine)

    @property
    def msaa(self) -> int:
        return self._msaa

    @msaa.setter
    def msaa(self, value: int):
        self._msaa = value
        if self._initialized:
            win = Window._current
            if win:
                win._engine.scene_set_msaa(value)


# ═══════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════
def _normalize_tensor(tensor):
    """Normalize tensor dtype/shape for display.

    Accepts float32/float16/uint8 CUDA tensors and returns float32.
    """
    if not HAS_CUDA:
        raise RuntimeError("Vultorch was built without CUDA support")

    import torch

    if not tensor.is_cuda:
        raise ValueError("Tensor must be on a CUDA device")

    if tensor.dtype == torch.uint8:
        tensor = tensor.float().div(255.0)
    elif tensor.dtype == torch.float16:
        tensor = tensor.float()
    elif tensor.dtype != torch.float32:
        raise ValueError("Tensor dtype must be float32, float16, or uint8")

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    if tensor.ndim == 2:
        h, w = tensor.shape
        c = 1
    elif tensor.ndim == 3:
        h, w, c = tensor.shape
        if c not in (1, 3, 4):
            raise ValueError(f"Expected 1, 3, or 4 channels, got {c}")
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {tensor.shape}")

    return tensor, h, w, c


# ═══════════════════════════════════════════════════════════════════════
#  Declarative API  (import last to avoid circular deps)
# ═══════════════════════════════════════════════════════════════════════
from .app import View, Canvas, Panel  # noqa: E402, F401
