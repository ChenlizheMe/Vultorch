"""Vultorch – Vulkan + PyTorch visualization with ImGui.

High-level API:
    vultorch.show(tensor)                  — display a tensor in ImGui (CUDA or CPU)
    vultorch.create_tensor(...)            — allocate a shared tensor (zero-copy on CUDA)
    vultorch.SceneView(...)                — 3D plane viewer with MSAA, lighting, orbit camera
    vultorch.imread(path)                  — load an image file as a torch.Tensor
    vultorch.imwrite(path, t)              — save a torch.Tensor to an image file
    vultorch.colormap(tensor, cmap=...)    — apply a colormap LUT to a scalar tensor
    vultorch.open_file_dialog(...)         — open a native file-picker dialog

Recording:
    canvas.start_recording("out.gif")      — record canvas to GIF
    canvas.stop_recording()                — finalize and save
    panel.record_button(canvas, "out.gif") — toggle record/stop button
"""

from ._vultorch import Engine
from ._vultorch import ui  # noqa: F401  (re-export)

# Re-export CUDA availability flag
try:
    from ._vultorch import HAS_CUDA
except ImportError:
    HAS_CUDA = False

# Re-export log level control
try:
    from ._vultorch import set_log_level
except ImportError:
    def set_log_level(level: str) -> None:
        """Stub for set_log_level when native module unavailable."""
        pass

__version__ = "0.5.1"


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

    def __init__(self, title: str = "Vultorch", width: int = 1280, height: int = 720, vsync: bool = True):
        self._engine = Engine()
        self._engine.init(title, width, height, vsync)
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
        """Upload a PyTorch tensor for display.

        Supports CUDA and CPU tensors.
        Supports shape ``(H, W)``, ``(H, W, C)`` with C = 1, 3, or 4.
        dtype must be ``torch.float32``, ``torch.float16``, or ``torch.uint8``.
        """
        tensor, h, w, channels = _normalize_tensor(tensor)
        if tensor.is_cuda and HAS_CUDA:
            self._engine.upload_tensor(name, tensor.data_ptr(), w, h, channels,
                                       tensor.device.index or 0)
        else:
            if tensor.is_cuda:
                tensor = tensor.cpu()
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
            self._engine.upload_tensor_cpu(name, tensor.data_ptr(), w, h, channels)

    @property
    def tensor_texture_id(self) -> int:
        """ImGui texture ID of the last uploaded tensor (0 if none)."""
        return self._engine.tensor_texture_id("tensor")

    def get_texture_id(self, name: str = "tensor") -> int:
        """ImGui texture ID of a named tensor (0 if none)."""
        return self._engine.tensor_texture_id(name)

    @property
    def tensor_size(self) -> tuple:
        """(width, height) of the last uploaded tensor."""
        return (self._engine.tensor_width("tensor"), self._engine.tensor_height("tensor"))

    def get_texture_size(self, name: str = "tensor") -> tuple:
        """(width, height) of a named tensor."""
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
    """Display a tensor in the current ImGui context.

    Supports both CUDA and CPU tensors.  When built with CUDA and the
    tensor is on GPU, a zero-copy / GPU-GPU path is used.  CPU tensors
    use a host-visible staging buffer.

    Args:
        tensor:  torch.Tensor (CUDA or CPU), float32/float16/uint8, shape (H,W) or (H,W,C).
        name:    Unique label (for caching when showing multiple tensors).
        width:   Display width in pixels (0 = auto-fit to tensor).
        height:  Display height in pixels (0 = auto-fit to tensor).
        filter:  ``"nearest"`` or ``"linear"``.
        window:  Target window (defaults to the current window).
    """
    win = window or Window._current
    if win is None:
        raise RuntimeError("No active vultorch.Window — create one first")

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

    # Upload: choose CUDA (GPU-GPU) or CPU (host memcpy) path
    if tensor.is_cuda and HAS_CUDA:
        win._engine.upload_tensor(name, tensor.data_ptr(), w, h, c,
                                  tensor.device.index or 0)
    else:
        if tensor.is_cuda:
            import warnings
            warnings.warn(
                "Vultorch built without CUDA; moving tensor to CPU for display",
                RuntimeWarning, stacklevel=2)
            tensor = tensor.cpu()
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
        win._engine.upload_tensor_cpu(name, tensor.data_ptr(), w, h, c)

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

    import torch

    # ── CPU-only mode (no CUDA at all) ──────────────────────────
    if not HAS_CUDA or device == "cpu":
        return torch.zeros(height, width, channels, dtype=torch.float32,
                           device="cpu")

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

        if tensor.is_cuda and HAS_CUDA:
            win._engine.upload_tensor(self._texture_name, tensor.data_ptr(), w, h, c,
                                      tensor.device.index or 0)
        else:
            if tensor.is_cuda:
                tensor = tensor.cpu()
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
            win._engine.upload_tensor_cpu(self._texture_name,
                                          tensor.data_ptr(), w, h, c)

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
#  imread / imwrite — image I/O via stb_image (no PIL dependency)
# ═══════════════════════════════════════════════════════════════════════
def imread(path: str, *, channels: int = 4,
           size: "tuple[int, int] | None" = None,
           device: str = "cuda", shared: bool = False,
           name: str = "tensor",
           window: "Window | None" = None):
    """Load an image from disk as a ``torch.Tensor``.

    Uses stb_image internally — supports PNG, JPEG, BMP, TGA, HDR, PSD,
    and GIF (first frame).

    Args:
        path:     File path (str or ``pathlib.Path``).
        channels: Desired output channels (1 grey, 3 RGB, 4 RGBA, 0 = auto).
        size:     Optional ``(height, width)`` to resize the loaded image
                  using bilinear interpolation.  ``None`` keeps original size.
        device:   ``"cuda"`` / ``"cuda:N"`` / ``"cpu"``.
        shared:   If ``True`` and channels == 4, allocate via Vulkan shared
                  memory for zero-copy display (calls ``create_tensor``).
        name:     Texture slot name (only relevant when *shared* is True).
        window:   Target window (only relevant when *shared* is True).

    Returns:
        ``torch.Tensor`` of shape ``(H, W, C)`` with dtype ``float32``,
        values in [0, 1].
    """
    import torch
    import torch.nn.functional as F
    from ._vultorch import _imread

    raw_bytes, w, h, c = _imread(str(path), channels)

    # Build a CPU tensor from the raw buffer
    cpu_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.float32)
    cpu_tensor = cpu_tensor.reshape(h, w, c)

    # Resize if requested
    if size is not None:
        # F.interpolate expects (N, C, H, W)
        cpu_tensor = F.interpolate(
            cpu_tensor.permute(2, 0, 1).unsqueeze(0),
            size=size, mode="bilinear", align_corners=False,
        ).squeeze(0).permute(1, 2, 0).contiguous()
        h, w = size

    if shared and c == 4:
        t = create_tensor(h, w, c, device, name=name, window=window)
        t.copy_(cpu_tensor.to(t.device))
        return t

    if device == "cpu":
        return cpu_tensor.clone()  # clone to own memory

    return cpu_tensor.to(device)


def imwrite(path: str, tensor, *, channels: int = 0,
            size: "tuple[int, int] | None" = None,
            quality: int = 95) -> None:
    """Save a ``torch.Tensor`` to an image file on disk.

    Uses stb_image_write — supports PNG, JPEG, BMP, TGA, HDR.

    Args:
        path:     Output file path (str or ``pathlib.Path``).
        tensor:   ``torch.Tensor`` (CUDA or CPU), float32/float16/uint8,
                  shape ``(H, W)`` or ``(H, W, C)`` with C = 1, 3, or 4.
        channels: Number of channels to write (0 = same as tensor).
                  When *channels* < tensor channels, extra channels are
                  dropped.  When *channels* > tensor channels, the tensor
                  is expanded (grey → RGB, RGB → RGBA with alpha = 1).
        size:     Optional ``(height, width)`` to resize before saving.
                  ``None`` keeps original size.
        quality:  JPEG quality (1–100, default 95).
    """
    import torch
    import torch.nn.functional as F
    from ._vultorch import _imwrite

    # Normalize to float32 CPU contiguous (H, W, C)
    tensor, h, w, c = _normalize_tensor(tensor)

    if tensor.is_cuda:
        tensor = tensor.cpu()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # Resize if requested
    if size is not None:
        tensor = F.interpolate(
            tensor.permute(2, 0, 1).unsqueeze(0),
            size=size, mode="bilinear", align_corners=False,
        ).squeeze(0).permute(1, 2, 0).contiguous()
        h, w = size

    out_channels = channels if channels > 0 else c

    # Channel adaptation
    if out_channels != c:
        if out_channels > c:
            # Expand: 1→3 (replicate), 3→4 (add alpha)
            if c == 1 and out_channels >= 3:
                tensor = tensor.expand(h, w, 3).contiguous()
                c = 3
            if c == 3 and out_channels == 4:
                alpha = torch.ones(h, w, 1, dtype=torch.float32)
                tensor = torch.cat([tensor, alpha], dim=-1).contiguous()
                c = 4
        else:
            # Drop extra channels
            tensor = tensor[:, :, :out_channels].contiguous()
            c = out_channels

    _imwrite(str(path), tensor.data_ptr(), w, h, c, quality)


# ═══════════════════════════════════════════════════════════════════════
#  colormap() — apply a LUT colormap to a scalar tensor
# ═══════════════════════════════════════════════════════════════════════

# Anchor colors for each colormap (RGB float tuples, evenly spaced from 0→1).
# These are interpolated to a 256-entry LUT on first use.
_CMAP_ANCHORS: "dict[str, list[tuple[float,float,float]]]" = {
    # ── Turbo (Google, 2019) ─────────────────────────────────────────
    "turbo": [
        (.190,.072,.232),(.225,.164,.451),(.251,.252,.634),(.268,.338,.784),
        (.276,.421,.899),(.275,.501,.973),(.259,.580,1.00),(.214,.659,.980),
        (.158,.736,.923),(.112,.806,.839),(.093,.866,.737),(.120,.912,.624),
        (.210,.945,.504),(.362,.964,.382),(.539,.967,.265),(.708,.956,.160),
        (.849,.932,.078),(.946,.890,.022),(.993,.826,.008),(.996,.738,.006),
        (.977,.644,.040),(.943,.549,.092),(.894,.458,.134),(.836,.374,.160),
        (.768,.299,.168),(.697,.232,.163),(.622,.174,.148),(.547,.126,.129),
        (.473,.087,.109),(.402,.058,.090),(.335,.037,.074),(.270,.015,.051),
    ],
    # ── Viridis (matplotlib default) ─────────────────────────────────
    "viridis": [
        (.267,.005,.329),(.281,.094,.414),(.277,.185,.490),(.254,.272,.530),
        (.225,.351,.551),(.192,.425,.556),(.163,.492,.557),(.135,.556,.547),
        (.122,.619,.529),(.155,.683,.499),(.267,.747,.441),(.416,.804,.360),
        (.569,.852,.263),(.717,.879,.151),(.856,.892,.063),(.993,.906,.144),
    ],
    # ── Magma (matplotlib/seaborn) ────────────────────────────────────
    "magma": [
        (.001,.000,.014),(.042,.013,.087),(.102,.024,.178),(.175,.034,.272),
        (.257,.045,.361),(.348,.060,.424),(.440,.079,.444),(.534,.098,.441),
        (.626,.120,.416),(.715,.149,.373),(.800,.198,.313),(.874,.270,.261),
        (.929,.373,.267),(.961,.498,.350),(.981,.641,.481),(.993,.798,.656),
        (.988,.962,.856),
    ],
    # ── Plasma (matplotlib) ───────────────────────────────────────────
    "plasma": [
        (.050,.030,.528),(.167,.021,.590),(.277,.014,.626),(.381,.018,.629),
        (.477,.060,.610),(.562,.130,.570),(.637,.198,.516),(.703,.261,.455),
        (.761,.322,.390),(.816,.384,.326),(.868,.451,.264),(.916,.527,.210),
        (.955,.614,.176),(.982,.715,.166),(.993,.823,.215),(.975,.942,.329),
    ],
    # ── Inferno (matplotlib) ──────────────────────────────────────────
    "inferno": [
        (.001,.000,.014),(.040,.012,.094),(.099,.023,.193),(.174,.035,.302),
        (.261,.051,.403),(.356,.066,.453),(.451,.080,.451),(.546,.099,.428),
        (.644,.124,.388),(.738,.161,.333),(.822,.222,.279),(.893,.313,.240),
        (.945,.437,.220),(.977,.582,.228),(.991,.738,.310),(.988,.899,.498),
        (.988,.998,.645),
    ],
    # ── Jet (classic rainbow) ─────────────────────────────────────────
    "jet": [
        (.000,.000,.500),(.000,.000,1.00),(.000,.500,1.00),(.000,1.00,1.00),
        (.500,1.00,.500),(1.00,1.00,.000),(1.00,.500,.000),(1.00,.000,.000),
        (.500,.000,.000),
    ],
    # ── Grayscale ─────────────────────────────────────────────────────
    "gray":  [(.000,.000,.000),(1.00,1.00,1.00)],
    "grey":  [(.000,.000,.000),(1.00,1.00,1.00)],
    # ── Hot (black→red→yellow→white) ──────────────────────────────────
    "hot":   [(.000,.000,.000),(.500,.000,.000),(1.00,.000,.000),
              (1.00,.500,.000),(1.00,1.00,.000),(1.00,1.00,.500),(1.00,1.00,1.00)],
    # ── Cool (cyan→magenta) ───────────────────────────────────────────
    "cool":  [(.000,1.00,1.00),(1.00,.000,1.00)],
}

#: All available colormap names.
COLORMAPS: "tuple[str, ...]" = tuple(sorted(_CMAP_ANCHORS))

# Device-keyed LUT cache: (name, device_str) → (256, 3) float32 tensor
_cmap_lut_cache: "dict[tuple[str, str], object]" = {}


def _get_cmap_lut(name: str, device):
    """Return a (256, 3) float32 LUT tensor for the given colormap on *device*."""
    import torch
    import torch.nn.functional as F

    key = (name, str(device))
    if key not in _cmap_lut_cache:
        anchors = _CMAP_ANCHORS[name]
        k = torch.tensor(anchors, dtype=torch.float32)  # (N, 3)
        lut = F.interpolate(
            k.T.unsqueeze(0),          # (1, 3, N)
            size=256,
            mode="linear",
            align_corners=True,
        )                              # (1, 3, 256)
        _cmap_lut_cache[key] = lut.squeeze(0).T.contiguous().to(device)  # (256, 3)
    return _cmap_lut_cache[key]


def colormap(tensor, cmap: str = "turbo", *,
             vmin: "float | None" = None,
             vmax: "float | None" = None):
    """Apply a colormap LUT to a scalar tensor.

    Maps single-channel (scalar) data to an RGB color image using one of the
    built-in perceptually-uniform or classic colormaps.

    Args:
        tensor: ``torch.Tensor`` of shape ``(H, W)`` or ``(H, W, 1)``.
                Can be CUDA or CPU, float32/float16/uint8.
        cmap:   Colormap name.  One of ``vultorch.COLORMAPS``.
                Defaults to ``'turbo'``.
        vmin:   Value mapped to the bottom of the colormap.
                Defaults to ``tensor.min()``.
        vmax:   Value mapped to the top of the colormap.
                Defaults to ``tensor.max()``.

    Returns:
        ``torch.Tensor`` of shape ``(H, W, 3)`` float32, values in [0, 1],
        on the same device as *tensor*.

    Example::

        depth = model.predict_depth(img)     # (H, W)
        rgb   = vultorch.colormap(depth, cmap='viridis')
        panel.canvas("depth").bind(rgb)
    """
    import torch

    name = cmap.lower()
    if name not in _CMAP_ANCHORS:
        raise ValueError(
            f"Unknown colormap '{cmap}'. "
            f"Available: {sorted(_CMAP_ANCHORS)}"
        )

    # Normalise input dtype
    t = tensor
    if t.dtype == torch.uint8:
        t = t.float().div(255.0)
    elif t.dtype == torch.float16:
        t = t.float()
    elif t.dtype != torch.float32:
        t = t.float()

    # Accept (H,W,1) → (H,W)
    if t.ndim == 3:
        if t.shape[2] != 1:
            raise ValueError(
                f"colormap expects a single-channel tensor, got shape {tuple(t.shape)}"
            )
        t = t.squeeze(-1)
    elif t.ndim != 2:
        raise ValueError(
            f"colormap expects shape (H,W) or (H,W,1), got {tuple(t.shape)}"
        )

    # Normalise to [0, 1]
    if vmin is None:
        lo = t.min()
    else:
        lo = torch.tensor(float(vmin), dtype=torch.float32, device=t.device)
    if vmax is None:
        hi = t.max()
    else:
        hi = torch.tensor(float(vmax), dtype=torch.float32, device=t.device)

    rng = (hi - lo).clamp(min=1e-8)
    t = (t - lo) / rng

    lut = _get_cmap_lut(name, t.device)              # (256, 3)
    indices = (t.clamp(0.0, 1.0) * 255.0).long()     # (H, W) int64
    return lut[indices]                               # (H, W, 3)


# ═══════════════════════════════════════════════════════════════════════
#  open_file_dialog() — native file picker (no extra dependencies)
# ═══════════════════════════════════════════════════════════════════════

def open_file_dialog(
    title: str = "Open File",
    filters: "list[tuple[str, str]] | None" = None,
    initial_dir: "str | None" = None,
) -> "str | None":
    """Open a native file-selection dialog and return the chosen path.

    Uses ``tkinter`` from the Python standard library — no extra
    package is required.  On Linux, ``python3-tk`` must be installed
    (``sudo apt install python3-tk``).

    .. note::

        Calling this inside an ``@view.on_frame`` or ``@panel.on_frame``
        callback will briefly pause rendering while the dialog is open.
        This is expected behaviour.

    Args:
        title:       Dialog window title.
        filters:     List of ``(description, glob_pattern)`` tuples, e.g.
                     ``[("Images", "*.png *.jpg"), ("All files", "*.*")]``.
                     Defaults to a preset of common image formats.
        initial_dir: Starting directory.  ``None`` uses the current directory.

    Returns:
        The selected file path as a ``str``, or ``None`` if the user
        cancelled the dialog.

    Example::

        @controls.on_frame
        def draw():
            if controls.button("Open image…"):
                path = vultorch.open_file_dialog()
                if path:
                    t = vultorch.imread(path, channels=3)
                    canvas.bind(t)
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        raise RuntimeError(
            "open_file_dialog() requires tkinter (part of Python's standard "
            "library).  On Linux, install it with:  sudo apt install python3-tk"
        )

    if filters is None:
        filters = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tga *.hdr"),
            ("PNG",         "*.png"),
            ("JPEG",        "*.jpg *.jpeg"),
            ("HDR",         "*.hdr"),
            ("All files",   "*.*"),
        ]

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=title,
        filetypes=filters,
        initialdir=initial_dir,
    )
    root.destroy()
    return path or None


def save_file_dialog(
    title: str = "Save File",
    filters: "list[tuple[str, str]] | None" = None,
    initial_dir: "str | None" = None,
    default_extension: str = ".png",
) -> "str | None":
    """Open a native save-file dialog and return the chosen path.

    Uses ``tkinter`` from the Python standard library.

    Args:
        title:              Dialog window title.
        filters:            List of ``(description, glob_pattern)`` tuples.
                            Defaults to common image formats.
        initial_dir:        Starting directory.  ``None`` uses current directory.
        default_extension:  Extension appended when the user omits one
                            (e.g. ``".png"``).

    Returns:
        The selected file path as a ``str``, or ``None`` if the user cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        raise RuntimeError(
            "save_file_dialog() requires tkinter (part of Python's standard "
            "library).  On Linux, install it with:  sudo apt install python3-tk"
        )

    if filters is None:
        filters = [
            ("PNG",         "*.png"),
            ("JPEG",        "*.jpg *.jpeg"),
            ("HDR",         "*.hdr"),
            ("All files",   "*.*"),
        ]

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.asksaveasfilename(
        title=title,
        filetypes=filters,
        initialdir=initial_dir,
        defaultextension=default_extension,
    )
    root.destroy()
    return path or None


# ═══════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════
def _normalize_tensor(tensor):
    """Normalize tensor dtype/shape for display.

    Accepts float32/float16/uint8 tensors (CUDA or CPU) and returns float32.
    """
    import torch

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
