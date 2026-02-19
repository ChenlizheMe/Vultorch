"""Vultorch Recorder — capture canvas frames to GIF.

Uses ``PIL`` / ``Pillow`` to write animated GIF files.
Quality (0–1) controls the number of colours per frame:
0 → 2 colours, 1 → 256 colours (full quality).

Saving runs in a background thread so the main loop is never blocked.

Usage::

    canvas.start_recording("output.gif", fps=15, quality=0.8)
    # … run frames …
    canvas.stop_recording()
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional


class Recorder:
    """Accumulates tensor frames and writes an animated GIF on stop.

    Not instantiated directly — use :meth:`Canvas.start_recording`.
    """

    __slots__ = (
        "_path", "_fps", "_width", "_height",
        "_gif_frames", "_recording", "_quality",
        "_saving", "_save_thread", "_save_error",
    )

    def __init__(self, path: str, fps: int = 30, quality: float = 0.8):
        ext = os.path.splitext(path)[1].lower()
        if ext != ".gif":
            raise ValueError(
                f"Unsupported recording format '{ext}'. "
                f"Only .gif is supported."
            )

        self._path = os.path.abspath(path)
        self._fps = fps
        self._quality = max(0.0, min(1.0, float(quality)))
        self._recording = True  # ready; backend starts lazily on first feed
        self._width = 0
        self._height = 0
        self._gif_frames: List = []
        self._saving = False
        self._save_thread: Optional[threading.Thread] = None
        self._save_error: Optional[str] = None

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def saving(self) -> bool:
        """``True`` while the GIF is being written in the background."""
        if self._saving and self._save_thread is not None:
            if not self._save_thread.is_alive():
                self._saving = False
                self._save_thread = None
        return self._saving

    @property
    def save_error(self) -> Optional[str]:
        """Error message from background save, or ``None``."""
        return self._save_error

    @property
    def path(self) -> str:
        return self._path

    @property
    def quality(self) -> float:
        return self._quality

    def _start(self, width: int, height: int):
        """Begin recording with known frame dimensions."""
        self._width = width
        self._height = height
        self._recording = True
        self._gif_frames = []

    def feed(self, tensor) -> None:
        """Feed one frame (a bound tensor) to the recorder.

        The tensor is moved to CPU, converted to uint8 RGB, colour-
        quantized according to *quality*, and appended to the frame list.
        """
        if not self._recording:
            return

        import torch

        t = tensor
        # Ensure float32
        if t.dtype == torch.uint8:
            t = t.float().div(255.0)
        elif t.dtype == torch.float16:
            t = t.float()

        # Move to CPU
        if t.is_cuda:
            t = t.cpu()

        # Handle shapes: (H,W), (H,W,1), (H,W,3), (H,W,4)
        if t.ndim == 2:
            t = t.unsqueeze(-1)
        c = t.shape[2]
        if c == 1:
            t = t.expand(-1, -1, 3)
        elif c == 4:
            t = t[:, :, :3]  # drop alpha
        # Now (H, W, 3)

        # Lazy start (first frame determines size)
        h, w = t.shape[0], t.shape[1]
        if self._width == 0:
            self._start(w, h)

        # Apply linear → sRGB gamma so GIF colours match on-screen
        t = t.clamp(0.0, 1.0)
        t = t.pow(1.0 / 2.2)

        # Convert to uint8 RGB bytes → PIL Image
        rgb = (t * 255.0 + 0.5).byte()
        raw = rgb.contiguous().numpy().tobytes()
        img = self._bytes_to_pil(raw, w, h)

        # Colour-quantize based on quality (2..256 colours)
        n_colors = max(2, int(self._quality * 254 + 2))
        if n_colors < 256:
            img = img.quantize(colors=n_colors, method=2).convert("RGB")

        self._gif_frames.append(img)

    def stop(self) -> str:
        """Finalize the recording and start writing the GIF in a
        background thread.

        The file is not ready until :attr:`saving` becomes ``False``.

        Returns the output file path.
        """
        if not self._recording:
            return self._path

        self._recording = False
        self._save_error = None

        # Hand the frame list to a background thread
        frames = list(self._gif_frames)
        self._gif_frames.clear()
        if frames:
            self._saving = True
            self._save_thread = threading.Thread(
                target=self._write_gif_thread,
                args=(frames,),
                daemon=True,
            )
            self._save_thread.start()
        return self._path

    def _write_gif_thread(self, frames: list):
        """Background thread: write frames to an animated GIF."""
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            self._save_error = "GIF recording requires Pillow: pip install Pillow"
            self._saving = False
            return

        try:
            # Ensure output directory exists
            out_dir = os.path.dirname(self._path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            duration_ms = max(1, int(1000 / self._fps))
            frames[0].save(
                self._path,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=True,
            )
        except Exception as exc:
            self._save_error = str(exc)
        finally:
            self._saving = False

    @staticmethod
    def _bytes_to_pil(raw: bytes, w: int, h: int):
        """Convert raw RGB bytes to a PIL Image."""
        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError(
                "GIF recording requires Pillow: pip install Pillow"
            )
        return Image.frombytes("RGB", (w, h), raw)

    @property
    def frame_count(self) -> int:
        """Number of frames recorded so far."""
        return len(self._gif_frames)

    def __del__(self):
        try:
            if self._recording:
                self.stop()
            # Wait for background save so file is complete
            if self._save_thread is not None:
                self._save_thread.join(timeout=60)
        except Exception:
            pass
