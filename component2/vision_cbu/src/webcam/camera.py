"""Single-webcam capture with guaranteed cleanup, plus a mock camera for tests/demo.

Use as a context manager so the device is released on stop, exception or shutdown:

    with Camera(cfg) as cam:
        for ts, frame in cam.frames(max_seconds=10):
            ...
Open cameras are also released by an atexit hook.
"""
from __future__ import annotations

import atexit
import platform
import time
import weakref
from typing import Iterator

import cv2
import numpy as np

_OPEN_CAMERAS: "weakref.WeakSet[BaseCamera]" = weakref.WeakSet()


class CameraError(RuntimeError):
    pass


class BaseCamera:
    def __init__(self, sample_every_n: int = 1):
        self.sample_every_n = max(1, int(sample_every_n))
        self.is_open = False
        self._t0: float | None = None

    def open(self) -> "BaseCamera":
        raise NotImplementedError

    def _read_raw(self) -> tuple[bool, np.ndarray | None]:
        raise NotImplementedError

    def _release_raw(self) -> None:
        raise NotImplementedError

    def read(self) -> tuple[float, np.ndarray | None]:
        """Return (timestamp_s since open, frame or None on failure)."""
        if not self.is_open:
            raise CameraError("Camera is not open")
        ok, frame = self._read_raw()
        ts = time.monotonic() - (self._t0 or time.monotonic())
        return ts, (frame if ok and frame is not None else None)

    def frames(self, max_frames: int | None = None, max_seconds: float | None = None,
               max_consecutive_failures: int = 30) -> Iterator[tuple[float, np.ndarray]]:
        """Yield sampled (timestamp, frame). Raises CameraError after repeated failures."""
        failures = emitted = idx = 0
        start_ts: float | None = None
        while self.is_open:
            ts, frame = self.read()
            if start_ts is None:
                start_ts = ts
            if max_seconds is not None and ts - start_ts >= max_seconds:   # camera clock, not loop speed
                break
            if frame is None:
                failures += 1
                if failures >= max_consecutive_failures:
                    raise CameraError(f"{failures} consecutive frame-read failures")
                continue
            failures = 0
            idx += 1
            if (idx - 1) % self.sample_every_n:
                continue
            yield ts, frame
            emitted += 1
            if max_frames is not None and emitted >= max_frames:
                break

    def release(self) -> None:
        if self.is_open:
            try:
                self._release_raw()
            finally:
                self.is_open = False
                _OPEN_CAMERAS.discard(self)

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.release()
        return False

    def __del__(self):  # last-resort cleanup
        try:
            self.release()
        except Exception:
            pass


class Camera(BaseCamera):
    """Physical webcam via OpenCV. On Windows, DirectShow is used by default (fast open)."""

    def __init__(self, cfg: dict):
        c = cfg["camera"]
        super().__init__(c.get("sample_every_n_frames", 1))
        self.index, self.width, self.height, self.fps = c["index"], c["width"], c["height"], c["fps"]
        self.backend = c.get("backend", "auto")
        self._cap = None
        self.api_name: str | None = None

    def _apis(self) -> list[tuple[str, int]]:
        """Capture backends to try, in order. 'auto' on Windows: DirectShow, Media Foundation, default."""
        table = {"dshow": ("DirectShow", cv2.CAP_DSHOW), "msmf": ("MediaFoundation", cv2.CAP_MSMF),
                 "any": ("default", cv2.CAP_ANY)}
        if self.backend in table:
            return [table[self.backend]]
        if platform.system() == "Windows":
            return [table["dshow"], table["msmf"], table["any"]]
        return [table["any"]]

    def open(self, warmup_frames: int = 5) -> "Camera":
        tried = []
        for name, api in self._apis():
            cap = cv2.VideoCapture(self.index, api)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                ok = False
                for _ in range(max(1, warmup_frames)):          # many webcams return black/None first
                    ok, frame = cap.read()
                if ok and frame is not None:
                    self._cap, self.api_name = cap, name
                    break
                tried.append(f"{name}: opened but returned no frames")
            else:
                tried.append(f"{name}: could not open")
            cap.release()
        if self._cap is None:
            raise CameraError(f"Cannot open camera index {self.index} ({'; '.join(tried)}). Check that a webcam "
                              "is connected and not in use by another app (Teams, Zoom, Camera), that camera "
                              "access for desktop apps is allowed in Windows Settings > Privacy & security > "
                              "Camera, or try another camera index in Settings.")
        self.is_open = True
        self._t0 = time.monotonic()
        _OPEN_CAMERAS.add(self)
        return self

    def _read_raw(self):
        return self._cap.read()

    def _release_raw(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class MockCamera(BaseCamera):
    """Generates synthetic frames (no device). `fail_after` simulates a camera failure."""

    def __init__(self, width: int = 640, height: int = 480, sample_every_n: int = 1,
                 fail_after: int | None = None, brightness: int = 120, fps: float = 30.0):
        super().__init__(sample_every_n)
        self.width, self.height, self.fail_after, self.brightness = width, height, fail_after, brightness
        self.n_read = 0
        self.fps = fps
        self.released_count = 0

    def open(self) -> "MockCamera":
        self.is_open = True
        self._t0 = time.monotonic()
        _OPEN_CAMERAS.add(self)
        return self

    def read(self):
        if not self.is_open:
            raise CameraError("Camera is not open")
        ok, frame = self._read_raw()
        return self.n_read / self.fps, (frame if ok else None)

    def _read_raw(self):
        self.n_read += 1
        if self.fail_after is not None and self.n_read > self.fail_after:
            return False, None
        frame = np.full((self.height, self.width, 3), self.brightness, np.uint8)
        cv2.ellipse(frame, (self.width // 2, self.height // 2), (self.width // 6, self.height // 4),
                    0, 0, 360, (180, 160, 150), -1)
        cv2.putText(frame, "MOCK FRAME", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return True, frame

    def _release_raw(self):
        self.released_count += 1


def probe_camera(cfg: dict) -> tuple[bool, str]:
    """Try to open the configured camera and read one frame; always releases."""
    try:
        with Camera(cfg) as cam:
            _, frame = cam.read()
            if frame is None:
                return False, "Camera opened but returned no frame"
            return True, f"Camera {cam.index} OK ({frame.shape[1]}x{frame.shape[0]})"
    except CameraError as exc:
        return False, str(exc)
    except Exception as exc:  # pragma: no cover
        return False, f"Camera error: {exc}"


@atexit.register
def _release_all() -> None:
    for cam in list(_OPEN_CAMERAS):
        try:
            cam.release()
        except Exception:
            pass
