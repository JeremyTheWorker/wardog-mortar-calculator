"""Hotkey-driven mortar range calculator for the War Dog map.

F7 reads a TEAM coordinate as the mortar position. F8 reads the map coordinate
under the mouse and calculates the straight-line range between those points.
The utility only reads the screen and never sends game input.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import queue
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


REFERENCE_WIDTH = 1920
REFERENCE_HEIGHT = 1080
GRID_METERS = 100.0
SET_MORTAR_HOTKEY = "<f7>"
TARGET_HOTKEY = "<f8>"
EXIT_HOTKEY = "<ctrl>+<f8>"

# Measured from the supplied 1920x1080 screenshots.
REF_MAP_RECT = (630.0, 208.0, 1290.0, 854.0)
REF_TEAM_RECT = (15.0, 80.0, 620.0, 300.0)

COORDINATE_PATTERN = re.compile(
    r"(?P<axis>[xy])\s*[:=]?\s*(?P<value>-?\d{1,3}(?:[.,]\d{1,3})?)",
    re.IGNORECASE,
)


class CalculatorError(RuntimeError):
    """An expected error that can be shown directly to the user."""


@dataclass(frozen=True)
class PixelPoint:
    x: float
    y: float


@dataclass(frozen=True)
class MapPoint:
    x: float
    y: float


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True)
class ScreenLayout:
    width: int
    height: int
    scale_x: float
    scale_y: float
    map_rect: Rect
    team_rect: Rect

    @classmethod
    def from_frame(cls, frame: np.ndarray) -> "ScreenLayout":
        if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
            raise CalculatorError("The captured image is not a color screenshot.")

        height, width = frame.shape[:2]
        reference_aspect = REFERENCE_WIDTH / REFERENCE_HEIGHT
        actual_aspect = width / height
        if not math.isclose(actual_aspect, reference_aspect, abs_tol=0.015):
            raise CalculatorError(
                f"Expected a 16:9 screen like the reference image; captured "
                f"{width}x{height}."
            )

        scale_x = width / REFERENCE_WIDTH
        scale_y = height / REFERENCE_HEIGHT
        if scale_x < 0.75 or scale_x > 2.0:
            raise CalculatorError(
                f"Unsupported screen size {width}x{height}; use the 1920x1080 "
                "layout shown in the reference screenshot."
            )

        def scaled_rect(values: Sequence[float]) -> Rect:
            left, top, right, bottom = values
            return Rect(
                round(left * scale_x),
                round(top * scale_y),
                round(right * scale_x),
                round(bottom * scale_y),
            )

        return cls(
            width=width,
            height=height,
            scale_x=scale_x,
            scale_y=scale_y,
            map_rect=scaled_rect(REF_MAP_RECT),
            team_rect=scaled_rect(REF_TEAM_RECT),
        )


@dataclass(frozen=True)
class OcrText:
    text: str
    confidence: float
    left: float
    center_y: float


@dataclass(frozen=True)
class CalculationResult:
    mortar: MapPoint
    target: MapPoint
    distance_m: float
    target_ocr_text: tuple[str, ...]

    @property
    def rounded_distance_m(self) -> int:
        return round(self.distance_m)


def parse_coordinate_texts(texts: Iterable[str]) -> MapPoint:
    """Parse x/y coordinate labels from OCR output."""

    found: dict[str, float] = {}
    for original in texts:
        normalized = (
            str(original)
            .strip()
            .lower()
            .replace(" ", "")
            .replace("−", "-")
            .replace(",", ".")
        )
        for match in COORDINATE_PATTERN.finditer(normalized):
            axis = match.group("axis").lower()
            try:
                found[axis] = float(match.group("value").replace(",", "."))
            except ValueError:
                continue

    missing = [axis for axis in ("x", "y") if axis not in found]
    if missing:
        axes = " and ".join(axis.upper() for axis in missing)
        raise CalculatorError(
            f"Could not read the {axes} coordinate. Open the map, keep the "
            "pointer over the target, and try again."
        )

    return MapPoint(x=found["x"], y=found["y"])


def calculate_distance_m(mortar: MapPoint, target: MapPoint) -> float:
    """Return map-grid Euclidean distance in meters."""

    return GRID_METERS * math.hypot(target.x - mortar.x, target.y - mortar.y)


def create_ocr_engine() -> Any:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise CalculatorError(
            "RapidOCR is not installed. Run: py -m pip install -r requirements.txt"
        ) from exc
    return RapidOCR()


def _unpack_ocr_result(raw_result: Any) -> list[Any]:
    # RapidOCR 1.x returns (lines, elapsed_times).
    if isinstance(raw_result, tuple):
        lines = raw_result[0]
    else:
        lines = raw_result
    return list(lines or [])


def _run_ocr(ocr_engine: Any, image: np.ndarray) -> list[OcrText]:
    lines = _unpack_ocr_result(ocr_engine(image))
    recognized: list[OcrText] = []
    for line in lines:
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            continue
        text = str(line[1])
        confidence = 1.0
        if len(line) >= 3:
            try:
                confidence = float(line[2])
            except (TypeError, ValueError):
                pass
        if confidence >= 0.35:
            left = 0.0
            center_y = 0.0
            try:
                points = np.asarray(line[0], dtype=float).reshape(-1, 2)
                left = float(points[:, 0].min())
                center_y = float(points[:, 1].mean())
            except (TypeError, ValueError, IndexError):
                pass
            recognized.append(
                OcrText(
                    text=text,
                    confidence=confidence,
                    left=left,
                    center_y=center_y,
                )
            )
    return recognized


def read_target_coordinates(
    frame: np.ndarray, layout: ScreenLayout, ocr_engine: Any
) -> tuple[MapPoint, tuple[str, ...]]:
    """OCR the x/y label displayed beside the mouse pointer."""

    rect = layout.map_rect
    map_image = frame[rect.top : rect.bottom, rect.left : rect.right]
    recognized = _run_ocr(ocr_engine, map_image)
    texts = tuple(item.text for item in recognized)

    try:
        point = parse_coordinate_texts(texts)
    except CalculatorError:
        # A high-contrast upscale is useful if compression or scaling softens
        # the small coordinate characters.
        gray = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.75, fy=1.75, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
        retry = _run_ocr(ocr_engine, enhanced)
        texts = texts + tuple(item.text for item in retry)
        point = parse_coordinate_texts(texts)

    if not (
        math.isfinite(point.x)
        and math.isfinite(point.y)
        and -9999.0 <= point.x <= 9999.0
        and -9999.0 <= point.y <= 9999.0
    ):
        raise CalculatorError(
            f"OCR returned invalid coordinates X={point.x:.2f}, Y={point.y:.2f}."
        )

    return point, texts


def _find_latest_team_coordinate(
    recognized: Sequence[OcrText],
    row_tolerance: float,
) -> MapPoint:
    team_labels = [
        item
        for item in recognized
        if re.search(r"\btea[mn]\b", item.text, re.IGNORECASE)
    ]
    candidates: list[tuple[float, MapPoint]] = []

    for team_label in team_labels:
        same_row = [
            item
            for item in recognized
            if abs(item.center_y - team_label.center_y) <= row_tolerance
        ]
        row_text = " ".join(
            item.text for item in sorted(same_row, key=lambda item: item.left)
        )
        try:
            coordinate = parse_coordinate_texts([row_text])
        except CalculatorError:
            continue
        candidates.append((team_label.center_y, coordinate))

    if not candidates:
        raise CalculatorError(
            "Could not read a TEAM mortar coordinate. Mark the mortar location "
            "so its TEAM X/Y line is visible, then press F7."
        )

    return max(candidates, key=lambda item: item[0])[1]


def read_team_mortar_coordinates(
    frame: np.ndarray,
    layout: ScreenLayout,
    ocr_engine: Any,
) -> tuple[MapPoint, tuple[str, ...]]:
    """Read the newest TEAM x/y line from the top-left chat area."""

    rect = layout.team_rect
    team_image = frame[rect.top : rect.bottom, rect.left : rect.right]
    recognized = _run_ocr(ocr_engine, team_image)
    texts = tuple(item.text for item in recognized)

    try:
        point = _find_latest_team_coordinate(
            recognized, row_tolerance=24.0 * layout.scale_y
        )
    except CalculatorError:
        gray = cv2.cvtColor(team_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.75, fy=1.75, interpolation=cv2.INTER_CUBIC)
        enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        retry = _run_ocr(ocr_engine, enhanced)
        texts = texts + tuple(item.text for item in retry)
        point = _find_latest_team_coordinate(
            retry, row_tolerance=42.0 * layout.scale_y
        )

    if not (
        math.isfinite(point.x)
        and math.isfinite(point.y)
        and -9999.0 <= point.x <= 9999.0
        and -9999.0 <= point.y <= 9999.0
    ):
        raise CalculatorError(
            f"OCR returned invalid TEAM coordinates X={point.x:.2f}, "
            f"Y={point.y:.2f}."
        )

    return point, texts


def require_saved_mortar(mortar: MapPoint | None) -> MapPoint:
    if mortar is None:
        raise CalculatorError(
            "No mortar position is saved. Show the TEAM coordinate and press "
            "F7 first."
        )
    return mortar


def analyze_target_frame(
    frame: np.ndarray,
    mortar: MapPoint,
    ocr_engine: Any,
) -> CalculationResult:
    layout = ScreenLayout.from_frame(frame)
    target, ocr_text = read_target_coordinates(frame, layout, ocr_engine)
    distance = calculate_distance_m(mortar, target)

    return CalculationResult(
        mortar=mortar,
        target=target,
        distance_m=distance,
        target_ocr_text=ocr_text,
    )


def draw_debug_image(frame: np.ndarray, result: CalculationResult) -> np.ndarray:
    output = frame.copy()
    layout = ScreenLayout.from_frame(output)
    map_rect = layout.map_rect
    team_rect = layout.team_rect
    cv2.rectangle(
        output,
        (map_rect.left, map_rect.top),
        (map_rect.right, map_rect.bottom),
        (70, 220, 255),
        2,
    )
    cv2.rectangle(
        output,
        (team_rect.left, team_rect.top),
        (team_rect.right, team_rect.bottom),
        (0, 255, 0),
        2,
    )
    label = (
        f"{result.rounded_distance_m} m | "
        f"mortar {result.mortar.x:.2f},{result.mortar.y:.2f} | "
        f"target {result.target.x:.2f},{result.target.y:.2f}"
    )
    cv2.putText(
        output,
        label,
        (40, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def _enable_dpi_awareness() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


class _WinPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _get_cursor_position() -> PixelPoint:
    if os.name != "nt":
        raise CalculatorError("Live mode currently supports Windows only.")
    point = _WinPoint()
    if not ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
        raise CalculatorError("Windows did not return the mouse pointer position.")
    return PixelPoint(float(point.x), float(point.y))


def capture_monitor_under_cursor() -> np.ndarray:
    try:
        import mss
    except ImportError as exc:
        raise CalculatorError(
            "MSS is not installed. Run: py -m pip install -r requirements.txt"
        ) from exc

    global_cursor = _get_cursor_position()
    with mss.MSS() as capture:
        monitor = next(
            (
                item
                for item in capture.monitors[1:]
                if item["left"] <= global_cursor.x < item["left"] + item["width"]
                and item["top"] <= global_cursor.y < item["top"] + item["height"]
            ),
            None,
        )
        if monitor is None:
            raise CalculatorError("Could not identify the monitor under the pointer.")
        shot = np.asarray(capture.grab(monitor))

    return np.ascontiguousarray(shot[:, :, :3])


class RangeOverlay:
    """Small Windows overlay configured to display without taking focus."""

    def __init__(self) -> None:
        import tkinter as tk

        self._tk = tk
        self.root = tk.Tk()
        self.root.withdraw()
        self.window = tk.Toplevel(self.root)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.window.attributes("-alpha", 0.93)
        self.window.configure(bg="#101820", padx=18, pady=12)

        self.range_label = tk.Label(
            self.window,
            text="",
            font=("Segoe UI", 24, "bold"),
            fg="#72ff72",
            bg="#101820",
        )
        self.range_label.pack()
        self.detail_label = tk.Label(
            self.window,
            text="",
            font=("Segoe UI", 10),
            fg="#e9eef2",
            bg="#101820",
        )
        self.detail_label.pack(pady=(3, 0))
        self._hide_after: str | None = None
        self.window.update_idletasks()
        self._configure_no_activate()

    def _configure_no_activate(self) -> None:
        if os.name != "nt":
            return
        hwnd = self.window.winfo_id()
        get_style = ctypes.windll.user32.GetWindowLongW
        set_style = ctypes.windll.user32.SetWindowLongW
        ex_style = get_style(hwnd, -20)
        ws_ex_transparent = 0x00000020
        ws_ex_toolwindow = 0x00000080
        ws_ex_noactivate = 0x08000000
        set_style(
            hwnd,
            -20,
            ex_style | ws_ex_transparent | ws_ex_toolwindow | ws_ex_noactivate,
        )

    def show_result(self, result: CalculationResult) -> None:
        self.range_label.configure(
            text=f"RANGE  {result.rounded_distance_m} m", fg="#72ff72"
        )
        self.detail_label.configure(
            text=(
                f"Mortar {result.mortar.x:.2f}, {result.mortar.y:.2f}   "
                f"Target {result.target.x:.2f}, {result.target.y:.2f}"
            )
        )
        self._show(3200)

    def show_mortar_saved(self, mortar: MapPoint) -> None:
        self.range_label.configure(text="MORTAR SAVED", fg="#72ff72")
        self.detail_label.configure(text=f"X={mortar.x:.2f}, Y={mortar.y:.2f}")
        self._show(3200)

    def show_error(self, message: str) -> None:
        self.range_label.configure(text="NO RANGE", fg="#ff6b6b")
        self.detail_label.configure(text=message)
        self._show(4800)

    def _show(self, duration_ms: int) -> None:
        if self._hide_after is not None:
            self.root.after_cancel(self._hide_after)
        self.window.update_idletasks()
        width = self.window.winfo_reqwidth()
        height = self.window.winfo_reqheight()
        x = (self.window.winfo_screenwidth() - width) // 2
        y = max(35, round(self.window.winfo_screenheight() * 0.075))
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        self.window.deiconify()
        self.window.lift()
        if os.name == "nt":
            hwnd = self.window.winfo_id()
            hwnd_topmost = -1
            swp_noactivate = 0x0010
            swp_showwindow = 0x0040
            ctypes.windll.user32.SetWindowPos(
                hwnd,
                hwnd_topmost,
                x,
                y,
                width,
                height,
                swp_noactivate | swp_showwindow,
            )
        self._hide_after = self.root.after(duration_ms, self.window.withdraw)


def run_live(ocr_engine: Any) -> None:
    try:
        from pynput import keyboard
    except ImportError as exc:
        raise CalculatorError(
            "pynput is not installed. Run: py -m pip install -r requirements.txt"
        ) from exc

    overlay = RangeOverlay()
    messages: queue.Queue[tuple[str, Any]] = queue.Queue()
    busy = threading.Event()
    state_lock = threading.Lock()
    mortar_position: MapPoint | None = None

    def save_mortar_once() -> None:
        nonlocal mortar_position
        try:
            frame = capture_monitor_under_cursor()
            layout = ScreenLayout.from_frame(frame)
            mortar, _ = read_team_mortar_coordinates(frame, layout, ocr_engine)
            with state_lock:
                mortar_position = mortar
            messages.put(("mortar", mortar))
        except CalculatorError as exc:
            messages.put(("error", str(exc)))
        except Exception as exc:  # Keep the hotkey listener alive after a failure.
            messages.put(("error", f"Unexpected error: {exc}"))
        finally:
            busy.clear()

    def calculate_once() -> None:
        try:
            with state_lock:
                saved_mortar = require_saved_mortar(mortar_position)
            frame = capture_monitor_under_cursor()
            result = analyze_target_frame(frame, saved_mortar, ocr_engine)
            messages.put(("result", result))
        except CalculatorError as exc:
            messages.put(("error", str(exc)))
        except Exception as exc:  # Keep the hotkey listener alive after a failure.
            messages.put(("error", f"Unexpected error: {exc}"))
        finally:
            busy.clear()

    def start_worker(worker: Any) -> None:
        if busy.is_set():
            return
        busy.set()
        threading.Thread(target=worker, daemon=True).start()

    def on_set_mortar_hotkey() -> None:
        start_worker(save_mortar_once)

    def on_target_hotkey() -> None:
        start_worker(calculate_once)

    def on_exit_hotkey() -> None:
        messages.put(("exit", None))

    listener = keyboard.GlobalHotKeys(
        {
            SET_MORTAR_HOTKEY: on_set_mortar_hotkey,
            TARGET_HOTKEY: on_target_hotkey,
            EXIT_HOTKEY: on_exit_hotkey,
        }
    )
    listener.start()

    def poll_messages() -> None:
        try:
            while True:
                kind, payload = messages.get_nowait()
                if kind == "mortar":
                    overlay.show_mortar_saved(payload)
                    print(
                        f"Mortar saved: X={payload.x:.2f}, Y={payload.y:.2f}"
                    )
                elif kind == "result":
                    overlay.show_result(payload)
                    print(
                        f"Range {payload.rounded_distance_m} m | "
                        f"mortar=({payload.mortar.x:.2f}, {payload.mortar.y:.2f}) "
                        f"target=({payload.target.x:.2f}, {payload.target.y:.2f})"
                    )
                elif kind == "error":
                    overlay.show_error(payload)
                    print(f"Could not calculate: {payload}", file=sys.stderr)
                elif kind == "exit":
                    overlay.root.quit()
                    return
        except queue.Empty:
            pass
        overlay.root.after(40, poll_messages)

    overlay.root.after(40, poll_messages)
    print("Mortar calculator ready.")
    print("Show the TEAM mortar coordinate and press F7 to save it.")
    print("Then point at a map target and press F8 for the range.")
    print("Press Ctrl+F8 to quit.")
    try:
        overlay.root.mainloop()
    finally:
        listener.stop()
        listener.join(timeout=1.0)
        overlay.root.destroy()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate mortar range from the War Dog map."
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Analyze a saved screenshot instead of listening for F8.",
    )
    parser.add_argument(
        "--debug-output",
        type=Path,
        help="Write an annotated image showing the OCR regions.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    _enable_dpi_awareness()
    try:
        print("Loading coordinate reader...")
        ocr_engine = create_ocr_engine()
        if args.image:
            frame = cv2.imread(str(args.image))
            if frame is None:
                raise CalculatorError(f"Could not open image: {args.image}")
            layout = ScreenLayout.from_frame(frame)
            mortar, _ = read_team_mortar_coordinates(frame, layout, ocr_engine)
            result = analyze_target_frame(frame, mortar, ocr_engine)
            print(f"Range: {result.rounded_distance_m} m")
            print(f"Mortar: X={result.mortar.x:.2f}, Y={result.mortar.y:.2f}")
            print(f"Target: X={result.target.x:.2f}, Y={result.target.y:.2f}")
            if args.debug_output:
                debug_image = draw_debug_image(frame, result)
                args.debug_output.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.debug_output), debug_image):
                    raise CalculatorError(
                        f"Could not write debug image: {args.debug_output}"
                    )
                print(f"Debug image: {args.debug_output}")
            return 0

        if args.debug_output:
            raise CalculatorError("--debug-output requires --image.")
        if os.name != "nt":
            raise CalculatorError("Live hotkey mode currently supports Windows only.")
        run_live(ocr_engine)
        return 0
    except CalculatorError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
