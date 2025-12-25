"""
Frame + input logger for NitroGen fine-tuning on any Windows game.

- Captures frames via DXCam at a target FPS.
- Records keyboard/mouse buttons and mouse deltas per frame.
- Writes chunked frames plus aligned action/mouse logs, then merges into dataset files.
- Optional downscale before saving (e.g., 1280x720) to cut disk and speed up conversion.
- Emits alignment metadata (capture tick indices) to catch dropped frames/drift.

Usage (PowerShell):
    python capture_logger.py --out dataset/session1 --fps 30 --process-name Game.exe
    python capture_logger.py --out dataset/session1 --fps 30 --region 0 0 1920 1080

Outputs:
    actions.npy, mouse_deltas.npy, frame_indices.npy, dataset.npz/.pt, meta.json, frames/chunk_XXX/*.png

Credits:
- zrorisc
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PATH_REPO = Path(__file__).resolve().parent
PATH_KEYBINDS = PATH_REPO / "keybinds"

def _install_comtypes_unraisable_suppressor() -> None:
    """
    dxcam uses COM underneath (via comtypes on some setups). On some Python/comtypes
    builds, interpreter shutdown can emit an "Exception ignored while calling deallocator
    _compointer_base.__del__ ... access violation" message even after a clean run.

    This hook suppresses that specific shutdown-only noise without hiding other errors.
    """
    if not hasattr(sys, "unraisablehook") or not hasattr(sys, "__unraisablehook__"):
        return

    orig = sys.__unraisablehook__  # type: ignore[attr-defined]

    def _hook(args):  # type: ignore[no-untyped-def]
        msg = getattr(args, "err_msg", "") or ""
        exc = getattr(args, "exc_value", None)
        if (
            "_compointer_base.__del__" in msg
            and isinstance(exc, OSError)
            and "access violation" in str(exc).lower()
        ):
            return
        return orig(args)

    sys.unraisablehook = _hook  # type: ignore[assignment]


_install_comtypes_unraisable_suppressor()

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("opencv-python is required (pip install opencv-python)") from exc

try:
    import dxcam
except ImportError as exc:  # pragma: no cover
    raise SystemExit("dxcam is required (pip install dxcam)") from exc

try:
    from pynput import keyboard, mouse
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pynput is required (pip install pynput)") from exc

try:
    import keyboard as kb_global  # optional global hotkey helper
except ImportError:
    kb_global = None  # type: ignore[assignment]

try:
    import psutil
except ImportError as exc:  # pragma: no cover
    raise SystemExit("psutil is required (pip install psutil)") from exc

try:
    import win32gui
except ImportError:
    win32gui = None  # type: ignore[assignment]
try:
    import win32process
except ImportError:
    win32process = None  # type: ignore[assignment]


# Default action list if the keybind profile is missing/unreadable.
# This is the "old" 25-key capture layout used by earlier runs.
DEFAULT_ACTIONS: Sequence[str] = (
    "w",
    "a",
    "s",
    "d",
    "space",
    "ctrl",
    "shift",
    "e",
    "left_click",
    "right_click",
    "q",
    "r",
    "f",
    "g",
    "c",
    "v",
    "1",
    "2",
    "3",
    "4",
    "x",
    "z",
    "enter",
    "esc",
    "i",
)


def _resolve_keybinds_path(spec: str | None) -> Path:
    s = (spec or "").strip()
    if not s:
        return (PATH_KEYBINDS / "default.json").resolve()
    p = Path(s)
    if p.suffix.lower() == ".json":
        if p.exists():
            return p.resolve()
        # Allow `--keybinds gowr.json` to refer to keybinds/gowr.json
        return (PATH_KEYBINDS / p).resolve()
    return (PATH_KEYBINDS / f"{p.name}.json").resolve()


def _load_keybind_profile(spec: str | None) -> tuple[list[str], dict]:
    path = _resolve_keybinds_path(spec)
    if not path.exists():
        return list(DEFAULT_ACTIONS), {"path": str(path), "fallback": True}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        return list(DEFAULT_ACTIONS), {"path": str(path), "fallback": True}
    keys = obj.get("keybinds") or []
    special = obj.get("special_keys") or []
    mouse_keys = obj.get("mouse") or []
    raw: list[str] = []
    for seq in (keys, special, mouse_keys):
        if isinstance(seq, (list, tuple)):
            raw.extend([str(x).strip().lower() for x in seq if str(x).strip()])
        elif isinstance(seq, str) and seq.strip():
            raw.append(seq.strip().lower())

    # Stable de-dupe preserving order.
    seen: set[str] = set()
    actions: list[str] = []
    for a in raw:
        if a in seen:
            continue
        seen.add(a)
        actions.append(a)

    if not actions:
        actions = list(DEFAULT_ACTIONS)
    return actions, {"path": str(path), "profile": obj, "fallback": False}


def key_to_action(key: keyboard.Key | keyboard.KeyCode, allowed: set[str]) -> Optional[str]:
    if isinstance(key, keyboard.KeyCode):
        if key.char:
            ch = str(key.char).lower()
            return ch if ch in allowed else None
        # Some pynput versions report Enter as KeyCode(vk=0x0D) with no char.
        if getattr(key, "vk", None) == 0x0D:
            return "enter" if "enter" in allowed else None
        return None

    special_map = {
        keyboard.Key.space: "space",
        keyboard.Key.tab: "tab",
        keyboard.Key.caps_lock: "caps_lock",
        keyboard.Key.enter: "enter",
        keyboard.Key.esc: "esc",
        keyboard.Key.backspace: "backspace",
        keyboard.Key.delete: "delete",
        keyboard.Key.insert: "insert",
        keyboard.Key.home: "home",
        keyboard.Key.end: "end",
        keyboard.Key.page_up: "page_up",
        keyboard.Key.page_down: "page_down",
        keyboard.Key.up: "arrow_up",
        keyboard.Key.down: "arrow_down",
        keyboard.Key.left: "arrow_left",
        keyboard.Key.right: "arrow_right",
    }
    if key in special_map:
        name = special_map[key]
        return name if name in allowed else None

    if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
        return "shift" if "shift" in allowed else None
    if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        return "ctrl" if "ctrl" in allowed else None
    if key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
        return "alt" if "alt" in allowed else None

    # Function keys
    for i in range(1, 13):
        if key == getattr(keyboard.Key, f"f{i}", None):
            name = f"f{i}"
            return name if name in allowed else None

    return None


def button_to_action(button: mouse.Button, allowed: set[str]) -> Optional[str]:
    mapping = {
        mouse.Button.left: "left_click",
        mouse.Button.right: "right_click",
        getattr(mouse.Button, "middle", None): "middle_click",
        getattr(mouse.Button, "x1", None): "mouse_x1",
        getattr(mouse.Button, "x2", None): "mouse_x2",
    }
    name = mapping.get(button)
    if not name:
        return None
    return name if name in allowed else None


class InputState:
    def __init__(self, actions: Iterable[str]) -> None:
        self.lock = threading.Lock()
        self.actions: list[str] = list(actions)
        self.state: Dict[str, bool] = {a: False for a in self.actions}
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
        self.mouse_scroll_y = 0.0
        self._last_mouse_pos: Optional[tuple[float, float]] = None

    def set_action(self, action: str, pressed: bool) -> None:
        with self.lock:
            if action in self.state:
                self.state[action] = pressed

    def on_mouse_move(self, x: float, y: float) -> None:
        with self.lock:
            if self._last_mouse_pos is None:
                self._last_mouse_pos = (x, y)
                return
            last_x, last_y = self._last_mouse_pos
            self.mouse_dx += x - last_x
            self.mouse_dy += y - last_y
            self._last_mouse_pos = (x, y)

    def on_mouse_scroll(self, dx: float, dy: float) -> None:
        with self.lock:
            # We only track vertical scroll for now (mouse wheel).
            self.mouse_scroll_y += float(dy)

    def snapshot(self) -> tuple[List[int], float, float]:
        with self.lock:
            # Scroll is treated as an impulse (pressed for one frame if wheel moved).
            scroll_up = 1 if self.mouse_scroll_y > 0 else 0
            scroll_down = 1 if self.mouse_scroll_y < 0 else 0
            buttons: list[int] = []
            for a in self.actions:
                if a == "scroll_up":
                    buttons.append(scroll_up)
                elif a == "scroll_down":
                    buttons.append(scroll_down)
                else:
                    buttons.append(1 if self.state.get(a, False) else 0)
            dx, dy = self.mouse_dx, self.mouse_dy
            self.mouse_dx = 0.0
            self.mouse_dy = 0.0
            self.mouse_scroll_y = 0.0
            return buttons, dx, dy

    def clear_mouse(self) -> None:
        with self.lock:
            self.mouse_dx = 0.0
            self.mouse_dy = 0.0
            self.mouse_scroll_y = 0.0
            self._last_mouse_pos = None


def rect_from_window_title(title_substr: str) -> Optional[tuple[int, int, int, int]]:
    if not win32gui:
        print("[!] pywin32 not installed; window-title capture unavailable.")
        return None

    title_substr_l = title_substr.lower()
    target_hwnd = None

    def enum_handler(hwnd, _):
        nonlocal target_hwnd
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if title and title_substr_l in title.lower():
            target_hwnd = hwnd

    win32gui.EnumWindows(enum_handler, None)
    if not target_hwnd:
        print(f"[!] No window found containing title: {title_substr}")
        return None

    rect = win32gui.GetWindowRect(target_hwnd)
    left, top, right, bottom = rect
    if right - left <= 0 or bottom - top <= 0:
        print(f"[!] Window rect invalid: {rect}")
        return None
    return left, top, right, bottom


def rect_from_process_name(proc_name: str) -> Optional[Tuple[tuple[int, int, int, int], int, int]]:
    if not win32gui:
        print("[!] pywin32 not installed; process-name capture unavailable.")
        return None
    if not win32process:
        print("[!] win32process not available; process-name capture unavailable.")
        return None

    proc_name_l = proc_name.lower()
    target_pid = None
    for p in psutil.process_iter(["pid", "name"]):
        try:
            if p.info["name"] and p.info["name"].lower() == proc_name_l:
                target_pid = p.info["pid"]
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not target_pid:
        print(f"[!] No process found with name: {proc_name}")
        return None

    rect_out = None
    pid_out = None
    hwnd_out = None

    def enum_handler(hwnd, _):
        nonlocal rect_out, pid_out, hwnd_out
        if not win32gui.IsWindowVisible(hwnd):
            return
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        if pid != target_pid:
            return
        rect = win32gui.GetWindowRect(hwnd)
        left, top, right, bottom = rect
        if right - left > 0 and bottom - top > 0:
            rect_out = (left, top, right, bottom)
            pid_out = pid
            hwnd_out = hwnd

    win32gui.EnumWindows(enum_handler, None)
    if not rect_out or pid_out is None or hwnd_out is None:
        print(f"[!] Could not find visible window for PID {target_pid} ({proc_name})")
        return None
    return rect_out, pid_out, hwnd_out


def list_visible_windows() -> List[Tuple[int, str, tuple[int, int, int, int]]]:
    windows: List[Tuple[int, str, tuple[int, int, int, int]]] = []
    if not win32gui or not win32process:
        return windows

    def enum_handler(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
        except Exception:
            return
        rect = win32gui.GetWindowRect(hwnd)
        windows.append((pid, title, rect))

    win32gui.EnumWindows(enum_handler, None)
    return windows


def parse_args() -> argparse.Namespace:
    def _positive_float(v: str) -> float:
        try:
            f = float(v)
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Invalid float: {v!r}") from e
        if f <= 0:
            raise argparse.ArgumentTypeError(f"--fps must be > 0 (got {f})")
        return f

    def _positive_int(v: str) -> int:
        try:
            i = int(v)
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Invalid int: {v!r}") from e
        if i <= 0:
            raise argparse.ArgumentTypeError(f"value must be > 0 (got {i})")
        return i

    def _nonneg_float(v: str) -> float:
        try:
            f = float(v)
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Invalid float: {v!r}") from e
        if f < 0:
            raise argparse.ArgumentTypeError(f"value must be >= 0 (got {f})")
        return f

    parser = argparse.ArgumentParser(description="Frame + input logger for NitroGen fine-tuning.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for dataset.")
    parser.add_argument("--fps", type=_positive_float, default=30.0, help="Target capture FPS (default 30).")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap on frames (0 = unlimited).")
    parser.add_argument("--region", type=int, nargs=4, metavar=("left", "top", "right", "bottom"), help="Capture region.")
    parser.add_argument("--window-title", type=str, help="Partial window title to auto-detect capture region.")
    parser.add_argument("--process-name", type=str, help="Exact process name to auto-detect window region. Example: Game.exe")
    parser.add_argument("--monitor", type=int, default=0, help="Monitor index for dxcam (0 = primary).")
    parser.add_argument("--chunk-size", type=_positive_int, default=500, help="Frames per chunk before flushing to disk.")
    parser.add_argument("--status-interval", type=_nonneg_float, default=2.0, help="Seconds between status prints (0=off).")
    parser.add_argument("--skip-late", action="store_true", help="Skip a frame if behind schedule.")
    parser.add_argument("--start-immediately", action="store_true", help="Start capturing without waiting for hotkey.")
    parser.add_argument("--debug-list-windows", action="store_true", help="List visible window titles/PIDs then exit.")
    parser.add_argument(
        "--keybinds",
        type=str,
        default="default",
        help="Keybind profile JSON path or name under keybinds/<name>.json (default: keybinds/default.json).",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("width", "height"),
        help="Resize frames before saving (e.g., 1280 720 for 720p). Uses area interpolation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.debug_list_windows:
        wins = list_visible_windows()
        if not wins:
            print("[!] No visible windows found or pywin32 missing.")
        else:
            for pid, title, rect in wins:
                print(f"PID={pid} | title='{title}' | rect={rect}")
        sys.exit(0)

    out_dir: Path = args.out
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    actions_list, keybind_profile_meta = _load_keybind_profile(args.keybinds)
    allowed_actions = set(actions_list)
    input_state = InputState(actions_list)
    running = True
    capturing = bool(args.start_immediately)
    capture_state = "running" if capturing else "paused"
    resize_shape: Optional[Tuple[int, int]] = tuple(args.resize) if args.resize else None
    frame_size: Optional[Tuple[int, int]] = None

    def stop_running(signum, frame):  # type: ignore[unused-argument]
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop_running)
    signal.signal(signal.SIGTERM, stop_running)

    def handle_hotkeys(action: Optional[str], raw_key=None) -> None:
        nonlocal capturing, capture_state
        if raw_key == keyboard.Key.f8:
            capturing = not capturing
            capture_state = "running" if capturing else "paused"
            state_txt = "RESUMED" if capturing else "PAUSED"
            print(f"[*] Capture {state_txt} (F8)")
        if raw_key == keyboard.Key.f6:
            capturing = True
            capture_state = "running"
            print("[*] Capture RESUMED (F6)")
        if raw_key == keyboard.Key.f7:
            capturing = False
            capture_state = "paused"
            input_state.clear_mouse()
            print("[*] Capture PAUSED (F7)")

        if action == "t":
            if input_state.state.get("ctrl", False):
                capturing = True
                capture_state = "running"
                print("[*] Capture RESUMED (Ctrl+T)")
        elif action == "p":
            if input_state.state.get("ctrl", False):
                capturing = False
                capture_state = "paused"
                input_state.clear_mouse()
                print("[*] Capture PAUSED (Ctrl+P)")

    def on_press(k):
        action = key_to_action(k, allowed_actions)
        if action:
            input_state.set_action(action, True)
        handle_hotkeys(action, raw_key=k)

    def on_release(k):
        action = key_to_action(k, allowed_actions)
        if action:
            input_state.set_action(action, False)

    kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    mouse_listener = mouse.Listener(
        on_click=lambda x, y, button, pressed: input_state.set_action(action, pressed)
        if (action := button_to_action(button, allowed_actions))
        else None,
        on_move=lambda x, y: input_state.on_mouse_move(x, y),
        on_scroll=lambda x, y, dx, dy: input_state.on_mouse_scroll(dx, dy),
    )
    kb_listener.start()
    mouse_listener.start()

    if kb_global:
        kb_global.add_hotkey("ctrl+t", lambda: handle_hotkeys("t"))
        kb_global.add_hotkey("ctrl+p", lambda: handle_hotkeys("p"))
        # Avoid double-logging F6/F7/F8 since pynput already captures those keys in on_press.

    region = tuple(args.region) if args.region else None
    target_pid: Optional[int] = None
    target_hwnd: Optional[int] = None
    if region is None and args.process_name:
        res = rect_from_process_name(args.process_name)
        if res:
            region, target_pid, target_hwnd = res
    if region is None and args.window_title:
        rect = rect_from_window_title(args.window_title)
        if rect:
            region = rect

    if region is None:
        print("[!] Could not resolve capture region. Falling back to full monitor capture.")

    cam = dxcam.create(output_idx=args.monitor, output_color="BGRA")

    frame_interval = 1.0 / args.fps
    next_tick = time.perf_counter()
    frame_id = 0
    capture_tick = 0
    actions_log: List[List[int]] = []
    mouse_deltas: List[tuple[float, float]] = []
    frame_indices: List[int] = []
    chunk_id = 1
    chunk_meta: List[dict] = []
    chunk_start_frame = 1
    chunk_frame_count = 0
    chunk_dir = frames_dir / f"chunk_{chunk_id:03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    skipped_ticks = 0
    none_frames = 0
    timestamps = deque(maxlen=240)
    last_status = time.perf_counter()

    print(
        f"[+] Starting capture -> {out_dir} (target_fps={args.fps}, frame_interval={frame_interval*1000:.1f}ms, "
        f"region={region}, resize={resize_shape or 'native'}, hotkey_paused={not capturing}, skip_late={args.skip_late})"
    )
    kb_path = keybind_profile_meta.get("path")
    kb_fallback = bool(keybind_profile_meta.get("fallback"))
    print(f"[+] Keybind profile: {kb_path} (fallback={kb_fallback})")
    print(f"[+] Actions order ({len(actions_list)}): {actions_list}")

    def flush_chunk() -> None:
        nonlocal chunk_id, chunk_start_frame, actions_log, mouse_deltas, frame_indices, chunk_dir, chunk_frame_count
        if not actions_log:
            return
        actions_array = np.array(actions_log, dtype=np.int8)
        mouse_array = np.array(mouse_deltas, dtype=np.float32)
        indices_array = np.array(frame_indices, dtype=np.int64)
        actions_file = out_dir / f"actions_chunk_{chunk_id:03d}.npy"
        mouse_file = out_dir / f"mouse_deltas_chunk_{chunk_id:03d}.npy"
        indices_file = out_dir / f"frame_indices_chunk_{chunk_id:03d}.npy"
        np.save(actions_file, actions_array)
        np.save(mouse_file, mouse_array)
        np.save(indices_file, indices_array)
        chunk_entry = {
            "chunk": int(chunk_id),
            "start_frame": int(chunk_start_frame),
            "end_frame": int(chunk_start_frame + len(actions_array) - 1),
            "actions_file": actions_file.name,
            "mouse_deltas_file": mouse_file.name,
            "frame_indices_file": indices_file.name,
            "frames_dir": str(chunk_dir.relative_to(frames_dir)),
            "num_frames": int(len(actions_array)),
        }
        if frame_size:
            chunk_entry["frame_size"] = {"width": int(frame_size[0]), "height": int(frame_size[1])}
        chunk_meta.append(chunk_entry)
        chunk_id += 1
        chunk_start_frame += len(actions_array)
        actions_log.clear()
        mouse_deltas.clear()
        frame_indices.clear()
        chunk_frame_count = 0
        chunk_dir = frames_dir / f"chunk_{chunk_id:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        print(f"[+] Flushed chunk {chunk_id - 1:03d} ({actions_array.shape[0]} frames)")

    try:
        while running and (args.max_frames <= 0 or frame_id < args.max_frames):
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(next_tick - now)
                continue

            if not capturing:
                next_tick = time.perf_counter() + frame_interval
                input_state.clear_mouse()
                continue

            # One scheduled capture attempt.
            capture_tick += 1

            if args.skip_late and now - next_tick > frame_interval:
                next_tick = now + frame_interval
                skipped_ticks += 1
                input_state.clear_mouse()
                continue
            next_tick += frame_interval

            frame = cam.grab(region=region)
            if frame is None:
                none_frames += 1
                input_state.clear_mouse()
                continue

            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]
            if resize_shape:
                frame = cv2.resize(frame, resize_shape, interpolation=cv2.INTER_AREA)
            frame_size = (frame.shape[1], frame.shape[0])  # width, height

            frame_id += 1
            local_idx = len(actions_log) + 1
            frame_path = chunk_dir / f"{local_idx:06d}.png"
            ok = cv2.imwrite(str(frame_path), frame)
            if not ok:
                print(f"[!] Failed to write {frame_path}; dropping frame")
                input_state.clear_mouse()
                frame_id -= 1
                continue

            buttons, dx, dy = input_state.snapshot()
            actions_log.append(buttons)
            mouse_deltas.append((dx, dy))
            frame_indices.append(capture_tick)
            chunk_frame_count += 1
            timestamps.append(time.perf_counter())

            if chunk_frame_count >= args.chunk_size:
                flush_chunk()

            if args.status_interval > 0 and time.perf_counter() - last_status >= args.status_interval:
                fps_now = 0.0
                if len(timestamps) >= 2:
                    dt = timestamps[-1] - timestamps[0]
                    if dt > 0:
                        fps_now = (len(timestamps) - 1) / dt
                print(
                    f"[=] Frames: {frame_id} | Chunk: {chunk_id:03d} size {len(actions_log)} | "
                    f"FPS~{fps_now:.1f} | DroppedTicks: {skipped_ticks + none_frames} "
                    f"(skip_late={skipped_ticks}, none={none_frames}) | State: {capture_state}"
                )
                last_status = time.perf_counter()

    finally:
        running = False
        kb_listener.stop()
        mouse_listener.stop()
        try:
            cam.stop()
        except Exception:
            pass
        try:
            cam.release()
        except Exception:
            pass
        # dxcam uses COM underneath; encourage cleanup before interpreter shutdown.
        try:
            del cam
        except Exception:
            pass
        gc.collect()

    flush_chunk()

    def chunk_key(p: Path) -> int:
        m = re.search(r"(\\d+)", p.stem)
        return int(m.group(1)) if m else 0

    actions_arrays = [np.load(p) for p in sorted(out_dir.glob("actions_chunk_*.npy"), key=chunk_key)]
    mouse_arrays = [np.load(p) for p in sorted(out_dir.glob("mouse_deltas_chunk_*.npy"), key=chunk_key)]
    index_arrays = [np.load(p) for p in sorted(out_dir.glob("frame_indices_chunk_*.npy"), key=chunk_key)]

    actions_merged = (
        np.concatenate(actions_arrays, axis=0) if actions_arrays else np.zeros((0, len(actions_list)), dtype=np.int8)
    )
    mouse_merged = np.concatenate(mouse_arrays, axis=0) if mouse_arrays else np.zeros((0, 2), dtype=np.float32)
    frame_indices_merged = np.concatenate(index_arrays, axis=0) if index_arrays else np.zeros((0,), dtype=np.int64)

    if len(actions_merged) != len(mouse_merged) or len(actions_merged) != len(frame_indices_merged):
        raise SystemExit(
            f"Length mismatch after merge: actions={len(actions_merged)}, mouse={len(mouse_merged)}, indices={len(frame_indices_merged)}"
        )

    # Drift diagnostics: indices are "capture ticks" (attempt counter), so gaps indicate dropped frames.
    dropped_ticks = 0
    if len(frame_indices_merged) > 0:
        if np.any(np.diff(frame_indices_merged) <= 0):
            raise SystemExit("frame_indices are not strictly increasing; capture alignment is corrupted.")
        span = int(frame_indices_merged[-1] - frame_indices_merged[0] + 1)
        dropped_ticks = int(span - len(frame_indices_merged))
        if dropped_ticks > 0:
            print(f"[warn] Detected {dropped_ticks} dropped capture ticks (gaps in frame_indices).")

    np.save(out_dir / "actions.npy", actions_merged)
    np.save(out_dir / "mouse_deltas.npy", mouse_merged)
    np.save(out_dir / "frame_indices.npy", frame_indices_merged)
    np.savez(
        out_dir / "dataset.npz",
        actions=actions_merged,
        mouse_deltas=mouse_merged,
        frame_indices=frame_indices_merged,
    )

    try:
        import torch  # type: ignore

        torch.save(
            {
                "actions": torch.tensor(actions_merged, dtype=torch.int8),
                "mouse_deltas": torch.tensor(mouse_merged, dtype=torch.float32),
                "frame_indices": torch.tensor(frame_indices_merged, dtype=torch.int64),
            },
            out_dir / "dataset.pt",
        )
        torch_info = "dataset.pt"
    except ImportError:
        torch_info = "torch not installed; skipped dataset.pt"

    meta = {
        "fps": args.fps,
        "region": region,
        "monitor": args.monitor,
        "frame_size": {"width": int(frame_size[0]), "height": int(frame_size[1])} if frame_size else None,
        "resize_applied": bool(resize_shape),
        "resize_target": {"width": resize_shape[0], "height": resize_shape[1]} if resize_shape else None,
        "actions": list(actions_list),
        "keybinds_profile": keybind_profile_meta,
        "num_frames": int(frame_id),
        "num_capture_ticks": int(capture_tick),
        "dropped_ticks": int(dropped_ticks),
        "skipped_ticks": int(skipped_ticks),
        "none_frames": int(none_frames),
        "frame_index_units": "capture_tick_index (gaps indicate dropped frames)",
        "action_file": "actions.npy",
        "mouse_deltas_file": "mouse_deltas.npy",
        "frame_indices_file": "frame_indices.npy",
        "mouse_delta_units": "pixels_per_frame",
        "chunk_size": args.chunk_size,
        "num_chunks": int(len(chunk_meta)),
        "chunks": chunk_meta,
        "torch_dataset": torch_info,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[+] Saved {frame_id} frames to {frames_dir}")
    print(f"[+] Action tensor: {actions_merged.shape} -> {out_dir / 'actions.npy'}")
    print(f"[+] Mouse deltas: {mouse_merged.shape} -> {out_dir / 'mouse_deltas.npy'}")
    print(f"[+] Frame indices: {frame_indices_merged.shape} -> {out_dir / 'frame_indices.npy'}")
    print(f"[+] Merged dataset: {out_dir / 'dataset.npz'}")
    print(f"[+] Torch dataset: {torch_info}")
    print(f"[+] Meta: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

