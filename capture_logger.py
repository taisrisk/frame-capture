"""
Frame + input logger for NitroGen fine-tuning on any Windows game.

- Captures frames via DXCam at a target FPS.
- Records keyboard/mouse buttons and mouse deltas per frame.
- Writes chunked frames plus aligned action/mouse logs, then merges into dataset files.
- Emits alignment metadata (frame indices) to catch any drift.

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


# Action vector layout (order matters).
ACTIONS: Sequence[str] = (
    "w",
    "a",
    "s",
    "d",
    "space",
    "shift",
    "ctrl",
    "e",
    "q",
    "r",
    "f",
    "g",
    "c",
    "v",
    "x",
    "z",
    "t",
    "p",  # pause/resume hotkey
    "i",
    "backspace",
    "esc",
    "1",
    "2",
    "3",
    "4",
    "enter",
    "lmb",
    "rmb",
)

KEY_CHAR_MAP = {
    "w": "w",
    "a": "a",
    "s": "s",
    "d": "d",
    "e": "e",
    "q": "q",
    "r": "r",
    "f": "f",
    "g": "g",
    "c": "c",
    "v": "v",
    "x": "x",
    "z": "z",
    "t": "t",
    "p": "p",
    "i": "i",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
}


def key_to_action(key: keyboard.Key | keyboard.KeyCode) -> Optional[str]:
    if isinstance(key, keyboard.KeyCode):
        ch = key.char.lower() if key.char else None
        return KEY_CHAR_MAP.get(ch)
    if key in (keyboard.Key.space,):
        return "space"
    if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
        return "shift"
    if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        return "ctrl"
    if key == keyboard.Key.backspace:
        return "backspace"
    if key == keyboard.Key.esc:
        return "esc"
    if key == keyboard.Key.enter:
        return "enter"
    if isinstance(key, keyboard.KeyCode) and key.vk == 0x0D:
        return "enter"
    return None


def button_to_action(button: mouse.Button) -> Optional[str]:
    if button == mouse.Button.left:
        return "lmb"
    if button == mouse.Button.right:
        return "rmb"
    return None


class InputState:
    def __init__(self, actions: Iterable[str]) -> None:
        self.lock = threading.Lock()
        self.state: Dict[str, bool] = {a: False for a in actions}
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
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

    def snapshot(self) -> tuple[List[int], float, float]:
        with self.lock:
            buttons = [1 if self.state.get(a, False) else 0 for a in ACTIONS]
            dx, dy = self.mouse_dx, self.mouse_dy
            self.mouse_dx = 0.0
            self.mouse_dy = 0.0
            return buttons, dx, dy

    def clear_mouse(self) -> None:
        with self.lock:
            self.mouse_dx = 0.0
            self.mouse_dy = 0.0
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
        nonlocal rect_out
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
    parser = argparse.ArgumentParser(description="Frame + input logger for NitroGen fine-tuning.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for dataset.")
    parser.add_argument("--fps", type=float, default=30.0, help="Target capture FPS (default 30).")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap on frames (0 = unlimited).")
    parser.add_argument("--region", type=int, nargs=4, metavar=("left", "top", "right", "bottom"), help="Capture region.")
    parser.add_argument("--window-title", type=str, help="Partial window title to auto-detect capture region.")
    parser.add_argument(
        "--process-name", type=str, help="Exact process name to auto-detect window region. Example: Game.exe"
    )
    parser.add_argument("--monitor", type=int, default=0, help="Monitor index for dxcam (0 = primary).")
    parser.add_argument("--chunk-size", type=int, default=500, help="Frames per chunk before flushing to disk.")
    parser.add_argument("--status-interval", type=float, default=2.0, help="Seconds between status prints (0=off).")
    parser.add_argument("--skip-late", action="store_true", help="Skip a frame if behind schedule.")
    parser.add_argument("--start-immediately", action="store_true", help="Start capturing without waiting for hotkey.")
    parser.add_argument("--debug-list-windows", action="store_true", help="List visible window titles/PIDs then exit.")
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

    input_state = InputState(ACTIONS)
    running = True
    capturing = bool(args.start_immediately)
    capture_state = "running" if capturing else "paused"

    def stop_running(signum, frame):  # type: ignore[unused-argument]
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop_running)
    signal.signal(signal.SIGTERM, stop_running)

    def handle_hotkeys(action: Optional[str], raw_key=None) -> None:
        nonlocal capturing, capture_state
        if not action:
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
            return
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
        action = key_to_action(k)
        if action:
            input_state.set_action(action, True)
        handle_hotkeys(action, raw_key=k)

    def on_release(k):
        action = key_to_action(k)
        if action:
            input_state.set_action(action, False)

    kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    mouse_listener = mouse.Listener(
        on_click=lambda x, y, button, pressed: input_state.set_action(action, pressed)
        if (action := button_to_action(button))
        else None,
        on_move=lambda x, y: input_state.on_mouse_move(x, y),
    )
    kb_listener.start()
    mouse_listener.start()

    if kb_global:
        kb_global.add_hotkey("ctrl+t", lambda: handle_hotkeys("t"))
        kb_global.add_hotkey("ctrl+p", lambda: handle_hotkeys("p"))
        kb_global.add_hotkey("f6", lambda: handle_hotkeys(None, keyboard.Key.f6))
        kb_global.add_hotkey("f7", lambda: handle_hotkeys(None, keyboard.Key.f7))
        kb_global.add_hotkey("f8", lambda: handle_hotkeys(None, keyboard.Key.f8))

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

    frame_interval = 1.0 / max(args.fps, 1e-3)
    next_tick = time.perf_counter()
    frame_id = 0
    actions_log: List[List[int]] = []
    mouse_deltas: List[tuple[float, float]] = []
    frame_indices: List[int] = []
    chunk_id = 1
    chunk_meta: List[dict] = []
    chunk_start_frame = 1
    chunk_frame_count = 0
    chunk_dir = frames_dir / f"chunk_{chunk_id:03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    skipped_frames = 0
    timestamps = deque(maxlen=240)
    last_status = time.perf_counter()

    print(f"[+] Starting capture -> {out_dir} (FPS={args.fps}, region={region}, hotkey_paused={not capturing})")
    print(f"[+] Actions order: {ACTIONS}")

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
        chunk_meta.append(
            {
                "chunk": int(chunk_id),
                "start_frame": int(chunk_start_frame),
                "end_frame": int(chunk_start_frame + len(actions_array) - 1),
                "actions_file": actions_file.name,
                "mouse_deltas_file": mouse_file.name,
                "frame_indices_file": indices_file.name,
                "frames_dir": str(chunk_dir.relative_to(out_dir)),
                "num_frames": int(len(actions_array)),
            }
        )
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
            if args.skip_late and now - next_tick > frame_interval:
                next_tick = now + frame_interval
                skipped_frames += 1
                input_state.clear_mouse()
                continue
            next_tick += frame_interval

            frame = cam.grab(region=region)
            if frame is None:
                print("[!] dxcam returned None frame; skipping")
                input_state.clear_mouse()
                continue

            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]

            frame_id += 1
            local_idx = len(actions_log) + 1
            frame_path = chunk_dir / f"{local_idx:06d}.png"
            cv2.imwrite(str(frame_path), frame)

            buttons, dx, dy = input_state.snapshot()
            actions_log.append(buttons)
            mouse_deltas.append((dx, dy))
            frame_indices.append(frame_id)
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
                    f"FPS~{fps_now:.1f} | Skipped: {skipped_frames} | State: {capture_state}"
                )
                last_status = time.perf_counter()

    finally:
        running = False
        kb_listener.stop()
        mouse_listener.stop()

    flush_chunk()

    def chunk_key(p: Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 0

    actions_arrays = [np.load(p) for p in sorted(out_dir.glob("actions_chunk_*.npy"), key=chunk_key)]
    mouse_arrays = [np.load(p) for p in sorted(out_dir.glob("mouse_deltas_chunk_*.npy"), key=chunk_key)]
    index_arrays = [np.load(p) for p in sorted(out_dir.glob("frame_indices_chunk_*.npy"), key=chunk_key)]

    actions_merged = (
        np.concatenate(actions_arrays, axis=0) if actions_arrays else np.zeros((0, len(ACTIONS)), dtype=np.int8)
    )
    mouse_merged = np.concatenate(mouse_arrays, axis=0) if mouse_arrays else np.zeros((0, 2), dtype=np.float32)
    frame_indices_merged = np.concatenate(index_arrays, axis=0) if index_arrays else np.zeros((0,), dtype=np.int64)

    if len(actions_merged) != len(mouse_merged) or len(actions_merged) != len(frame_indices_merged):
        raise SystemExit(
            f"Length mismatch after merge: actions={len(actions_merged)}, mouse={len(mouse_merged)}, indices={len(frame_indices_merged)}"
        )
    if len(frame_indices_merged) > 0:
        expected = np.arange(frame_indices_merged[0], frame_indices_merged[0] + len(frame_indices_merged))
        if not np.array_equal(frame_indices_merged, expected):
            raise SystemExit("Frame indices are not contiguous after merge; check capture alignment.")

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
        "actions": list(ACTIONS),
        "num_frames": int(frame_id),
        "action_file": "actions.npy",
        "mouse_deltas_file": "mouse_deltas.npy",
        "frame_indices_file": "frame_indices.npy",
        "mouse_delta_units": "pixels_per_frame",
        "chunk_size": args.chunk_size,
        "num_chunks": int(len(chunk_meta)),
        "chunks": chunk_meta,
        "torch_dataset": torch_info,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

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
