"""
Convert captured frames/actions into a NitroGen-style dataset with:
- sliding T-step (configurable) action chunks (T must match your checkpoint's action_horizon; ng.pt uses 18)
- resized RGB frames (square, configurable)

Outputs a single .pt file ready for training:
    {"obs": (M,3,H,W), "actions": (M,T,A), "meta": {...}}

Usage:
    python encode.py --root dataset/gow --out dataset/gow_nitro.pt --img-size 640 --seq-len 18 --workers 8

Credits:
- zrorisc
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import queue
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


_print_lock = threading.Lock()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert capture logs to NitroGen-style .pt")
    p.add_argument("--root", type=Path, required=True, help="Dataset root (contains meta.json, actions.npy, frames/)")
    p.add_argument("--out", type=Path, required=True, help="Output .pt path")
    p.add_argument("--img-size", type=int, default=256, help="Resize frames to img-size x img-size (default 256)")
    p.add_argument("--seq-len", type=int, default=18, help="Action sequence length (default 18; matches ng.pt)")
    p.add_argument("--mouse-scale", type=float, default=300.0, help="Divisor for mouse dx/dy to map into [-1,1]")
    p.add_argument("--mouse-clip", type=float, default=1.0, help="Clip for mouse stick values")
    # Keyboard action space is the only supported output format:
    # action_dim = len(meta.actions) + 4, layout = [buttons..., lx, ly, rx, ry]
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Workers for conversion (frame decode/resize). 1=sequential, 0=auto, >1=threaded.",
    )
    p.add_argument(
        "--cache",
        type=int,
        default=0,
        help="Cache checkpoint interval in frames (0=disabled). Writes a resumable cache alongside --out.",
    )
    p.add_argument(
        "--cache-load",
        action="store_true",
        help="Resume from an existing cache (if it matches current args).",
    )
    p.add_argument("--log-every", type=int, default=500, help="Log progress every N frames while loading.")
    p.add_argument(
        "--save-legacy",
        action="store_true",
        help="Use legacy torch.save format (disables zipfile serialization).",
    )
    return p.parse_args()


def load_meta(root: Path) -> dict:
    meta_path = root / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"meta.json not found at {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def build_frame_list(root: Path, meta: dict) -> List[Path]:
    frames_root = root / "frames"
    frame_paths: List[Path] = []
    for chunk in sorted(meta.get("chunks", []), key=lambda c: c["chunk"]):
        chunk_dir = Path(chunk["frames_dir"])
        if chunk_dir.parts and chunk_dir.parts[0].lower() == "frames":
            chunk_dir = Path(*chunk_dir.parts[1:])
        cdir = frames_root / chunk_dir
        n = int(chunk["num_frames"])
        for i in range(n):
            frame_paths.append(cdir / f"{i+1:06d}.png")
    return frame_paths


def log(msg: str, worker: Optional[int] = None) -> None:
    with _print_lock:
        if worker is None:
            print(f"[convert] {msg}", flush=True)
        else:
            print(f"[convert w{worker}] {msg}", flush=True)


def _human_bytes(n: int) -> str:
    n = int(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(x)}{u}"
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}TB"


class _CountingWriter:
    def __init__(self, f) -> None:
        self._f = f
        self._lock = threading.Lock()
        self._bytes = 0

    @property
    def bytes_written(self) -> int:
        with self._lock:
            return int(self._bytes)

    def write(self, b) -> int:
        n = self._f.write(b)
        with self._lock:
            self._bytes += int(n)
        return n

    def flush(self) -> None:
        self._f.flush()

    def fileno(self) -> int:
        return self._f.fileno()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: object) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2))


def cache_dir_for_out(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".cache")


def _best_effort_write_json_in_place(
    path: Path,
    obj: object,
    *,
    retries: int = 10,
    retry_sleep_s: float = 0.05,
    worker: Optional[int] = None,
) -> bool:
    """
    Best-effort JSON write that never raises (intended for cache progress updates).
    Uses in-place overwrite (no os.replace) to avoid Windows rename/lock issues.
    """
    text = json.dumps(obj, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max(1, int(retries)) + 1):
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            return True
        except PermissionError as e:
            last_exc = e
            if attempt < retries:
                time.sleep(retry_sleep_s * attempt)
                continue
            break
        except Exception as e:
            last_exc = e
            break
    if last_exc is not None:
        log(f"Cache progress write failed: {type(last_exc).__name__}: {last_exc}", worker=worker)
    return False


def _backup_existing_cache(cache_dir: Path) -> None:
    if not cache_dir.exists():
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = cache_dir.with_name(cache_dir.name + f".bak_{ts}")
    cache_dir.rename(backup)


def init_or_load_cache(
    *,
    cache_dir: Path,
    cache_load: bool,
    cache_interval: int,
    root: Path,
    img_size: int,
    seq_len: int,
    action_dim: int,
    mouse_scale: float,
    mouse_clip: float,
    M: int,
) -> Tuple[np.ndarray, Optional[np.memmap], Optional[np.memmap]]:
    """
    Returns (obs_array, obs_mmap_or_none, done_mmap_or_none).
    If caching is disabled, obs_array is an in-memory ndarray and the others are None.
    """
    caching = cache_interval > 0 or cache_load
    if not caching:
        obs_np = np.empty((M, 3, img_size, img_size), dtype=np.float32)
        return obs_np, None, None

    meta_path = cache_dir / "meta.json"
    obs_path = cache_dir / "obs.dat"
    done_path = cache_dir / "done.dat"
    meta = {
        "version": 1,
        "root": str(root),
        "img_size": int(img_size),
        "seq_len": int(seq_len),
        "action_dim": int(action_dim),
        "mouse_scale": float(mouse_scale),
        "mouse_clip": float(mouse_clip),
        "M": int(M),
        "dtype": "float32",
        "shape": [int(M), 3, int(img_size), int(img_size)],
    }

    if cache_load and cache_dir.exists() and meta_path.exists() and obs_path.exists() and done_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing = None
        if existing == meta:
            obs_mm = np.memmap(str(obs_path), mode="r+", dtype=np.float32, shape=(M, 3, img_size, img_size))
            done_mm = np.memmap(str(done_path), mode="r+", dtype=np.uint8, shape=(M,))
            obs_np = np.asarray(obs_mm)
            return obs_np, obs_mm, done_mm
        log(f"Cache found but args mismatch; starting fresh at {cache_dir}")

    if cache_dir.exists():
        _backup_existing_cache(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(meta_path, meta)
    _best_effort_write_json_in_place(cache_dir / "progress.json", {"done": 0, "M": int(M)})
    obs_mm = np.memmap(str(obs_path), mode="w+", dtype=np.float32, shape=(M, 3, img_size, img_size))
    done_mm = np.memmap(str(done_path), mode="w+", dtype=np.uint8, shape=(M,))
    done_mm[:] = 0
    done_mm.flush()
    obs_np = np.asarray(obs_mm)
    return obs_np, obs_mm, done_mm


def resolve_workers(workers: int) -> int:
    if workers < 0:
        raise SystemExit("--workers must be >= 0")
    if workers == 0:
        cpu_ct = os.cpu_count() or 4
        return min(32, max(2, cpu_ct))
    return int(workers)


def split_work(total: int, workers: int) -> List[Tuple[int, int, int]]:
    if total < 0:
        raise ValueError("total must be >= 0")
    workers = max(1, int(workers))
    workers = min(workers, max(1, total))
    ranges: List[Tuple[int, int, int]] = []
    for wid in range(workers):
        start = (wid * total) // workers
        end = ((wid + 1) * total) // workers
        if start < end:
            ranges.append((wid, start, end))
    return ranges


def build_action_windows(actions: np.ndarray, T: int) -> np.ndarray:
    if actions.ndim != 2:
        raise ValueError(f"Expected actions shape (N,A), got {actions.shape}")
    if T <= 0:
        raise ValueError("T must be > 0")
    N, A = actions.shape
    M = N - T + 1
    if M <= 0:
        raise ValueError(f"Not enough actions ({N}) for seq_len {T}")
    s0, s1 = actions.strides
    view = np.lib.stride_tricks.as_strided(actions, shape=(M, T, A), strides=(s0, s0, s1))
    return np.array(view, copy=True)


def map_actions_keyboard_space(
    btns: np.ndarray,
    mouse: np.ndarray,
    actions_list: Sequence[str],
    mouse_scale: float,
    mouse_clip: float,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Keyboard action space:
    - buttons: exactly meta.actions (dynamic K)
    - sticks:  lx/ly derived from WASD/arrow keys, rx/ry from mouse deltas

    Output layout is [buttons..., lx, ly, rx, ry] (action_dim = K + 4).
    Returns (actions_mapped, button_names, action_names).
    """
    if btns.ndim != 2:
        raise ValueError(f"Expected btns shape (N,K), got {btns.shape}")
    if mouse.ndim != 2 or mouse.shape[1] != 2:
        raise ValueError(f"Expected mouse shape (N,2), got {mouse.shape}")

    button_names = [str(x).strip().lower() for x in actions_list if str(x).strip()]
    K = int(btns.shape[1])
    if len(button_names) != K:
        # Trust the actual array width; still keep best-effort names.
        if len(button_names) < K:
            button_names = button_names + [f"key_{i}" for i in range(len(button_names), K)]
        else:
            button_names = button_names[:K]

    index: dict[str, int] = {}
    for i, name in enumerate(button_names):
        if name not in index:
            index[name] = int(i)

    def col(*names: str) -> np.ndarray:
        for name in names:
            i = index.get(name)
            if i is not None:
                return btns[:, i].astype(np.float32, copy=False)
        return np.zeros((btns.shape[0],), dtype=np.float32)

    # Buttons: copy as floats.
    out = np.zeros((btns.shape[0], K + 4), dtype=np.float32)
    out[:, :K] = btns.astype(np.float32, copy=False)

    # Left stick from WASD/arrows (normalized -1..1)
    lx = col("d", "arrow_right") - col("a", "arrow_left")
    ly = col("w", "arrow_up") - col("s", "arrow_down")
    out[:, K + 0] = np.clip(lx, -1.0, 1.0)
    out[:, K + 1] = np.clip(ly, -1.0, 1.0)

    # Right stick from mouse deltas (normalized -1..1)
    out[:, K + 2] = np.clip(mouse[:, 0] / float(mouse_scale), -float(mouse_clip), float(mouse_clip))
    out[:, K + 3] = np.clip(-mouse[:, 1] / float(mouse_scale), -float(mouse_clip), float(mouse_clip))

    action_names = button_names + ["lx", "ly", "rx", "ry"]
    return out, button_names, action_names


def load_and_resize(frame_path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(frame_path))
    if img is None:
        raise FileNotFoundError(f"Frame not found or unreadable: {frame_path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)  # CHW
    return img.astype(np.float32) / 255.0


def stack_with_progress(items: Sequence[np.ndarray], log_every: int, label: str) -> np.ndarray:
    n = len(items)
    if n == 0:
        raise ValueError(f"Cannot stack empty list for {label}")

    sample = np.asarray(items[0])
    out = np.empty((n,) + sample.shape, dtype=sample.dtype)

    if log_every <= 0:
        out[...] = np.stack(items, axis=0)
        return out

    chunk = max(1, log_every)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        out[start:end] = np.stack(items[start:end], axis=0)
        if end % log_every == 0 or end == n:
            log(f"{label} {end}/{n}")
    return out


def main() -> None:
    args = parse_args()
    root: Path = args.root
    log(f"Starting conversion from {root} -> {args.out}")
    meta = load_meta(root)
    log(f"Loaded meta.json (chunks={len(meta.get('chunks', []))}, resize_applied={meta.get('resize_applied')})")
    source_actions = meta.get("actions", []) if isinstance(meta, dict) else []
    if not isinstance(source_actions, list):
        source_actions = []
    source_actions_set = set([str(x).strip().lower() for x in source_actions if str(x).strip()])

    # Friendly warnings to avoid "trained for 12 hours on a broken capture" mistakes.
    move_present = any(k in source_actions_set for k in ("w", "a", "s", "d", "arrow_up", "arrow_down", "arrow_left", "arrow_right"))
    if not move_present:
        log("[warn] No WASD or arrow keys found in meta.actions; left stick will be all zeros.")
    lmb_present = any(k in source_actions_set for k in ("lmb", "left_click"))
    rmb_present = any(k in source_actions_set for k in ("rmb", "right_click"))
    if not lmb_present:
        log("[warn] No left_click/lmb found in meta.actions; mapped LMB slot will be all zeros.")
    if not rmb_present:
        log("[warn] No right_click/rmb found in meta.actions; mapped RMB slot will be all zeros.")

    actions_path = root / "actions.npy"
    mouse_path = root / "mouse_deltas.npy"
    if not actions_path.exists():
        raise SystemExit(f"actions.npy not found at {actions_path}")
    if not mouse_path.exists():
        raise SystemExit(f"mouse_deltas.npy not found at {mouse_path}")

    actions = np.load(actions_path)
    mouse = np.load(mouse_path)
    log(f"Loaded actions {actions.shape} and mouse_deltas {mouse.shape}")
    if actions.shape[0] != mouse.shape[0]:
        raise SystemExit(f"actions and mouse_deltas length mismatch: {actions.shape[0]} vs {mouse.shape[0]}")

    frame_paths = build_frame_list(root, meta)
    log(f"Discovered {len(frame_paths)} frame paths from chunk manifest")
    if len(frame_paths) != actions.shape[0]:
        raise SystemExit(f"Frame count {len(frame_paths)} != actions rows {actions.shape[0]}")

    actions_mapped, button_names, action_names = map_actions_keyboard_space(
        actions, mouse, meta.get("actions", []), args.mouse_scale, args.mouse_clip
    )
    action_dim = int(actions_mapped.shape[1])
    log(f"Mapped actions to keyboard action space ({action_dim}D) -> {actions_mapped.shape}")
    mapping_mode = "keyboard"

    T = args.seq_len
    N = actions_mapped.shape[0]
    M = N - T + 1
    if M <= 0:
        raise SystemExit(f"Not enough frames/actions ({N}) for seq_len {T}")
    log(f"Sequence length {T}; will produce {M} samples")

    def load_idx(idx: int) -> np.ndarray:
        return load_and_resize(frame_paths[idx], args.img_size)

    workers = resolve_workers(args.workers)
    if workers > 1:
        # Avoid oversubscription: OpenCV may use many internal threads (resize/cvtColor),
        # and we also parallelize across workers.
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass
        try:
            log(f"OpenCV threads set to {cv2.getNumThreads()}")
        except Exception:
            pass
    work_ranges = split_work(M, workers)
    cache_dir = cache_dir_for_out(args.out)
    obs_np, obs_mm, done_mm = init_or_load_cache(
        cache_dir=cache_dir,
        cache_load=bool(args.cache_load),
        cache_interval=int(args.cache),
        root=root,
        img_size=int(args.img_size),
        seq_len=int(args.seq_len),
        action_dim=int(action_dim),
        mouse_scale=float(args.mouse_scale),
        mouse_clip=float(args.mouse_clip),
        M=int(M),
    )
    initial_done = 0
    if done_mm is not None:
        try:
            done_count = int(np.count_nonzero(np.asarray(done_mm)))
        except Exception:
            done_count = 0
        initial_done = done_count
        log(f"Cache: {cache_dir} (done={done_count}/{M})")

    log(f"Loading frames at size {args.img_size} (workers={workers}, ranges={len(work_ranges)})")

    if workers == 1:
        done_runtime = initial_done
        persisted_done = initial_done

        def flush_checkpoint_seq(done_indices: List[int]) -> None:
            nonlocal persisted_done
            if not done_indices:
                return
            t0 = time.perf_counter()
            log(f"Cache save attempt: batch={len(done_indices)}")
            persisted_now: Optional[int] = None
            progress_ok = False
            added = 0
            try:
                if obs_mm is not None:
                    obs_mm.flush()
                if done_mm is not None:
                    for ii in done_indices:
                        if int(done_mm[ii]) == 0:
                            done_mm[ii] = 1
                            added += 1
                    done_mm.flush()
                    # persisted_done is tracked incrementally to avoid O(M) scans.
                    persisted_done += int(added)
                    persisted_now = int(persisted_done)
                    progress_ok = _best_effort_write_json_in_place(
                        cache_dir / "progress.json",
                        {"done": persisted_now, "M": int(M)},
                    )
            except Exception as e:
                log(f"Cache save failed (ignored): {type(e).__name__}: {e}")
                return
            if persisted_now is not None:
                log(
                    f"Cache save result: batch={len(done_indices)} added={added} -> done={persisted_now}/{M} progress_json={'ok' if progress_ok else 'failed'} ({time.perf_counter()-t0:.2f}s)"
                )

        done_batch: List[int] = []
        since_checkpoint = 0
        for t in range(M):
            if done_mm is not None and int(done_mm[t]) == 1:
                continue
            obs_np[t] = load_idx(t)
            done_runtime += 1
            if done_mm is not None:
                done_batch.append(t)
                since_checkpoint += 1
                if args.cache > 0 and since_checkpoint >= args.cache:
                    flush_checkpoint_seq(done_batch)
                    done_batch.clear()
                    since_checkpoint = 0
            if args.log_every > 0 and (t + 1) % args.log_every == 0:
                log(f"Progress {done_runtime}/{M} | remaining {M - done_runtime}")
        if done_batch and done_mm is not None:
            flush_checkpoint_seq(done_batch)
        if args.log_every > 0 and M % args.log_every != 0:
            log(f"Progress {done_runtime}/{M} | remaining {M - done_runtime}")
    else:
        cache_lock = threading.Lock()
        progress_lock = threading.Lock()
        global_done_runtime = initial_done
        worker_tick: Dict[int, int] = {wid: 0 for (wid, _, _) in work_ranges}
        last_global_tick_printed = 0
        inactive_tick = 1_000_000_000
        index_q: "queue.SimpleQueue[List[int] | None]" = queue.SimpleQueue()
        checkpoint_done = threading.Event()

        def flush_worker_progress(wid: int, local_done: int, tick: Optional[int]) -> None:
            nonlocal global_done_runtime, last_global_tick_printed
            with progress_lock:
                if local_done:
                    global_done_runtime += int(local_done)
                if args.log_every > 0 and tick is not None:
                    worker_tick[wid] = max(int(worker_tick.get(wid, 0)), int(tick))
                    min_tick = min(worker_tick.values()) if worker_tick else int(tick)
                    if min_tick > last_global_tick_printed:
                        last_global_tick_printed = min_tick
                        log(f"Total {global_done_runtime}/{M} | remaining {M - global_done_runtime}")

        def checkpoint_worker() -> None:
            """
            Global cache saver: flushes after every --cache frames TOTAL across all workers.
            Never raises; failures only log.
            """
            persisted_done = initial_done
            since_checkpoint_total = 0
            pending: List[int] = []

            def flush_pending() -> None:
                nonlocal persisted_done, since_checkpoint_total, pending
                if not pending:
                    return
                t0 = time.perf_counter()
                batch_sz = len(pending)
                log(f"Cache save attempt: batch={batch_sz}")
                added = 0
                progress_ok = False
                try:
                    if obs_mm is not None:
                        obs_mm.flush()
                    if done_mm is not None:
                        with cache_lock:
                            for ii in pending:
                                if int(done_mm[ii]) == 0:
                                    done_mm[ii] = 1
                                    added += 1
                            done_mm.flush()
                            persisted_done += int(added)
                            progress_ok = _best_effort_write_json_in_place(
                                cache_dir / "progress.json",
                                {"done": int(persisted_done), "M": int(M)},
                            )
                except Exception as e:
                    log(f"Cache save failed (ignored): {type(e).__name__}: {e}")
                    pending = []
                    since_checkpoint_total = 0
                    return
                log(
                    f"Cache save result: batch={batch_sz} added={added} -> done={persisted_done}/{M} progress_json={'ok' if progress_ok else 'failed'} ({time.perf_counter()-t0:.2f}s)"
                )
                pending = []
                since_checkpoint_total = 0

            try:
                while True:
                    item = index_q.get()
                    if item is None:
                        break
                    pending.extend(item)
                    since_checkpoint_total += len(item)
                    if args.cache > 0 and since_checkpoint_total >= args.cache:
                        flush_pending()
            except Exception as e:
                log(f"Cache saver crashed (ignored): {type(e).__name__}: {e}")
            finally:
                try:
                    flush_pending()
                finally:
                    checkpoint_done.set()

        def load_range(wid: int, start: int, end: int) -> int:
            total = end - start
            log(f"Starting range 1-{total} (global {start + 1}-{end})", worker=wid)
            cache_batch: List[int] = []
            local_done_unreported = 0
            for j, idx in enumerate(range(start, end), start=1):
                if done_mm is not None and int(done_mm[idx]) == 1:
                    if args.log_every > 0 and (j % args.log_every == 0 or j == total):
                        log(f"Progress {j}/{total}", worker=wid)
                        if j % args.log_every == 0:
                            tick = j // args.log_every
                            flush_worker_progress(wid, local_done_unreported, tick)
                            local_done_unreported = 0
                    continue

                obs_np[idx] = load_idx(idx)
                local_done_unreported += 1
                if args.cache > 0 and done_mm is not None:
                    cache_batch.append(idx)
                    if len(cache_batch) >= 512:
                        index_q.put(cache_batch)
                        cache_batch = []

                if args.log_every > 0 and (j % args.log_every == 0 or j == total):
                    log(f"Progress {j}/{total}", worker=wid)
                    if j % args.log_every == 0:
                        tick = j // args.log_every
                        flush_worker_progress(wid, local_done_unreported, tick)
                        local_done_unreported = 0

            if cache_batch:
                index_q.put(cache_batch)
            flush_worker_progress(wid, local_done_unreported, None)
            local_done_unreported = 0
            log(f"Finished range 1-{total} (global {start + 1}-{end})", worker=wid)
            return wid

        saver_thread: Optional[threading.Thread] = None
        if args.cache > 0 and done_mm is not None:
            saver_thread = threading.Thread(target=checkpoint_worker, name="cache-saver", daemon=True)
            saver_thread.start()

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(load_range, wid, start, end) for (wid, start, end) in work_ranges]
            for fut in as_completed(futures):
                wid = fut.result()
        if saver_thread is not None:
            index_q.put(None)
            checkpoint_done.wait(timeout=60.0)
            saver_thread.join(timeout=5.0)
        if args.log_every > 0:
            with progress_lock:
                log(f"Total {global_done_runtime}/{M} | remaining {M - global_done_runtime}")
        log(f"Finished loading frames ({M}/{M})")

    log("Building action windows")
    acts_np = build_action_windows(actions_mapped, T)
    log(f"Built action windows -> {acts_np.shape}")

    obs = torch.from_numpy(obs_np)  # (M, 3, H, W)
    acts = torch.from_numpy(acts_np)  # (M, T, A)
    log(f"Stacked tensors -> obs {obs.shape}, actions {acts.shape}")

    out_dict: Dict[str, object] = {
        "obs": obs,
        "actions": acts,
        "meta": {
            # Preserve the full capture metadata so downstream tools can merge/trace provenance.
            # This includes keybind profiles, chunks, fps, etc.
            "source_meta": meta,
            "source_root": str(root),
            "img_size": args.img_size,
            "seq_len": args.seq_len,
            "mouse_scale": args.mouse_scale,
            "mouse_clip": args.mouse_clip,
            "action_dim": action_dim,
            "action_names": action_names,
            "button_names": button_names,
            "actions_layout": (
                f"{action_dim}D keyboard space: buttons=len(source_meta.actions), plus lx/ly from WASD/arrows and rx/ry from mouse"
                if not args.map_controller
                else f"{action_dim}D legacy controller mapping derived from PC controls (WASD->LS, mouse->RS)"
            ),
            "mapping_mode": mapping_mode,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(obs.numel() * obs.element_size() + acts.numel() * acts.element_size())
    log(f"Saving dataset (this can take a while) -> {args.out} (approx {_human_bytes(expected_bytes)})")

    tmp_out = args.out.with_suffix(args.out.suffix + ".saving")
    stop_save = threading.Event()

    def monitor_save() -> None:
        t0 = time.perf_counter()
        last_sz = -1
        last_change = t0
        warned_zero = False
        cpu0 = time.process_time()
        while not stop_save.is_set():
            try:
                sz = tmp_out.stat().st_size if tmp_out.exists() else 0
            except Exception:
                sz = 0
            now = time.perf_counter()
            if sz != last_sz:
                last_sz = sz
                last_change = now
            idle_s = now - last_change
            elapsed_s = now - t0
            cpu_s = time.process_time() - cpu0
            if expected_bytes > 0:
                pct = min(100.0, (sz / expected_bytes) * 100.0)
                log(
                    f"Save progress: {_human_bytes(sz)}/{_human_bytes(expected_bytes)} ({pct:.1f}%) | idle {idle_s:.0f}s | elapsed {elapsed_s:.0f}s | cpu {cpu_s:.0f}s"
                )
            else:
                log(f"Save progress: {_human_bytes(sz)} written | idle {idle_s:.0f}s | elapsed {elapsed_s:.0f}s | cpu {cpu_s:.0f}s")
            if not warned_zero and sz == 0 and elapsed_s >= 60:
                warned_zero = True
                log("Save note: 0 bytes so far; torch.save may be preparing metadata before the first write.")
            stop_save.wait(10.0)

    mon = threading.Thread(target=monitor_save, name="save-monitor", daemon=True)
    mon.start()
    save_ok = False
    try:
        torch.save(out_dict, tmp_out, _use_new_zipfile_serialization=not bool(args.save_legacy))
        os.replace(tmp_out, args.out)
        save_ok = True
    except Exception as e:
        log(f"Save failed: {type(e).__name__}: {e}")
        log(f"Partial output kept at {tmp_out}")
    finally:
        stop_save.set()
        mon.join(timeout=1.0)
        if save_ok:
            try:
                if tmp_out.exists():
                    tmp_out.unlink()
            except Exception:
                pass
    if not save_ok:
        raise SystemExit(f"Failed to save dataset to {args.out}")
    log(f"Saved NitroGen-style dataset to {args.out}")
    log(f"obs: {obs.shape}, actions: {acts.shape}")


if __name__ == "__main__":
    main()
