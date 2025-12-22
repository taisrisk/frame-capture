"""
Convert captured frames/actions into a NitroGen-style dataset with
- actions mapped to a 20-D or 25-D gamepad-like space
- sliding 16-step action chunks
- resized RGB frames (square, configurable)

Outputs a single .pt file ready for training: {"obs": (M,3,H,W), "actions": (M,T,A), "meta": {...}}

Usage:
    python convert_to_nitrogen.py --root dataset/gow --out dataset/gow_nitro.pt --img-size 640 --workers 8 --action-dim 25

Credits:
- zrorisc
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert capture logs to NitroGen-style .pt")
    p.add_argument("--root", type=Path, required=True, help="Dataset root (contains meta.json, actions.npy, frames/)")
    p.add_argument("--out", type=Path, required=True, help="Output .pt path")
    p.add_argument("--img-size", type=int, default=256, help="Resize frames to img-size x img-size (default 256)")
    p.add_argument("--seq-len", type=int, default=16, help="Action sequence length (default 16)")
    p.add_argument("--mouse-scale", type=float, default=300.0, help="Divisor for mouse dx/dy to map into [-1,1]")
    p.add_argument("--mouse-clip", type=float, default=1.0, help="Clip for mouse stick values")
    p.add_argument(
        "--action-dim",
        type=int,
        choices=(20, 25),
        default=20,
        help="Output action dimension. Use 25 if your checkpoint expects 25-D actions.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Threads for image decode/resize (0=auto, 1=disabled/sequential).",
    )
    p.add_argument("--log-every", type=int, default=500, help="Log progress every N frames while loading.")
    return p.parse_args()


def load_meta(root: Path) -> dict:
    meta_path = root / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"meta.json not found at {meta_path}")
    return json.loads(meta_path.read_text())


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


def log(msg: str) -> None:
    print(f"[convert] {msg}")


def action_names_for_dim(action_dim: int) -> List[str]:
    base = [
        "lx",
        "ly",
        "rx",
        "ry",
        "space",
        "ctrl",
        "shift",
        "e",
        "lmb",
        "rmb",
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
    ]
    if action_dim == 20:
        return base
    if action_dim == 25:
        # Extra slots align with extended keyboard inputs used by some checkpoints.
        # Order: x, z, enter, esc, i
        return base + ["x", "z", "enter", "esc", "i"]
    raise SystemExit(f"Unsupported action_dim {action_dim}; use 20 or 25.")


def map_actions(
    btns: np.ndarray, mouse: np.ndarray, actions_list: Sequence[str], mouse_scale: float, mouse_clip: float, action_dim: int
) -> np.ndarray:
    def idx(name: str) -> Optional[int]:
        try:
            return actions_list.index(name)
        except ValueError:
            return None

    out = np.zeros((btns.shape[0], action_dim), dtype=np.float32)

    def b(name: str) -> np.ndarray:
        i = idx(name)
        return btns[:, i] if i is not None else np.zeros((btns.shape[0],), dtype=np.int8)

    # Left stick from WASD
    out[:, 0] = b("d") - b("a")  # x
    out[:, 1] = b("w") - b("s")  # y

    # Right stick from mouse deltas
    out[:, 2] = np.clip(mouse[:, 0] / mouse_scale, -mouse_clip, mouse_clip)  # x
    out[:, 3] = np.clip(-mouse[:, 1] / mouse_scale, -mouse_clip, mouse_clip)  # y (invert to typical look up/down)

    # Face / shoulders / misc mapping (binary)
    base_slots = [
        ("space", 4),
        ("ctrl", 5),
        ("shift", 6),
        ("e", 7),
        ("lmb", 8),
        ("rmb", 9),
        ("q", 10),
        ("r", 11),
        ("f", 12),
        ("g", 13),
        ("c", 14),
        ("v", 15),
        ("1", 16),
        ("2", 17),
        ("3", 18),
        ("4", 19),
    ]
    for name, slot in base_slots:
        if slot < action_dim:
            out[:, slot] = b(name)

    if action_dim == 25:
        # Extra 5 dims map to x, z, enter, esc, i (indices 20-24 respectively).
        extra_slots = [
            ("x", 20),
            ("z", 21),
            ("enter", 22),
            ("esc", 23),
            ("i", 24),
        ]
        for name, slot in extra_slots:
            out[:, slot] = b(name)
    elif action_dim != 20:
        raise SystemExit(f"Unsupported action_dim {action_dim}; use 20 or 25.")

    return out


def load_and_resize(frame_path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(frame_path))
    if img is None:
        raise FileNotFoundError(f"Frame not found or unreadable: {frame_path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)  # CHW
    return img.astype(np.float32) / 255.0


def main() -> None:
    args = parse_args()
    root: Path = args.root
    log(f"Starting conversion from {root} -> {args.out}")
    meta = load_meta(root)
    log(f"Loaded meta.json (chunks={len(meta.get('chunks', []))}, resize_applied={meta.get('resize_applied')})")

    actions = np.load(root / "actions.npy")
    mouse = np.load(root / "mouse_deltas.npy")
    log(f"Loaded actions {actions.shape} and mouse_deltas {mouse.shape}")
    if actions.shape[0] != mouse.shape[0]:
        raise SystemExit(f"actions and mouse_deltas length mismatch: {actions.shape[0]} vs {mouse.shape[0]}")

    frame_paths = build_frame_list(root, meta)
    log(f"Discovered {len(frame_paths)} frame paths from chunk manifest")
    if len(frame_paths) != actions.shape[0]:
        raise SystemExit(f"Frame count {len(frame_paths)} != actions rows {actions.shape[0]}")

    action_dim = args.action_dim
    action_names = action_names_for_dim(action_dim)
    actions_mapped = map_actions(actions, mouse, meta.get("actions", []), args.mouse_scale, args.mouse_clip, action_dim)
    log(f"Mapped actions to {action_dim}D layout -> {actions_mapped.shape}")

    T = args.seq_len
    N = actions_mapped.shape[0]
    M = N - T + 1
    if M <= 0:
        raise SystemExit(f"Not enough frames/actions ({N}) for seq_len {T}")
    log(f"Sequence length {T}; will produce {M} samples")

    obs_list: List[np.ndarray]
    act_list: List[np.ndarray] = []

    def load_idx(idx: int) -> np.ndarray:
        return load_and_resize(frame_paths[idx], args.img_size)

    if args.workers == 1:
        log(f"Loading frames sequentially (workers=1) at size {args.img_size}")
        obs_list = []
        for t in range(M):
            obs_list.append(load_idx(t))
            if args.log_every > 0 and (t + 1) % args.log_every == 0:
                log(f"Loaded {t + 1}/{M} frames")
    else:
        max_workers = None if args.workers == 0 else max(1, args.workers)
        if max_workers is None:
            cpu_ct = os.cpu_count() or 4
            max_workers = min(32, max(2, cpu_ct))
        log(f"Loading frames with {max_workers} workers at size {args.img_size}")
        obs_list = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for idx, frame in enumerate(ex.map(load_idx, range(M)), start=1):
                obs_list.append(frame)
                if args.log_every > 0 and idx % args.log_every == 0:
                    log(f"Loaded {idx}/{M} frames")
        log(f"Finished loading frames ({len(obs_list)}/{M})")

    for t in range(M):
        act_list.append(actions_mapped[t : t + T])
        if args.log_every > 0 and (t + 1) % args.log_every == 0:
            log(f"Prepared actions window {t + 1}/{M}")

    obs = torch.from_numpy(np.stack(obs_list))  # (M, 3, H, W)
    acts = torch.from_numpy(np.stack(act_list))  # (M, T, A)
    log(f"Stacked tensors -> obs {obs.shape}, actions {acts.shape}")

    out_dict: Dict[str, object] = {
        "obs": obs,
        "actions": acts,
        "meta": {
            "img_size": args.img_size,
            "seq_len": args.seq_len,
            "mouse_scale": args.mouse_scale,
            "mouse_clip": args.mouse_clip,
            "action_dim": action_dim,
            "action_names": action_names,
            "actions_layout": f"{action_dim}D mapping derived from PC controls (WASD->LS, mouse->RS)",
        },
    }
    torch.save(out_dict, args.out)
    log(f"Saved NitroGen-style dataset to {args.out}")
    log(f"obs: {obs.shape}, actions: {acts.shape}")


if __name__ == "__main__":
    main()
