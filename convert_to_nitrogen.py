"""
Convert captured frames/actions into a NitroGen-style dataset with
- actions mapped to a 20-D gamepad-like space
- sliding 16-step action chunks
- resized RGB frames

Outputs a single .pt file ready for training: {"obs": (M,3,H,W), "actions": (M,T,20), "meta": {...}}

Usage:
    python convert_to_nitrogen.py --root dataset/gow --out dataset/gow_nitro.pt

Credits:
- zrorisc
"""

from __future__ import annotations

import argparse
import json
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
        cdir = frames_root / chunk["frames_dir"]
        n = int(chunk["num_frames"])
        for i in range(n):
            frame_paths.append(cdir / f"{i+1:06d}.png")
    return frame_paths


def map_to_20d(btns: np.ndarray, mouse: np.ndarray, actions_list: Sequence[str], mouse_scale: float, mouse_clip: float) -> np.ndarray:
    def idx(name: str) -> Optional[int]:
        try:
            return actions_list.index(name)
        except ValueError:
            return None

    out = np.zeros((btns.shape[0], 20), dtype=np.float32)

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
    out[:, 4] = b("space")
    out[:, 5] = b("ctrl")
    out[:, 6] = b("shift")
    out[:, 7] = b("e")
    out[:, 8] = b("lmb")
    out[:, 9] = b("rmb")
    out[:, 10] = b("q")
    out[:, 11] = b("r")
    out[:, 12] = b("f")
    out[:, 13] = b("g")
    out[:, 14] = b("c")
    out[:, 15] = b("v")
    # Use last four for number keys 1-4
    out[:, 16] = b("1")
    out[:, 17] = b("2")
    out[:, 18] = b("3")
    out[:, 19] = b("4")

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
    meta = load_meta(root)

    actions = np.load(root / "actions.npy")
    mouse = np.load(root / "mouse_deltas.npy")
    if actions.shape[0] != mouse.shape[0]:
        raise SystemExit(f"actions and mouse_deltas length mismatch: {actions.shape[0]} vs {mouse.shape[0]}")

    frame_paths = build_frame_list(root, meta)
    if len(frame_paths) != actions.shape[0]:
        raise SystemExit(f"Frame count {len(frame_paths)} != actions rows {actions.shape[0]}")

    actions_20 = map_to_20d(actions, mouse, meta.get("actions", []), args.mouse_scale, args.mouse_clip)

    T = args.seq_len
    N = actions_20.shape[0]
    M = N - T + 1
    if M <= 0:
        raise SystemExit(f"Not enough frames/actions ({N}) for seq_len {T}")

    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []

    for t in range(M):
        frame = load_and_resize(frame_paths[t], args.img_size)
        obs_list.append(frame)
        act_list.append(actions_20[t : t + T])

    obs = torch.from_numpy(np.stack(obs_list))  # (M, 3, H, W)
    acts = torch.from_numpy(np.stack(act_list))  # (M, T, 20)

    out_dict: Dict[str, object] = {
        "obs": obs,
        "actions": acts,
        "meta": {
            "img_size": args.img_size,
            "seq_len": args.seq_len,
            "mouse_scale": args.mouse_scale,
            "mouse_clip": args.mouse_clip,
            "actions_layout": "20D gamepad mapping derived from PC controls",
        },
    }
    torch.save(out_dict, args.out)
    print(f"[+] Saved NitroGen-style dataset: {args.out}")
    print(f"    obs: {obs.shape}, actions: {acts.shape}")


if __name__ == "__main__":
    main()
