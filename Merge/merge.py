"""
Merge chunked capture outputs into consolidated numpy/torch files.

Use this if you recorded with `capture_logger.py` and want to rebuild:
- actions.npy
- mouse_deltas.npy
- frame_indices.npy (if present)
- dataset.npz
- dataset.pt (optional, if torch is installed)

Usage:
    python dataset_merge.py --root dataset/gow

Credits:
- zrorisc
"""

from __future__ import annotations

import argparse
import json
import re
import concurrent.futures
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge chunked capture outputs.")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root directory (contains meta.json).")
    parser.add_argument("--workers", type=int, default=0, help="Workers for parallel loading (0 = auto/threaded).")
    return parser.parse_args()


def load_meta(root: Path) -> dict:
    meta_path = root / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"meta.json not found in {root}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _chunk_key(path: Path) -> int:
    m = re.search(r"(\\d+)", path.stem)
    return int(m.group(1)) if m else -1


def _list_chunks(root: Path, pattern: str) -> List[Path]:
    return sorted(root.glob(pattern), key=_chunk_key)


def _load_many(paths: List[Path], *, workers: int) -> List[np.ndarray]:
    if not paths:
        return []
    if workers == 1:
        return [np.load(p) for p in paths]
    max_workers = None if workers == 0 else max(1, int(workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(np.load, paths))


def _concat(parts: List[np.ndarray], *, empty: np.ndarray) -> np.ndarray:
    if not parts:
        return empty
    return np.concatenate(parts, axis=0)


def main() -> None:
    args = parse_args()
    root: Path = args.root
    meta = load_meta(root)

    action_dim = int(len(meta.get("actions", [])) or 0)
    if action_dim <= 0:
        # Still allow merging even if meta is incomplete.
        action_dim = 1

    action_files = _list_chunks(root, "actions_chunk_*.npy")
    mouse_files = _list_chunks(root, "mouse_deltas_chunk_*.npy")
    index_files = _list_chunks(root, "frame_indices_chunk_*.npy")

    actions_parts = _load_many(action_files, workers=args.workers)
    mouse_parts = _load_many(mouse_files, workers=args.workers)
    index_parts = _load_many(index_files, workers=args.workers)

    actions = _concat(actions_parts, empty=np.zeros((0, action_dim), dtype=np.int8))
    mouse = _concat(mouse_parts, empty=np.zeros((0, 2), dtype=np.float32))
    frame_indices = _concat(index_parts, empty=np.zeros((0,), dtype=np.int64))

    if len(actions) != len(mouse):
        raise SystemExit(f"Length mismatch after merge: actions={len(actions)} vs mouse={len(mouse)}")
    if len(frame_indices) not in (0, len(actions)):
        raise SystemExit(
            f"Length mismatch after merge: frame_indices={len(frame_indices)} vs actions={len(actions)}"
        )

    np.save(root / "actions.npy", actions.astype(np.int8, copy=False))
    np.save(root / "mouse_deltas.npy", mouse.astype(np.float32, copy=False))
    if len(frame_indices) > 0:
        np.save(root / "frame_indices.npy", frame_indices.astype(np.int64, copy=False))

    np.savez(
        root / "dataset.npz",
        actions=actions.astype(np.int8, copy=False),
        mouse_deltas=mouse.astype(np.float32, copy=False),
        **({"frame_indices": frame_indices.astype(np.int64, copy=False)} if len(frame_indices) > 0 else {}),
    )

    torch_info = "torch not installed; skipped dataset.pt"
    try:
        import torch  # type: ignore

        payload = {
            "actions": torch.tensor(actions, dtype=torch.int8),
            "mouse_deltas": torch.tensor(mouse, dtype=torch.float32),
        }
        if len(frame_indices) > 0:
            payload["frame_indices"] = torch.tensor(frame_indices, dtype=torch.int64)
        torch.save(payload, root / "dataset.pt")
        torch_info = "dataset.pt"
    except ImportError:
        pass

    meta["num_frames"] = int(actions.shape[0])
    meta["action_file"] = "actions.npy"
    meta["mouse_deltas_file"] = "mouse_deltas.npy"
    if len(frame_indices) > 0:
        meta["frame_indices_file"] = "frame_indices.npy"
    meta["torch_dataset"] = torch_info
    (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[+] Merged actions: {actions.shape} -> {root / 'actions.npy'}")
    print(f"[+] Merged mouse:   {mouse.shape} -> {root / 'mouse_deltas.npy'}")
    if len(frame_indices) > 0:
        print(f"[+] Merged indices: {frame_indices.shape} -> {root / 'frame_indices.npy'}")
    print(f"[+] NPZ:            {root / 'dataset.npz'}")
    print(f"[+] Torch:          {torch_info}")
    print(f"[+] Meta updated:   {root / 'meta.json'}")


if __name__ == "__main__":
    main()

