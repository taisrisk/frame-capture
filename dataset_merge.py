"""
Merge chunked capture outputs into consolidated numpy/torch files.

Use this if you recorded with `capture_logger.py` and want to rebuild
`actions.npy`, `mouse_deltas.npy`, `dataset.npz`, and optionally `dataset.pt`.

Usage:
    python dataset_merge.py --root dataset/gow

Credits:
- zrorisc
"""

from __future__ import annotations

import argparse
import json
import concurrent.futures
from pathlib import Path
from typing import List

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
    return json.loads(meta_path.read_text())


def merge_arrays(root: Path, pattern: str) -> List[Path]:
    files = sorted(root.glob(pattern))
    return list(files)


def main() -> None:
    args = parse_args()
    root: Path = args.root

    meta = load_meta(root)
    action_files = merge_arrays(root, "actions_chunk_*.npy")
    mouse_files = merge_arrays(root, "mouse_deltas_chunk_*.npy")

    def load_files(files: List[Path], empty_shape) -> np.ndarray:
        if not files:
            return np.zeros(empty_shape, dtype=np.int8 if len(empty_shape) == 1 or empty_shape[1] == 1 else np.float32)
        if args.workers == 1:
            parts = [np.load(f) for f in files]
        else:
            workers = None if args.workers == 0 else args.workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                parts = list(ex.map(np.load, files))
        return np.concatenate(parts, axis=0)

    actions = load_files(action_files, (0, len(load_meta(root).get("actions", [])) or 1))
    mouse = load_files(mouse_files, (0, 2))

    np.save(root / "actions.npy", actions)
    np.save(root / "mouse_deltas.npy", mouse)
    np.savez(root / "dataset.npz", actions=actions, mouse_deltas=mouse)

    torch_info = "torch not installed; skipped dataset.pt"
    try:
        import torch  # type: ignore

        torch.save(
            {"actions": torch.tensor(actions, dtype=torch.int8), "mouse_deltas": torch.tensor(mouse, dtype=torch.float32)},
            root / "dataset.pt",
        )
        torch_info = "dataset.pt"
    except ImportError:
        pass

    meta["num_frames"] = int(actions.shape[0])
    meta["action_file"] = "actions.npy"
    meta["mouse_deltas_file"] = "mouse_deltas.npy"
    meta["torch_dataset"] = torch_info
    (root / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[+] Merged actions: {actions.shape} -> {root / 'actions.npy'}")
    print(f"[+] Merged mouse:   {mouse.shape} -> {root / 'mouse_deltas.npy'}")
    print(f"[+] NPZ:            {root / 'dataset.npz'}")
    print(f"[+] Torch:          {torch_info}")
    print(f"[+] Meta updated:   {root / 'meta.json'}")


if __name__ == "__main__":
    main()