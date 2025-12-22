<<<<<<< HEAD
# NitroGen Capture (Windows)
=======
# NitroGen Capture
>>>>>>> 7c45c4a55fdfee64f21c36eb63de4d26c702eab4

Frame + input logger for collecting training data from any Windows game and exporting it in a NitroGen-friendly format. Captures frames, keyboard/mouse buttons, and mouse deltas in lockstep, chunks files on disk, and can rebuild and convert the dataset to `.pt`.

## What it does
- Locks frame capture to a target FPS using DXCam.
- Tracks keyboard keys, mouse buttons, and mouse deltas per frame.
- Flushes data into chunked PNG folders plus aligned numpy arrays.
- Writes integrity metadata (frame indices) to detect drift.
- Re-merges chunks and optionally converts to NitroGen-style 20D action tensors.

## Requirements
- Windows with a GPU supported by DXCam.
- Python 3.9+.
- Packages from `requirements.txt`:
  - `dxcam`, `pynput`, `numpy`, `opencv-python`, `psutil`, `pywin32`
  - `torch` (optional; enables `.pt` exports)

## Install
```bash
pip install -r requirements.txt
```

If you do not need torch outputs, you can remove `torch` from the requirements file.

## Quickstart capture
1) Launch the game and bring it to the foreground.
2) Choose a capture mode:
   - Auto-detect window by process name:
     ```bash
     python capture_logger.py --out dataset/session1 --fps 30 --process-name Game.exe --chunk-size 800
     ```
   - Manual region (left top right bottom):
     ```bash
     python capture_logger.py --out dataset/session1 --fps 30 --region 0 0 1920 1080 --chunk-size 800
     ```
   - List visible windows if detection fails:
     ```bash
     python capture_logger.py --debug-list-windows
     ```

Flags you will likely tweak:
- `--fps`: target capture FPS (default 30).
- `--chunk-size`: frames per chunk directory before flushing (default 500).
- `--max-frames`: hard stop after N frames (0 = unlimited; Ctrl+C also stops).
- `--skip-late`: drop a frame if scheduling slips to avoid drift.
- `--start-immediately`: start recording without waiting for a hotkey.

### Hotkeys during capture
- `F8` toggles pause/resume (default start paused unless `--start-immediately`).
- `F6` force resume; `F7` pause.
- `Ctrl+T` resume; `Ctrl+P` pause.

### Outputs
After capture you will have:
```
dataset/session1/
  frames/
    chunk_001/
      000001.png
      000002.png
      ...
  actions_chunk_001.npy
  mouse_deltas_chunk_001.npy
  frame_indices_chunk_001.npy
  actions.npy
  mouse_deltas.npy
  frame_indices.npy
  dataset.npz
  dataset.pt          # if torch installed
  meta.json
```

- PNGs are saved in chunk folders to keep directories manageable.
- `actions.npy` shape: `(num_frames, 28)` boolean/int8.
- `mouse_deltas.npy` shape: `(num_frames, 2)` (dx, dy) in pixels per frame.
- `frame_indices.npy` keeps contiguous IDs to catch dropped frames.
- `dataset.npz` mirrors the numpy outputs; `dataset.pt` mirrors them as torch tensors.
- `meta.json` tracks chunk boundaries, FPS, capture region/monitor, action layout, and file names.

#### Action vector layout (index order)
| idx | action      | notes                        |
| --- | ----------- | ---------------------------- |
| 0   | w           | move forward                 |
| 1   | a           | move left                    |
| 2   | s           | move back                    |
| 3   | d           | move right                   |
| 4   | space       | jump/evade                   |
| 5   | shift       | sprint/modifier              |
| 6   | ctrl        | crouch/modifier              |
| 7   | e           | interact                     |
| 8   | q           | ability/use                 |
| 9   | r           | reload/use                  |
| 10  | f           | melee/use                   |
| 11  | g           | ability/use                 |
| 12  | c           | stance/use                  |
| 13  | v           | stance/use                  |
| 14  | x           | custom                       |
| 15  | z           | custom                       |
| 16  | t           | capture resume hotkey (Ctrl) |
| 17  | p           | capture pause hotkey (Ctrl)  |
| 18  | i           | custom                       |
| 19  | backspace   | custom/system                |
| 20  | esc         | menu/pause                   |
| 21  | 1           | hotbar 1                     |
| 22  | 2           | hotbar 2                     |
| 23  | 3           | hotbar 3                     |
| 24  | 4           | hotbar 4                     |
| 25  | enter       | confirm                      |
| 26  | lmb         | left mouse button            |
| 27  | rmb         | right mouse button           |

## Rebuilding merged outputs
If you delete merged files or want to re-merge after a partial run:
```bash
python dataset_merge.py --root dataset/session1
```
This concatenates all `actions_chunk_*.npy` / `mouse_deltas_chunk_*.npy`, regenerates `dataset.npz`/`.pt`, and refreshes `meta.json`.

## Convert to NitroGen format
`convert_to_nitrogen.py` remaps the capture into a 20D gamepad-style action space and produces a single training-ready `.pt`.

Basic usage:
```bash
python convert_to_nitrogen.py --root dataset/session1 --out dataset/session1_nitro.pt
```

Tunables:
- `--img-size 256`: resize frames to `img_size x img_size` (RGB CHW, float32 0-1).
- `--seq-len 16`: sliding sequence length for actions (produces M = N - T + 1 samples).
- `--mouse-scale 300 --mouse-clip 1.0`: map mouse dx/dy into [-1, 1] right-stick values.

Output structure:
- `obs`: `(M, 3, H, W)` torch float tensor.
- `actions`: `(M, T, 20)` torch float tensor (gamepad-like mapping from WASD/mouse/buttons).
- `meta`: conversion parameters and a short layout description.

## Tips
- Keep FPS stable; use `--skip-late` if your machine occasionally stutters.
- Start with a short test run, then open a few PNGs and compare with actions to verify alignment.
- Chunk long sessions (`--chunk-size`) so individual folders stay small and fast to inspect.
- If window detection fails, use `--debug-list-windows` to find the exact title/PID or fall back to `--region`.
