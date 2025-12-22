# NitroGen Capture (Windows)

Frame + input logger for collecting training data from any Windows game. Captures frames at a target FPS with DXCam, records keyboard/mouse buttons plus mouse deltas per frame, writes chunked PNGs and aligned numpy logs, then merges them at the end. A converter is included to build a NitroGen-style `.pt` with 20D actions and resized frames.

## Requirements
- Windows with a DXCam-supported GPU.
- Python 3.11+.
- Install deps: `pip install -r requirements.txt`  
  (`dxcam`, `pynput`, `numpy`, `opencv-python`, `psutil`, `pywin32`; `torch` optional for `.pt` exports).

## Capture usage (`capture_logger.py`)
Capture starts **paused** unless `--start-immediately` is set. Use hotkeys to control recording.

Common commands:
```bash
# Auto-detect by process name (exact match)
python capture_logger.py --out dataset/session1 --fps 30 --process-name Game.exe --chunk-size 800

# Manual region (left top right bottom)
python capture_logger.py --out dataset/session1 --fps 30 --region 0 0 1920 1080 --chunk-size 800

# Downscale frames to save disk/IO (example: 720p output)
python capture_logger.py --out dataset/session1 --process-name Game.exe --resize 1280 720

# If detection fails, list visible windows/PIDs then exit
python capture_logger.py --debug-list-windows
```

Notable flags:
- `--fps` (default 30): target capture FPS.
- `--chunk-size` (default 500): frames per chunk before flushing PNGs/logs.
- `--max-frames` (default 0): stop after N frames (0 = unlimited; Ctrl+C also stops).
- `--skip-late`: drop a frame if the loop is late to avoid drift.
- `--start-immediately`: begin recording right away (otherwise start paused).
- `--status-interval` (default 2s): status print cadence (0 = off).
- `--resize W H`: resize each captured frame before saving (e.g., `1280 720` for 720p) to shrink disk use and speed up conversion/training.

Hotkeys during capture:
- `F8` toggle pause/resume.
- `F6` force resume; `F7` pause.
- `Ctrl+T` resume; `Ctrl+P` pause (requires `keyboard` module).

## Outputs (after capture + automatic merge)
Capture flushes chunk files continuously and, on exit, merges them into consolidated files.

```
dataset/session1/
  frames/
    chunk_001/
      000001.png
      000002.png
      ...
  actions_chunk_001.npy          # chunked logs
  mouse_deltas_chunk_001.npy
  frame_indices_chunk_001.npy
  actions.npy                    # merged
  mouse_deltas.npy               # merged
  frame_indices.npy              # merged, contiguous check enforced
  dataset.npz                    # merged numpy bundle
  dataset.pt                     # merged torch bundle (if torch installed)
  meta.json                      # capture metadata and chunk manifest
```

Details:
- PNGs are saved via OpenCV (BGR), optionally resized if `--resize` is set. Frames are chunked to keep folders small.
- `actions.npy`: shape `(num_frames, 28)`, dtype `int8` (0/1 per action).
- `mouse_deltas.npy`: shape `(num_frames, 2)`, dtype `float32` (dx, dy in pixels/frame).
- `frame_indices.npy`: contiguous indices; non-contiguous raises an error at merge.
- `dataset.npz`: capture run includes actions, mouse deltas, and frame indices. Rebuilt via `dataset_merge.py` only contains actions + mouse.
- `meta.json` includes FPS, region/monitor, action layout, chunk list, recorded frame size, and file names. If resizing was applied, `frame_size`/`resize_target` show the final output dimensions.

Action vector layout (28-D, index order):
| idx | action    | notes                              |
| --- | --------- | ---------------------------------- |
| 0   | w         | move forward                       |
| 1   | a         | move left                          |
| 2   | s         | move back                          |
| 3   | d         | move right                         |
| 4   | space     | jump/evade                         |
| 5   | shift     | sprint/modifier                    |
| 6   | ctrl      | crouch/modifier                    |
| 7   | e         | interact                           |
| 8   | q         | action/ability                     |
| 9   | r         | action/ability                     |
| 10  | f         | action/ability                     |
| 11  | g         | action/ability                     |
| 12  | c         | stance/use                         |
| 13  | v         | stance/use                         |
| 14  | x         | custom                             |
| 15  | z         | custom                             |
| 16  | t         | capture resume hotkey (Ctrl+T)     |
| 17  | p         | capture pause hotkey (Ctrl+P)      |
| 18  | i         | custom                             |
| 19  | backspace | custom/system                      |
| 20  | esc       | menu/pause                         |
| 21  | 1         | hotbar 1                           |
| 22  | 2         | hotbar 2                           |
| 23  | 3         | hotbar 3                           |
| 24  | 4         | hotbar 4                           |
| 25  | enter     | confirm                            |
| 26  | lmb       | left mouse button                  |
| 27  | rmb       | right mouse button                 |

## Re-merge chunk outputs manually
If you delete merged files or need to rebuild:
```bash
python dataset_merge.py --root dataset/session1
```
This concatenates `actions_chunk_*.npy` / `mouse_deltas_chunk_*.npy`, regenerates `dataset.npz`/`.pt` (actions + mouse only), and updates `meta.json`. It does **not** rebuild `frame_indices.npy`; keep the originals from capture for conversion. Chunk PNGs must exist if you plan to run the converter.

## Convert to NitroGen format
`convert_to_nitrogen.py` reads `meta.json`, walks the chunk list to load frames in order, maps the 28-D actions to a 20-D gamepad-like space, and emits a single training-ready `.pt`.

```bash
python convert_to_nitrogen.py --root dataset/gow --out dataset/gow_nitro.pt
```

Key options:
- `--img-size 256`: resize frames to square RGB (CHW float32 in [0,1]). Use higher/lower values (e.g., 640 or 720) to control training resolution.
- `--seq-len 18`: sliding window length for actions (produces `M = N - T + 1` samples).
- `--mouse-scale 300 --mouse-clip 1.0`: scale/clip mouse dx/dy into [-1, 1] right-stick values.
- `--action-dim 20|25`: output action dimension; set to `25` to match checkpoints expecting `action_dim=25` (default 20).
- `--workers 0`: thread count for image decode/resize (0 = auto, 1 = sequential). Increase if you want faster conversion on multi-core CPUs.

Converter outputs:
- `obs`: `(M, 3, H, W)` torch float tensor (frames).
- `actions`: `(M, T, 20)` torch float tensor (20-D mapping from WASD/mouse/buttons).
- `meta`: conversion parameters and a brief layout note.

## Tips
- Run a short test first; open a few PNGs and check the same rows in `actions.npy`/`mouse_deltas.npy` for alignment.
- Use `--skip-late` or lower `--fps` if you see timing drift; keep chunk sizes reasonable for fast directory access.
- If auto-detection fails, try `--debug-list-windows` to get the title/PID or fall back to `--region`.
