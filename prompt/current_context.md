# Current Conversation Context

## Current Date / Environment

- Date in conversation: 2026-06-02
- Main project path: `E:/code/py_cpp_SfM/tiny-SfM`
- Main reconstruction entry: `incremental_unordered.py`
- Current target dataset discussed: Blender synthetic `chair/train`

## Recent Work Completed

### Comment and Logging Cleanup

All project comments were rewritten into a more consistent maintenance-oriented
style. Chatty comments, temporary descriptions, and stale blocks were removed.

Runtime output was standardized:

- Python `print(...)` was replaced by module-level `logger`.
- High-frequency matching diagnostics were moved to debug logging.
- `eval.py` still prints readable reports through logging.
- C++ BA backend no longer uses `std::cout`.
- Ceres `BriefReport()` is returned to Python and logged.

### Color Bug Fixes

Fixed color sampling and storage:

- `Frame.get_color(u, v)` now indexes OpenCV images as `img[v, u]`.
- Color sampling clips pixel coordinates to image bounds.
- Point colors are normalized to RGB float `[0, 1]`.
- `points3D.txt` export clips RGB values to `[0, 255]`.
- Incremental triangulation fallback color now also converts BGR to RGB `[0,1]`.

### Failed Frame Resurrection

Added a deferred frame mechanism.

Frame states:

```text
registered
active/unregistered
deferred
failed
```

Behavior:

- PnP failure calls `worldmap.defer_frame(frame_idx)` instead of permanently failing immediately.
- Deferred frames are retried only when the map has grown since their last failure.
- Deferred frames must pass a stricter PnP inlier threshold (`>= 15`) before being registered.
- Frames are permanently failed after `max_register_attempts = 3`.

Purpose:

- Improve registration rate without repeatedly retrying the same weak frame in
  an unchanged map state.

### BA Summary Logging

The Ceres backend now returns `summary.BriefReport()`:

```cpp
return summary.BriefReport();
```

Python logs it through `BundleAdjuster`.

The C++ extension was rebuilt successfully:

```text
build/Release/ba_core.cp312-win_amd64.pyd
```

### 3DGS Export Adapter

Added:

```text
tools/export_3dgs_scene.py
```

It packages `save_as_colmap()` output into a 3DGS-compatible scene directory.

Example command already tested:

```powershell
python tools/export_3dgs_scene.py `
  --image-dir E:\dataset\nerf_synthetic\nerf_synthetic\chair\train `
  --colmap-dir output\synthetic\train\chair `
  --output-dir output\3dgs\chair_self_sfm
```

Observed result:

```text
78/78 images copied, 1811 points
```

Generated stats:

```json
{
  "num_registered_images": 78,
  "num_copied_images": 78,
  "num_missing_images": 0,
  "num_points3d": 1811
}
```

## Current SfM Status

On Blender synthetic `chair/train` with 100 images:

- Registration rate before resurrection was around `75/100`.
- With resurrection, registration improved to around `80/100` in one run.
- More conservative resurrection produced an exported scene with `78` registered images.
- BA sometimes reports `NO_CONVERGENCE`, especially when weak resurrected frames enter.
- Visual reconstruction can still look acceptable despite BA not fully converging.

Interpretation:

- The pipeline is not catastrophically broken.
- ViewGraph, TrackManager, PnP, triangulation, and BA all work on a meaningful subset.
- Late-stage bottleneck is mainly insufficient 2D-3D correspondences for remaining frames.

## Important Clarification

`worldmap.save_as_colmap()` only saves registered frames.

Evidence:

```python
for f_idx in sorted(list(self._registered_ids)):
```

Therefore:

- `images.txt` contains only registered frames.
- `points3D.txt` observations are also filtered to registered frames.
- `tools/export_3dgs_scene.py` reads `images.txt` and copies only those registered images.

## Recommended Next Step

Go to the server and train 3DGS using the exported scene:

```text
output/3dgs/chair_self_sfm
```

Expected 3DGS directory layout:

```text
output/3dgs/chair_self_sfm/
  images/
  sparse/0/
    cameras.txt
    images.txt
    points3D.txt
  registered_images.txt
  reconstruction_stats.json
```

Use the external `gaussian-splatting` repository. Do not merge that source into
this repo yet.

Typical training command pattern:

```powershell
cd <gaussian-splatting-repo>
python train.py -s E:\code\py_cpp_SfM\tiny-SfM\output\3dgs\chair_self_sfm
```

On Linux/server, adjust the path accordingly.

## How To Restore Context In A New Session

Ask the assistant to read:

```text
prompt/project_design.md
prompt/current_context.md
```

Then continue from there.

## Current Project Positioning

Best project description:

> A self-built incremental SfM prototype that implements ViewGraph, multi-view
> TrackManager, Map state management, PnP registration, triangulation, Ceres BA,
> COLMAP-compatible export, and a 3DGS scene packaging adapter.

Do not claim:

> COLMAP-level robust SfM implementation.

Better phrasing:

> The project implements the core SfM data flow and uses lightweight heuristics
> for seed and next-frame selection. It is suitable as an educational/research
> prototype and a front-end initialization experiment for 3D Gaussian Splatting.

