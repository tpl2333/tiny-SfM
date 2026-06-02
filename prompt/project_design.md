# tiny-SfM Project Design

## Project Positioning

This project is an educational / research prototype of an incremental unordered
Structure-from-Motion pipeline. It is not intended to be a COLMAP-level
production system, but it implements the main SfM data flow end to end:

```text
input images
  -> feature extraction
  -> exhaustive pairwise matching
  -> two-view geometric verification
  -> ViewGraph
  -> multi-view TrackManager
  -> seed-pair initialization
  -> incremental PnP registration
  -> incremental triangulation
  -> Ceres bundle adjustment
  -> COLMAP-compatible sparse model export
  -> 3DGS scene packaging
```

The main value of the project is the implementation of the SfM data structures,
state management, and reconstruction pipeline, rather than achieving COLMAP-level
robustness.

## Main Entry

Current reconstruction entry:

```text
incremental_unordered.py
```

The legacy temporal prototype is kept at:

```text
pipeline/reconstruct.py
```

It is not the main path.

## Core Modules

### `management/viewgraph.py`

`ViewGraph` stores verified image-pair relations.

Each edge is keyed by normalized frame ids:

```python
(min_frame_id, max_frame_id) -> EdgeData
```

It also stores an adjacency table:

```python
frame_id -> set(neighbor_frame_ids)
```

Main use cases:

- seed-pair selection
- checking whether an unregistered frame is connected to the current map
- finding registered neighbors for incremental triangulation

Conceptually, this is a simplified in-memory version of the verified image-pair
graph represented by COLMAP's two-view geometries.

### `management/trackmanager.py`

`TrackManager` builds global multi-view feature tracks from pairwise matches.

Each 2D observation is represented as:

```python
(frame_idx, feature_idx)
```

Pairwise matches are treated as equivalence constraints and merged by union-find.
Each connected component becomes a `FeatureTrack`.

Important invariants:

- one track may contain multiple frame observations
- one valid track may contain at most one feature per frame
- a triangulated track stores `mappoint_idx`

Main methods:

- `build_from_viewgraph()`: build tracks from all view graph edges
- `get_track_from_feat()`: query the track for a 2D observation
- `get_2d_3d_pairs()`: collect PnP correspondences for a frame
- `classify_matches()`: split matches into already-triangulated tracks and new triangulation candidates
- `update_track_state()`: bind newly created 3D points back to tracks
- `reset_track_state()`: allow a deleted bad point's track to be triangulated again

This module is close in spirit to openMVG's track building logic, though much
simpler.

### `management/worldmap.py`

`Map` owns:

- camera intrinsics
- frames
- map points
- point-to-track mapping
- registration state

Frame states:

```text
registered: successfully added to the reconstruction
active/unregistered: normal candidate for registration
deferred: failed before, but can be retried after the map grows
failed: permanently abandoned after repeated failures
```

The deferred resurrection mechanism prevents early PnP failures from permanently
removing frames. A deferred frame is retried only after the map has grown since
its last failure.

`save_as_colmap(output_dir, track_manager)` exports only registered frames and
current map points to COLMAP text format:

```text
cameras.txt
images.txt
points3D.txt
```

It does not copy images and does not create a full 3DGS scene directory.

### `algorithm/match.py`

`FeatureMatcher` handles:

- SIFT or ORB feature extraction
- BF matching
- Lowe ratio test
- one-to-one match filtering
- H/F RANSAC geometric verification
- simple GRIC-like H/F model selection
- exhaustive matching into `ViewGraph`

High-frequency matching diagnostics are logged at debug level.

### `algorithm/datamine.py`

`DataMiner` is currently a lightweight heuristic scheduler.

Seed-pair selection:

```text
score = log10(num_inliers) * (GRIC_H / GRIC_F) * spatial_spread
```

Only F-model and good edges are considered.

Next-frame selection:

```text
score = log10(num_2d_3d_correspondences) * spatial_spread
```

Selection order:

1. active unregistered frames
2. deferred frames that are ready for retry

Limitations:

- seed selection does not actually test recoverPose + triangulation
- next-frame selection does not estimate PnP inlier ratio in advance
- it is greedy and not COLMAP-level

This is one of the main areas for future improvement.

### `algorithm/mvgsolver.py`

`MvgSolver` contains multi-view geometry routines:

- essential matrix / recoverPose initialization
- triangulation
- reprojection error calculation
- parallax calculation
- multi-view consensus check
- PnP RANSAC + iterative refinement

Pose convention:

```text
X_camera = R * X_world + t
```

### `algorithm/ba_ceres.py` and `src/ba_core.cpp`

`BundleAdjuster` wraps the C++ Ceres backend.

Camera parameter layout:

```text
[angle_axis_rotation(3), translation(3)]
```

The backend currently uses:

- SIMPLE_PINHOLE
- one shared focal length
- fixed principal point
- Huber loss
- Ceres Schur solver

Ceres `BriefReport()` is returned to Python and logged through `logger`.

### `tools/export_3dgs_scene.py`

This is the adapter from SfM output to 3DGS scene format.

Input:

```text
--image-dir   original image directory
--colmap-dir  directory containing cameras.txt, images.txt, points3D.txt
--output-dir  target 3DGS scene directory
```

Output:

```text
output_dir/
  images/
  sparse/0/
    cameras.txt
    images.txt
    points3D.txt
  registered_images.txt
  reconstruction_stats.json
```

The script reads `images.txt`, copies only registered images, and packages the
COLMAP text model into the standard 3DGS directory structure.

## Current Known Limitations

- Exhaustive matching is expensive: `O(N^2)`.
- `DataMiner` is heuristic and relatively shallow.
- Track and point maintenance is much simpler than COLMAP.
- No track completion or duplicate point merging.
- BA can report `NO_CONVERGENCE` when weak resurrected frames or unstable points enter the map.
- Current camera model assumes shared SIMPLE_PINHOLE intrinsics.
- `points3D.txt` uses a placeholder reprojection error value.

## Current 3DGS Plan

The intended integration is loose coupling:

```text
tiny-SfM exports a COLMAP-compatible sparse model
tools/export_3dgs_scene.py packages it as a 3DGS scene
external gaussian-splatting repo trains from that scene
```

Do not merge the 3DGS source code into this repository at this stage.

