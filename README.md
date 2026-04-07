# GenTraj Visualization

Interactive 3D trajectory visualization website for browsing EgoDex, HD-EPIC, and Xperience manipulation clips.

## Features

- **Interactive 3D scene** (Three.js): dense point cloud, query point trajectories, camera frustum with video texture, floor grid
- **Synced RGB video** with 2D track overlay
- **Text annotations**: molmo2 recaptions (EgoDex), action captions (HD-EPIC), label + description (Xperience)
- **FPS-style navigation**: WASD to move, mouse drag to orbit, scroll to zoom, Q/E for up/down
- **Light/dark mode** toggle with localStorage persistence
- **Dataset tabs**: EgoDex (210 clips), HD-EPIC (27 clips), Xperience (30 clips)

## Data per clip

| Data | Source | Frames used |
|------|--------|-------------|
| RGB video | `rgb.mp4` | All (streamed) |
| Depth | `.zip` (EXR) or HDF5 | Frame 0 only (background point cloud) |
| Camera poses (c2w) | `pose.npz` / `poses.npz` | All (camera trail + frustum) |
| Intrinsics (fx,fy,cx,cy) | `intrinsics.npz` / K matrix | Frame 0 (constant) |
| 3D tracks | `tracks_3d.npz` / `filtered_tracks/` | All (N, T, 3) |
| 2D tracks | `tracks_2d.npz` | All (T, N, 2) |

All datasets use **OpenCV/COLMAP** coordinate convention with cam0 = identity (origin).

## Clip selection

| Dataset | Selection | Count |
|---------|-----------|-------|
| EgoDex | Clips with molmo2 recaptions, 3 per task | 210 |
| HD-EPIC | 3 random per participant (P01-P09) | 27 |
| Xperience | 1 per episode, 30 total | 30 |

## Quick start

```bash
pip install flask numpy
python app.py
# Open http://localhost:8888
```

With SSH tunnel:
```bash
ssh -L 8888:localhost:8888 <your-server>
# Then open http://localhost:8888
```

## Pre-processing (regenerate data)

The `static/data/` directory contains pre-processed `.npz` bundles. To regenerate from raw data:

```bash
pip install flask numpy opencv-python h5py
python preprocess.py
```

This requires access to the original data paths on weka.

## Architecture

```
viz_website/
  app.py              # Flask server — serves pages, video, and clip data as JSON
  preprocess.py        # Extracts point clouds, tracks, poses, annotations from raw data
  start.sh             # Convenience start script
  templates/
    index.html         # Full frontend: Three.js 3D, video player, tabs, light/dark mode
  static/data/
    manifest.json      # Index of all clips with metadata and text annotations
    egodex/            # Pre-processed .npz bundles (210 clips)
    hdepic/            # Pre-processed .npz bundles (27 clips)
    xperience/         # Pre-processed .npz bundles (30 clips)
```

## 3D viewer controls

| Control | Action |
|---------|--------|
| Mouse drag | Orbit rotate |
| Scroll | Zoom in/out |
| W / Arrow Up | Move forward |
| S / Arrow Down | Move backward |
| A / Arrow Left | Strafe left |
| D / Arrow Right | Strafe right |
| Q | Move down |
| E | Move up |
| Space | Play/pause video |
| Esc | Close viewer |

## Color scheme

- **2D trails**: light pastel rainbow (per-point identity)
- **2D dots**: muted rainbow (visible but not overpowering)
- **3D trails**: same pastel rainbow as 2D
- **3D dots**: RGB sampled from video frame at track position
- **Point cloud**: 50% transparent, real RGB colors from depth backprojection
- **Camera**: wireframe frustum with video texture + white trail
