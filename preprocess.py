#!/usr/bin/env python3
"""
Pre-process clips for the visualization website.

For each selected clip, extracts:
  - Subsampled point cloud (pos + color, float16/uint8)
  - Camera poses per frame (position + rotation matrix)
  - 3D tracks + visibility
  - 2D tracks + visibility
  - Text annotation
  - Video path

Saves compact .npz bundles to viz_website/static/data/<dataset>/<clip_id>.npz
Also saves a manifest.json listing all clips with metadata.

Selection:
  - EgoDex: 3 random clips per task
  - HD-EPIC: 3 random clips per participant (P01-P09)
  - Xperience: 30 clips total, 1 per episode randomly
"""

import json
import os
import random
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FINAL_TRACKS = Path("/weka/prior-default/jianingz/home/project/_GenTraj/vipe/final_tracks")
VIPE_RESULTS = Path("/weka/prior-default/jianingz/home/project/_GenTraj/vipe/vipe_results")
XPERIENCE_DIR = Path("/weka/oe-training-default/chenhaoz/xperience")
XPERIENCE_RAW = Path("/weka/oe-training-default/chrisk/data/xperience-10m")
TMP_DIR = Path("/weka/prior-default/jianingz/home/project/_GenTraj/tmp")
OUTPUT_DIR = Path("/weka/prior-default/jianingz/home/project/_GenTraj/viz_website/static/data")

# Point cloud budget
MAX_PC_POINTS = 25000   # keep point cloud small for browser
MAX_TRACK_POINTS = 150  # max query points
PC_SUBSAMPLE = 4        # pixel stride for depth backprojection


# ---------------------------------------------------------------------------
# EXR depth
# ---------------------------------------------------------------------------
def load_exr_depth(raw_bytes):
    import OpenEXR
    import Imath
    with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as f:
        f.write(raw_bytes)
        fname = f.name
    try:
        exr = OpenEXR.InputFile(fname)
        dw = exr.header()['dataWindow']
        W = dw.max.x - dw.min.x + 1
        H = dw.max.y - dw.min.y + 1
        buf = exr.channel('Z', Imath.PixelType(Imath.PixelType.FLOAT))
        depth = np.frombuffer(buf, dtype=np.float32).reshape(H, W).copy()
        exr.close()
    finally:
        os.unlink(fname)
    return depth


# ---------------------------------------------------------------------------
# Point cloud from depth
# ---------------------------------------------------------------------------
def build_point_cloud(depth, rgb, c2w, intrinsics, subsample=PC_SUBSAMPLE):
    H, W = depth.shape
    fx, fy, cx, cy = intrinsics
    us = np.arange(0, W, subsample, dtype=np.int32)
    vs = np.arange(0, H, subsample, dtype=np.int32)
    uu, vv = np.meshgrid(us, vs)
    uu, vv = uu.ravel(), vv.ravel()
    z = depth[vv, uu]
    valid = (z > 0) & np.isfinite(z)
    uu, vv, z = uu[valid], vv[valid], z[valid]
    xc = (uu.astype(np.float32) - cx) / fx * z
    yc = (vv.astype(np.float32) - cy) / fy * z
    pts_cam = np.stack([xc, yc, z, np.ones_like(z)], axis=1)
    xyz = (c2w @ pts_cam.T).T[:, :3].astype(np.float32)
    colors = rgb[vv, uu]
    # Further subsample if too many
    if len(xyz) > MAX_PC_POINTS:
        idx = np.random.choice(len(xyz), MAX_PC_POINTS, replace=False)
        xyz = xyz[idx]
        colors = colors[idx]
    return xyz, colors


# ---------------------------------------------------------------------------
# Load data for EgoDex / HD-EPIC clips
# ---------------------------------------------------------------------------
def process_ft_clip(clip_id):
    """Process a final_tracks clip (EgoDex or HD-EPIC)."""
    # Check all required files exist
    rgb_path = VIPE_RESULTS / "rgb" / f"{clip_id}.mp4"
    depth_path = VIPE_RESULTS / "depth" / f"{clip_id}.zip"
    pose_path = VIPE_RESULTS / "pose" / f"{clip_id}.npz"
    intr_path = VIPE_RESULTS / "intrinsics" / f"{clip_id}.npz"
    t2d_path = FINAL_TRACKS / f"{clip_id}_2d.npz"
    t3d_path = FINAL_TRACKS / f"{clip_id}_3d.npz"

    for p in [rgb_path, depth_path, pose_path, intr_path, t2d_path, t3d_path]:
        if not p.exists():
            print(f"  SKIP {clip_id}: missing {p.name}")
            return None

    # Load poses (all frames)
    poses = np.load(pose_path)['data'].astype(np.float32)  # (T, 4, 4)
    intrinsics = np.load(intr_path)['data'][0].astype(np.float32)  # (4,)

    # Frame 0 RGB + depth for point cloud
    cap = cv2.VideoCapture(str(rgb_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  SKIP {clip_id}: can't read video")
        return None
    rgb0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with zipfile.ZipFile(depth_path) as zf:
        depth = load_exr_depth(zf.read(sorted(zf.namelist())[0]))

    # Build point cloud
    xyz, pc_colors = build_point_cloud(depth, rgb0, poses[0], intrinsics)

    # Load tracks
    t3 = np.load(t3d_path, allow_pickle=True)
    t2 = np.load(t2d_path, allow_pickle=True)

    pts_raw = t3['points_3d']
    is_dict = (pts_raw.dtype == object and pts_raw.shape == ())

    if is_dict:
        pts_dict = pts_raw[()]
        tr2_dict = t2['tracks'][()]
        vis2_dict = t2['visibility'][()]
        all_p3, all_v3, all_t2, all_v2 = [], [], [], []
        for obj in pts_dict:
            p3 = pts_dict[obj].astype(np.float32)
            t2d = tr2_dict[obj].astype(np.float32)
            v2 = vis2_dict[obj].astype(bool)
            all_p3.append(p3)
            all_v3.append(v2.T)
            all_t2.append(t2d)
            all_v2.append(v2)
        pts3d = np.concatenate(all_p3, axis=0)
        vis3d = np.concatenate(all_v3, axis=0)
        tracks2d = np.concatenate(all_t2, axis=1)
        vis2d = np.concatenate(all_v2, axis=1)
    else:
        pts3d = pts_raw.astype(np.float32)
        vis3d = t3['visibility'].squeeze(-1).astype(bool)
        tracks2d = t2['tracks'].astype(np.float32)
        vis2d = t2['visibility'].astype(bool)

    # Subsample tracks
    N = pts3d.shape[0]
    if N > MAX_TRACK_POINTS:
        idx = np.linspace(0, N - 1, MAX_TRACK_POINTS, dtype=int)
        pts3d = pts3d[idx]
        vis3d = vis3d[idx]
        tracks2d = tracks2d[:, idx]
        vis2d = vis2d[:, idx]

    # Camera positions for trail (T, 3) and forward vectors (T, 3)
    cam_pos = poses[:, :3, 3].astype(np.float32)   # (T, 3)
    cam_fwd = poses[:, :3, 2].astype(np.float32)   # (T, 3) — Z axis

    # Video dimensions
    H_img, W_img = rgb0.shape[:2]
    dim = [H_img, W_img]

    return {
        'pc_xyz': xyz.astype(np.float16),
        'pc_colors': pc_colors,
        'cam_pos': cam_pos.astype(np.float16),
        'cam_fwd': cam_fwd.astype(np.float16),
        'cam0_c2w': poses[0].astype(np.float32),
        'intrinsics': intrinsics,
        'pts3d': pts3d.astype(np.float16),
        'vis3d': vis3d,
        'tracks2d': tracks2d.astype(np.float16),
        'vis2d': vis2d,
        'dim': np.array(dim, dtype=np.int32),
        'rgb_path': str(rgb_path),
    }


# ---------------------------------------------------------------------------
# Load data for Xperience clips
# ---------------------------------------------------------------------------
def process_xp_clip(clip_rel):
    """Process a Xperience clip — same format as EgoDex/HD-EPIC with point cloud."""
    clip_path = XPERIENCE_DIR / clip_rel
    rgb_path = clip_path / "rgb.mp4"
    ft_file = clip_path / "filtered_tracks" / "object.npz"
    t2d_file = clip_path / "tracks_2d" / "object.npz"
    poses_file = clip_path / "poses.npz"

    for p in [rgb_path, ft_file, t2d_file]:
        if not p.exists():
            print(f"  SKIP {clip_rel}: missing {p.name}")
            return None

    # Load 2D tracks
    t2d_data = np.load(t2d_file, allow_pickle=True)
    dim = t2d_data['dim'].tolist()
    tracks2d_raw = t2d_data['tracks']     # (N, T, 2)
    vis2d_raw = t2d_data['visibility']    # (N, T)

    # Load 3D tracks
    ft = np.load(ft_file, allow_pickle=True)
    pts3d = ft['points_3d'].astype(np.float32)  # (N, T, 3)
    vis3d = ft.get('visibility', vis2d_raw).astype(bool)

    # Transpose 2D tracks to (T, N, 2) for consistency
    tracks2d = np.transpose(tracks2d_raw, (1, 0, 2)).astype(np.float32)  # (T, N, 2)
    vis2d = vis2d_raw.T.astype(bool)  # (T, N)

    # Subsample tracks
    N = pts3d.shape[0]
    if N > MAX_TRACK_POINTS:
        idx = np.linspace(0, N - 1, MAX_TRACK_POINTS, dtype=int)
        pts3d = pts3d[idx]
        vis3d = vis3d[idx]
        tracks2d = tracks2d[:, idx]
        vis2d = vis2d[:, idx]

    # Camera poses
    if not poses_file.exists():
        print(f"  SKIP {clip_rel}: missing poses.npz")
        return None
    pdata = np.load(poses_file, allow_pickle=True)
    poses = pdata['c2w'].astype(np.float32)  # (T, 4, 4)
    K_mat = pdata['K'].astype(np.float32)     # (3, 3)
    intrinsics = np.array([K_mat[0,0], K_mat[1,1], K_mat[0,2], K_mat[1,2]], dtype=np.float32)

    cam_pos = poses[:, :3, 3].astype(np.float32)  # (T, 3)
    cam_fwd = poses[:, :3, 2].astype(np.float32)  # (T, 3)
    cam0_c2w = poses[0]

    # Build point cloud from depth (HDF5 in xperience-10m)
    parts = clip_rel.split("/")
    uuid, ep = parts[0], parts[1]
    hdf5_path = XPERIENCE_RAW / uuid / ep / "annotation.hdf5"

    xyz, pc_colors = None, None
    if hdf5_path.exists():
        try:
            import h5py
            frame_indices = np.load(clip_path / "frame_indices.npy")
            fi0 = frame_indices[0]

            with h5py.File(hdf5_path, 'r') as hf:
                depth = hf['depth/depth'][fi0].astype(np.float32)  # (256, 256)

            # Depth intrinsics are for 256x256 resolution
            depth_H, depth_W = depth.shape
            # Scale K from clip resolution to depth resolution
            depth_fx = intrinsics[0] * depth_W / dim[1]
            depth_fy = intrinsics[1] * depth_H / dim[0]
            depth_cx = intrinsics[2] * depth_W / dim[1]
            depth_cy = intrinsics[3] * depth_H / dim[0]

            # Frame 0 RGB for colors
            cap = cv2.VideoCapture(str(rgb_path))
            ret, frame = cap.read()
            cap.release()
            if ret:
                rgb0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize RGB to depth resolution for color sampling
                rgb_depth = cv2.resize(rgb0, (depth_W, depth_H))

                # Backproject
                subsample = 2  # depth is already low-res
                us = np.arange(0, depth_W, subsample, dtype=np.int32)
                vs = np.arange(0, depth_H, subsample, dtype=np.int32)
                uu, vv = np.meshgrid(us, vs)
                uu, vv = uu.ravel(), vv.ravel()
                z = depth[vv, uu]
                valid = (z > 0) & np.isfinite(z) & (z < 10.0)
                uu, vv, z = uu[valid], vv[valid], z[valid]

                xc = (uu.astype(np.float32) - depth_cx) / depth_fx * z
                yc = (vv.astype(np.float32) - depth_cy) / depth_fy * z
                pts_cam = np.stack([xc, yc, z, np.ones_like(z)], axis=1)
                xyz = (cam0_c2w @ pts_cam.T).T[:, :3].astype(np.float32)
                pc_colors = rgb_depth[vv, uu]

                if len(xyz) > MAX_PC_POINTS:
                    idx = np.random.choice(len(xyz), MAX_PC_POINTS, replace=False)
                    xyz = xyz[idx]
                    pc_colors = pc_colors[idx]
        except Exception as e:
            print(f"    Warning: could not build point cloud: {e}")

    # Load metadata
    meta = {}
    meta_file = clip_path / "metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)

    result = {
        'pts3d': pts3d.astype(np.float16),
        'vis3d': vis3d,
        'tracks2d': tracks2d.astype(np.float16),
        'vis2d': vis2d,
        'dim': np.array(dim, dtype=np.int32),
        'rgb_path': str(rgb_path),
        'intrinsics': intrinsics,
        'cam0_c2w': cam0_c2w,
        'cam_pos': cam_pos.astype(np.float16),
        'cam_fwd': cam_fwd.astype(np.float16),
    }
    if xyz is not None:
        result['pc_xyz'] = xyz.astype(np.float16)
        result['pc_colors'] = pc_colors

    return result, meta


# ---------------------------------------------------------------------------
# Select clips
# ---------------------------------------------------------------------------
def select_egodex_clips():
    """3 random clips per task."""
    # Group clip IDs by task
    task_clips = defaultdict(list)
    for fname in os.listdir(FINAL_TRACKS):
        if not fname.endswith("_2d.npz"):
            continue
        cid = fname[:-len("_2d.npz")]
        if not cid.startswith("part"):
            continue
        # Extract task: remove partN_ prefix and trailing _NUMBER
        parts = cid.split("_")
        # Find the part prefix (part1, part2, etc.)
        rest = "_".join(parts[1:])  # everything after partN
        # The last segment is the numeric index
        # Find last underscore before a pure number
        idx = rest.rfind("_")
        if idx > 0 and rest[idx+1:].isdigit():
            task = rest[:idx]
        else:
            task = rest
        task_clips[task].append(cid)

    selected = []
    for task, clips in sorted(task_clips.items()):
        chosen = random.sample(clips, min(3, len(clips)))
        for cid in chosen:
            selected.append((cid, task))
    print(f"EgoDex: {len(task_clips)} tasks, {len(selected)} clips selected")
    return selected


def select_hdepic_clips():
    """3 random clips per participant P01-P09."""
    participant_clips = defaultdict(list)
    for fname in os.listdir(FINAL_TRACKS):
        if not fname.endswith("_2d.npz"):
            continue
        cid = fname[:-len("_2d.npz")]
        if not (cid.startswith("P") and cid[1:3].isdigit()):
            continue
        pid = cid.split("-")[0]  # P01, P02, etc.
        participant_clips[pid].append(cid)

    selected = []
    for pid, clips in sorted(participant_clips.items()):
        chosen = random.sample(clips, min(3, len(clips)))
        for cid in chosen:
            selected.append((cid, pid))
    print(f"HD-EPIC: {len(participant_clips)} participants, {len(selected)} clips selected")
    return selected


def select_xperience_clips():
    """30 clips total, 1 per episode randomly. Uses cached index if available."""
    INDEX_CACHE = Path("/weka/prior-default/jianingz/home/project/_GenTraj/vipe/annotation_app/index_cache.json")

    if INDEX_CACHE.exists():
        # Use cached index (fast)
        with open(INDEX_CACHE) as f:
            cache = json.load(f)
        all_clips = cache.get("xperience", [])
        print(f"  (using cached index: {len(all_clips)} xperience clips)")
    else:
        # Slow scan fallback
        all_clips = []
        for uuid_dir in os.listdir(XPERIENCE_DIR):
            uuid_path = XPERIENCE_DIR / uuid_dir
            if not uuid_path.is_dir():
                continue
            for ep in os.listdir(uuid_path):
                ep_path = uuid_path / ep
                if not ep_path.is_dir() or not ep.startswith("ep"):
                    continue
                for clip in os.listdir(ep_path):
                    clip_path = ep_path / clip
                    if (clip_path / "filtered_tracks" / "object.npz").is_file():
                        all_clips.append(f"{uuid_dir}/{ep}/{clip}")

    # Group by episode
    ep_clips = defaultdict(list)
    for clip_rel in all_clips:
        parts = clip_rel.split("/")
        ep_key = f"{parts[0]}/{parts[1]}"
        ep_clips[ep_key].append(clip_rel)

    # Pick 1 per episode, then take 30
    candidates = []
    for ep_key, clips in ep_clips.items():
        candidates.append(random.choice(clips))

    random.shuffle(candidates)
    selected = candidates[:30]
    print(f"Xperience: {len(ep_clips)} episodes, {len(selected)} clips selected")
    return selected


# ---------------------------------------------------------------------------
# Load text annotations
# ---------------------------------------------------------------------------
def load_egodex_annotations():
    """Load captions from pipeline_run_info.json + combined_prompts files."""
    import glob
    annots = {}
    # Best source: pipeline_run_info.json (molmo2_8b_caption)
    OUTPUTS_DIR = Path("/weka/prior-default/jianingz/home/project/_GenTraj/outputs")
    for f in glob.glob(str(OUTPUTS_DIR / "egodex_pipeline_*/pipeline_run_info.json")):
        with open(f) as fh:
            data = json.load(fh)
        for cid, info in data.get("episodes", {}).items():
            cap = info.get("molmo2_8b_caption", "")
            if cap and cid not in annots:
                annots[cid] = cap
    print(f"  pipeline_run_info: {len(annots)} captions")
    # Supplement with combined_prompts (molmo2_8b_recaption)
    for fname in os.listdir(TMP_DIR):
        if not fname.startswith("combined_prompts_egodex") or not fname.endswith(".json"):
            continue
        with open(TMP_DIR / fname) as f:
            data = json.load(f)
        for cid, info in data.items():
            if info.get("molmo2_8b_recaption") and cid not in annots:
                annots[cid] = info["molmo2_8b_recaption"]
    print(f"  + combined_prompts: {len(annots)} total")
    return annots


def load_hdepic_annotations():
    """Load all hd_epic stage6 task files into a dict {clip_id: caption}."""
    annots = {}
    for fname in os.listdir(TMP_DIR):
        if not fname.startswith("hd_epic_stage6_tasks") or not fname.endswith(".json"):
            continue
        with open(TMP_DIR / fname) as f:
            data = json.load(f)
        for item in data:
            vid = item.get("video_id", "")
            caption = item.get("caption", item.get("llm_description", ""))
            if vid and caption:
                annots[vid] = caption
    print(f"Loaded {len(annots)} HD-EPIC annotations")
    return annots


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(42)
    np.random.seed(42)

    # Select clips
    print("=== Selecting clips ===")
    egodex_clips = select_egodex_clips()
    hdepic_clips = select_hdepic_clips()
    xp_clips = select_xperience_clips()

    # Load text annotations
    print("\n=== Loading annotations ===")
    egodex_annots = load_egodex_annotations()
    hdepic_annots = load_hdepic_annotations()

    manifest = {"egodex": [], "hdepic": [], "xperience": []}

    # Process EgoDex
    print(f"\n=== Processing EgoDex ({len(egodex_clips)} clips) ===")
    out_dir = OUTPUT_DIR / "egodex"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (cid, task) in enumerate(egodex_clips):
        print(f"  [{i+1}/{len(egodex_clips)}] {cid}")
        result = process_ft_clip(cid)
        if result is None:
            continue
        rgb_path = result.pop('rgb_path')
        np.savez_compressed(out_dir / f"{cid}.npz", **result)
        text = egodex_annots.get(cid, task.replace("_", " "))
        manifest["egodex"].append({
            "id": cid, "task": task, "text": text,
            "rgb_url": f"/video/egodex/{cid}",
            "data_url": f"/static/data/egodex/{cid}.npz",
            "rgb_path": rgb_path,
        })

    # Process HD-EPIC
    print(f"\n=== Processing HD-EPIC ({len(hdepic_clips)} clips) ===")
    out_dir = OUTPUT_DIR / "hdepic"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (cid, pid) in enumerate(hdepic_clips):
        print(f"  [{i+1}/{len(hdepic_clips)}] {cid}")
        result = process_ft_clip(cid)
        if result is None:
            continue
        rgb_path = result.pop('rgb_path')
        np.savez_compressed(out_dir / f"{cid}.npz", **result)
        text = hdepic_annots.get(cid, "")
        manifest["hdepic"].append({
            "id": cid, "participant": pid, "text": text,
            "rgb_url": f"/video/hdepic/{cid}",
            "data_url": f"/static/data/hdepic/{cid}.npz",
            "rgb_path": rgb_path,
        })

    # Process Xperience
    print(f"\n=== Processing Xperience ({len(xp_clips)} clips) ===")
    out_dir = OUTPUT_DIR / "xperience"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, clip_rel in enumerate(xp_clips):
        print(f"  [{i+1}/{len(xp_clips)}] {clip_rel}")
        ret = process_xp_clip(clip_rel)
        if ret is None:
            continue
        result, meta = ret
        rgb_path = result.pop('rgb_path')
        safe_id = clip_rel.replace("/", "__")
        np.savez_compressed(out_dir / f"{safe_id}.npz", **result)
        text = meta.get("label", "") + (" — " + meta.get("description", "") if meta.get("description") else "")
        manifest["xperience"].append({
            "id": safe_id, "clip_rel": clip_rel, "text": text,
            "label": meta.get("label", ""),
            "rgb_url": f"/video/xperience/{safe_id}",
            "data_url": f"/static/data/xperience/{safe_id}.npz",
            "rgb_path": rgb_path,
        })

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n=== Done ===")
    print(f"Manifest: {manifest_path}")
    print(f"EgoDex: {len(manifest['egodex'])}, HD-EPIC: {len(manifest['hdepic'])}, Xperience: {len(manifest['xperience'])}")


if __name__ == "__main__":
    main()
