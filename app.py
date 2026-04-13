"""
Visualization website — browse EgoDex / HD-EPIC / Xperience clips
with interactive 3D scene viewer and synced RGB video.
"""

import json
import os
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, render_template, send_file, request

app = Flask(__name__)

DATA_DIR = Path(__file__).parent / "static" / "data"
MANIFEST = None
EVAL_MANIFEST = None


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            obj = np.nan_to_num(obj.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
            return obj.tolist()
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            v = float(obj)
            return 0.0 if v != v else v
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def get_manifest():
    global MANIFEST
    if MANIFEST is None:
        with open(DATA_DIR / "manifest.json") as f:
            MANIFEST = json.load(f)
    return MANIFEST


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/eval")
def eval_page():
    # Serve root-level eval.html (same file used by GitHub Pages)
    return send_file(Path(__file__).parent / "eval.html")


@app.route("/motion")
def motion_page():
    return send_file(Path(__file__).parent / "motion.html")


@app.route("/api/motion_manifest")
def api_motion_manifest():
    motion_path = DATA_DIR / "motion" / "motion_manifest.json"
    if motion_path.exists():
        with open(motion_path) as f:
            return jsonify(json.load(f))
    return jsonify({})


@app.route("/modeling")
def modeling_page():
    return send_file(Path(__file__).parent / "modeling.html")


MODELING_MANIFEST = None

def get_modeling_manifest():
    global MODELING_MANIFEST
    if MODELING_MANIFEST is None:
        p = DATA_DIR / "modeling_manifest.json"
        if p.exists():
            with open(p) as f:
                MODELING_MANIFEST = json.load(f)
        else:
            MODELING_MANIFEST = {}
    return MODELING_MANIFEST


@app.route("/api/modeling_manifest")
def api_modeling_manifest():
    return jsonify(get_modeling_manifest())


@app.route("/video/modeling/<dataset>/<clip_id>")
def serve_modeling_video(dataset, clip_id):
    vpath = Path(__file__).parent / "static" / "videos" / "modeling" / dataset / f"{clip_id}.mp4"
    if vpath.exists():
        return send_file(vpath, mimetype="video/mp4")
    return "Video not found", 404


def get_eval_manifest():
    global EVAL_MANIFEST
    if EVAL_MANIFEST is None:
        eval_path = DATA_DIR / "eval" / "manifest.json"
        if eval_path.exists():
            with open(eval_path) as f:
                EVAL_MANIFEST = json.load(f)
        else:
            EVAL_MANIFEST = {"eval": []}
    return EVAL_MANIFEST


@app.route("/api/eval_manifest")
def api_eval_manifest():
    return jsonify(get_eval_manifest())


@app.route("/video/eval/<clip_id>")
def serve_eval_video(clip_id):
    m = get_eval_manifest()
    for c in m.get("eval", []):
        if c["id"] == clip_id:
            rgb_path = c["rgb_path"]
            if os.path.exists(rgb_path):
                return send_file(rgb_path, mimetype="video/mp4")
            return "Video not found", 404
    return "Clip not found", 404


@app.route("/api/manifest")
def api_manifest():
    return jsonify(get_manifest())


@app.route("/api/clip/<dataset>/<clip_id>")
def api_clip(dataset, clip_id):
    """Serve clip data as JSON (converted from npz server-side)."""
    npz_path = DATA_DIR / dataset / f"{clip_id}.npz"
    if not npz_path.exists():
        return jsonify({"error": "not found"}), 404
    d = np.load(npz_path, allow_pickle=True)
    result = {}
    for k in d.files:
        result[k] = d[k]
    return Response(
        json.dumps(result, cls=NumpyEncoder),
        mimetype="application/json"
    )


@app.route("/video/<dataset>/<clip_id>")
def serve_video(dataset, clip_id):
    m = get_manifest()
    clips = m.get(dataset, [])
    for c in clips:
        if c["id"] == clip_id:
            rgb_path = c["rgb_path"]
            if os.path.exists(rgb_path):
                return send_file(rgb_path, mimetype="video/mp4")
            return "Video not found", 404
    return "Clip not found", 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888, debug=False, threaded=True)
