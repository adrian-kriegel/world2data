#!/usr/bin/env python
"""World2Data Demo Runner -- Full Multi-Model Pipeline for Investor Demos.

Processes a video through ALL available models and produces:
  1. OpenUSD scene graph with accuracy scores + interaction metadata
  2. Rerun .rrd interactive recording with all model overlays
  3. PLY colored point cloud
  4. JSON scene graph with 4D objects + evaluation metrics
  5. Annotated video with YOLO boxes + SAM3 masks
  6. Human-review JSON for label corrections

Usage:
    uv run python demo_run.py                              # default video
    uv run python demo_run.py --video path/to/video.mp4    # custom video
    uv run python demo_run.py --usdview                    # open usdview after
    uv run python demo_run.py --review                     # open review UI after
"""
import os
import sys
import json
import time
import argparse
import subprocess
import numpy as np

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")


# =========================================================================
# Default paths
# =========================================================================
DEFAULT_VIDEO = os.path.join(
    os.path.dirname(__file__),
    "testenvironment", "LFM2_5_VL_1_6B", "inputs", "videos",
    "video_2026-02-08_09-36-42.mp4",
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "demo_output")


def run_demo(video_path: str, output_dir: str, use_multimodel: bool = True,
             open_usdview: bool = False, open_review: bool = False,
             demo_fps: float = 5.0):
    """Run the full demo pipeline and produce all outputs."""
    from pipeline_controller import World2DataPipeline

    os.makedirs(output_dir, exist_ok=True)
    output_usda = os.path.join(output_dir, "demo_scene.usda")

    print("=" * 70)
    print("  WORLD2DATA DEMO RUNNER")
    print("=" * 70)
    print(f"  Video:       {video_path}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Multi-model: {use_multimodel}")
    print(f"  Demo FPS:    {demo_fps}")
    print("=" * 70)

    # Check video exists
    if not os.path.isfile(video_path):
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    # Get video info
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    print(f"  Duration:    {duration:.1f}s ({frame_count} frames @ {fps:.0f} fps)")
    print(f"  Resolution:  {w}x{h}")
    file_mb = os.path.getsize(video_path) / 1e6
    print(f"  File size:   {file_mb:.1f} MB")
    print("=" * 70)

    t0 = time.time()

    # --- Run pipeline ---
    pipeline = World2DataPipeline(
        video_path,
        output_path=output_usda,
        keyframe_dir=os.path.join(output_dir, "keyframes"),
        cache_dir=os.path.join(output_dir, "cache"),
        rerun_enabled=True,  # Generate .rrd for Rerun viewer
    )
    success = pipeline.run_ralph_loop(
        use_multimodel=use_multimodel, demo_fps=demo_fps,
    )

    elapsed = time.time() - t0

    if not success:
        print(f"\nPIPELINE FAILED after {elapsed:.1f}s")
        sys.exit(1)

    # --- Generate additional outputs ---

    # 1. Annotated video (YOLO overlay)
    if pipeline.yolo_detections:
        try:
            from scene_fusion import VideoAnnotator
            annotated_path = os.path.join(output_dir, "demo_annotated.mp4")
            VideoAnnotator.create_annotated_video(
                video_path, pipeline.yolo_detections,
                output_path=annotated_path,
                sample_fps=5.0,
            )
            print(f"  Annotated video: {annotated_path}")
        except Exception as e:
            print(f"  WARN: Annotated video failed: {e}")

    # 2. Human review JSON
    review_path = os.path.join(output_dir, "demo_human_review.json")
    _generate_review_json(pipeline, review_path)

    # 3. Print summary
    _print_summary(pipeline, output_dir, elapsed)

    # 4. Open usdview if requested
    if open_usdview:
        _launch_usdview(output_usda)

    # 5. Open review UI if requested
    if open_review:
        _launch_review_ui(review_path)

    return pipeline


def _generate_review_json(pipeline, review_path):
    """Generate a human-reviewable JSON with all detected objects + accuracy."""
    review_data = {
        "review_version": "1.0",
        "video_path": pipeline.video_path,
        "pipeline_version": "v2_multimodel",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "objects": [],
        "interactions": [],
        "scene_narrative": "",
    }

    # Objects from pipeline
    for i, obj in enumerate(pipeline.objects_3d):
        # Find matching 4D object for confidence
        confidence = obj.tracking_confidence if hasattr(obj, 'tracking_confidence') else 0.5
        detected_by = []

        # Check which models detected this
        if pipeline.yolo_detections:
            for det in pipeline.yolo_detections:
                if obj.obj_type.lower() in [n.lower().replace(" ", "_")
                                             for n in det.class_names]:
                    if "yolo" not in detected_by:
                        detected_by.append("yolo")
                    break

        # Check Gemini
        if pipeline.scene_description and hasattr(pipeline.scene_description, 'objects'):
            for g_obj in (pipeline.scene_description.objects or []):
                g_type = g_obj.get("type", "").lower()
                if g_type and (g_type in obj.obj_type.lower()
                               or obj.obj_type.lower() in g_type):
                    if "gemini" not in detected_by:
                        detected_by.append("gemini")
                    break

        if "mast3r" not in detected_by and np.any(obj.bbox_3d_max != 0):
            detected_by.append("mast3r_3d")

        review_obj = {
            "id": i,
            "entity": obj.entity,
            "type": obj.obj_type,
            "component_type": obj.component_type,
            "initial_state": obj.initial_state,
            "final_state": obj.final_state,
            "state_changes": obj.state_changes,
            "center_3d": obj.center.tolist(),
            "size_3d": obj.size.tolist(),
            "confidence": round(confidence, 3),
            "detected_by": detected_by,
            "needs_review": confidence < 0.5 or len(detected_by) < 2,
            "human_label": None,    # <-- for human to fill in
            "human_verified": False,  # <-- for human to set
            "human_notes": "",       # <-- for human to annotate
        }
        review_data["objects"].append(review_obj)

    # Interactions from scene graph
    if pipeline.scene_graph and isinstance(pipeline.scene_graph, dict):
        for obj_info in pipeline.scene_graph.get("objects", []):
            if isinstance(obj_info, dict) and obj_info.get("state_changes"):
                for sc in obj_info["state_changes"]:
                    review_data["interactions"].append({
                        "entity": obj_info.get("entity", "unknown"),
                        "time": sc.get("time", "unknown"),
                        "from_state": sc.get("from", sc.get("from_state", "?")),
                        "to_state": sc.get("to", sc.get("to_state", "?")),
                        "cause": sc.get("cause", "unknown"),
                        "human_verified": False,
                        "human_notes": "",
                    })

    # Scene narrative
    if pipeline.scene_description and hasattr(pipeline.scene_description, 'narrative'):
        review_data["scene_narrative"] = pipeline.scene_description.narrative or ""

    with open(review_path, "w") as f:
        json.dump(review_data, f, indent=2, default=str)
    print(f"  Human review JSON: {review_path}")


def _print_summary(pipeline, output_dir, elapsed):
    """Print a comprehensive demo summary."""
    print("\n" + "=" * 70)
    print("  DEMO RESULTS")
    print("=" * 70)
    print(f"  Time elapsed:     {elapsed:.1f}s")
    print(f"  Keyframes:        {len(pipeline.keyframes)}")
    if pipeline.point_cloud is not None:
        print(f"  3D Points:        {pipeline.point_cloud.shape[0]:,}")
    print(f"  Temporal frames:  {len(pipeline.frame_data)}")
    print(f"  Objects detected: {len(pipeline.objects_3d)}")

    if pipeline.yolo_detections:
        total = sum(len(d.class_names) for d in pipeline.yolo_detections)
        classes = set(n for d in pipeline.yolo_detections for n in d.class_names)
        print(f"  YOLO detections:  {total} ({len(classes)} classes)")

    if pipeline.scene_description and hasattr(pipeline.scene_description, 'objects'):
        n = len(pipeline.scene_description.objects) if pipeline.scene_description.objects else 0
        print(f"  Gemini objects:   {n}")

    if pipeline.evaluation:
        score = pipeline.evaluation.get("overall_score", 0)
        print(f"  Quality score:    {score:.2f}")

    # List objects
    print("\n  DETECTED OBJECTS:")
    for obj in pipeline.objects_3d:
        conf = obj.tracking_confidence if hasattr(obj, 'tracking_confidence') else 0.5
        state = f"{obj.initial_state} -> {obj.final_state}" if obj.initial_state != obj.final_state else obj.initial_state
        print(f"    [{conf:.0%}] {obj.entity} ({obj.obj_type}): "
              f"{obj.component_type}, {state}")

    # List outputs
    print("\n  OUTPUT FILES:")
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / 1e6
            print(f"    {fname:40s} ({size_mb:.2f} MB)")

    print("\n  VIEW COMMANDS:")
    print(f"    uv run rerun {os.path.join(output_dir, 'demo_scene.rrd')}")
    print(f"    uv run python demo_run.py --usdview")
    print(f"    uv run python demo_run.py --review")
    print("=" * 70)


def _launch_usdview(usda_path):
    """Launch usdview on the USD file."""
    print(f"\nLaunching usdview on {usda_path}...")
    # Try usdview from PATH, then from usd-core package
    try:
        subprocess.Popen(["usdview", usda_path])
        print("usdview launched.")
    except FileNotFoundError:
        # Try python -m pxr.Usdviewq
        try:
            subprocess.Popen([sys.executable, "-m", "pxr.Usdviewq", usda_path])
            print("usdview launched via pxr.Usdviewq.")
        except Exception:
            print("WARNING: usdview not found in PATH.")
            print(f"  You can view the USD file with:")
            print(f"    usdview {usda_path}")
            print(f"  Or open it in NVIDIA Omniverse / Blender")
            print(f"  Or use: uv run python -c \"from pxr import Usd; "
                  f"s=Usd.Stage.Open('{usda_path}'); print(s.ExportToString()[:2000])\"")


def _launch_review_ui(review_json_path):
    """Launch the human-in-the-loop review prototype UI."""
    print(f"\nLaunching review UI for {review_json_path}...")
    try:
        subprocess.Popen([
            sys.executable, "human_review_ui.py", "--json", review_json_path
        ])
    except Exception as e:
        print(f"WARNING: Could not launch review UI: {e}")
        print(f"  Run manually: uv run python human_review_ui.py --json {review_json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="World2Data Demo Runner -- Full Multi-Model Pipeline"
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO,
                        help="Input video file path")
    parser.add_argument("--output", default=OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Demo FPS (keyframes per second). Use 5 for demos. "
                             "Set to 0 to use old threshold-based extraction.")
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy (Gemini-only) pipeline instead of multi-model")
    parser.add_argument("--usdview", action="store_true",
                        help="Open usdview after pipeline completes")
    parser.add_argument("--review", action="store_true",
                        help="Open human review UI after pipeline completes")
    args = parser.parse_args()

    run_demo(
        video_path=args.video,
        output_dir=args.output,
        use_multimodel=not args.legacy,
        open_usdview=args.usdview,
        open_review=args.review,
        demo_fps=args.fps,
    )


if __name__ == "__main__":
    main()
