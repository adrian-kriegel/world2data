"""Generate a presentation-ready demo from pipeline outputs.

This script creates a polished Rerun recording suitable for showing to judges.
It can work from:
  1. Existing pipeline outputs (JSON sidecar + .rrd)
  2. Running the full pipeline on a video

Usage:
  # From existing outputs
  uv run python generate_demo.py --json real_scene_scene_graph.json --rrd real_scene.rrd

  # From a video (runs full pipeline)
  uv run python generate_demo.py --video path/to/video.mp4

  # Open directly in Rerun viewer
  uv run python generate_demo.py --json real_scene_scene_graph.json --rrd real_scene.rrd --open
"""
import os
import sys
import json
import argparse
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_demo_summary(json_path, output_dir="."):
    """Print a human-readable summary of pipeline outputs for the pitch."""
    with open(json_path) as f:
        data = json.load(f)

    n_pts = data.get("num_points", 0)
    n_cams = data.get("num_cameras", 0)
    n_frames = data.get("num_frames", 0)
    objects_3d = data.get("objects_3d", [])
    scene_objects = data.get("scene_graph", {}).get("objects", [])
    fps = data.get("video_fps", 30.0)
    timestamps = data.get("keyframe_timestamps", [])

    print()
    print("=" * 70)
    print("  WORLD2DATA -- 3D Dynamic Scene Graph from 2D Video")
    print("=" * 70)
    print()
    print("  PIPELINE OUTPUT SUMMARY")
    print("  " + "-" * 40)
    print(f"  3D Points Reconstructed:   {n_pts:>10,}")
    print(f"  Camera Poses:              {n_cams:>10}")
    print(f"  Temporal Keyframes:        {n_frames:>10}")
    if timestamps:
        duration = timestamps[-1] / fps
        print(f"  Video Duration Covered:    {duration:>9.1f}s")
    print(f"  Objects in 3D:             {len(objects_3d):>10}")
    print(f"  Objects (Reasoning):       {len(scene_objects):>10}")
    print()

    if objects_3d:
        print("  DETECTED OBJECTS (3D)")
        print("  " + "-" * 40)
        for obj in objects_3d:
            entity = obj.get("entity", "?")
            otype = obj.get("type", "?")
            joint = obj.get("component_type", "?")
            state_i = obj.get("initial_state", "?")
            state_f = obj.get("final_state", "?")
            center = obj.get("center", [0, 0, 0])
            size = obj.get("size", [0, 0, 0])
            print(f"  {entity:20s}  type={otype:10s}  joint={joint}")
            print(f"  {'':20s}  state: {state_i} -> {state_f}")
            print(f"  {'':20s}  center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})  "
                  f"size=({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})")
            changes = obj.get("state_changes", [])
            for sc in changes:
                print(f"  {'':20s}  [{sc.get('time', '?')}] "
                      f"{sc.get('from', '?')} -> {sc.get('to', '?')} "
                      f"({sc.get('cause', '?')})")
            print()

    print("  OUTPUT FILES")
    print("  " + "-" * 40)
    base = json_path.replace("_scene_graph.json", "")
    for ext, desc in [
        (".usda", "OpenUSD scene (point cloud, cameras, physics objects)"),
        ("_scene_graph.json", "Scene graph JSON (objects, reasoning, metadata)"),
        (".rrd", "Interactive Rerun recording (temporal 3D)"),
        (".ply", "Colored point cloud (any 3D viewer)"),
    ]:
        fpath = base + ext
        if os.path.isfile(fpath):
            sz = os.path.getsize(fpath)
            if sz > 1024 * 1024:
                sz_str = f"{sz / 1024 / 1024:.1f} MB"
            else:
                sz_str = f"{sz / 1024:.0f} KB"
            print(f"  {os.path.basename(fpath):35s}  {sz_str:>8s}  {desc}")

    print()
    print("  HOW TO VIEW")
    print("  " + "-" * 40)
    rrd = base + ".rrd"
    if os.path.isfile(rrd):
        print(f"  Interactive 3D:  uv run rerun {rrd}")
    ply = base + ".ply"
    if os.path.isfile(ply):
        print(f"  Point cloud:     Open {ply} in MeshLab / CloudCompare / Blender")
    usda = base + ".usda"
    if os.path.isfile(usda):
        print(f"  USD scene:       Open {usda} in NVIDIA Omniverse / usdview")
    print()
    print("=" * 70)
    print()


def run_pipeline_for_demo(video_path, output_name="demo"):
    """Run the full pipeline and generate all demo outputs."""
    from pipeline_controller import World2DataPipeline

    output_path = f"{output_name}.usda"
    pipeline = World2DataPipeline(
        video_path, output_path=output_path, rerun_enabled=False,
    )

    print(f"Running World2Data pipeline on: {video_path}")
    success = pipeline.run_ralph_loop()

    if not success:
        print("Pipeline failed. Check the output above for errors.")
        return None

    return output_path.replace(".usda", "_scene_graph.json")


def main():
    parser = argparse.ArgumentParser(
        description="World2Data Demo Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View summary of existing pipeline output
  uv run python generate_demo.py --json real_scene_scene_graph.json

  # Run pipeline on video and generate demo
  uv run python generate_demo.py --video video.mp4 --output demo

  # Open the Rerun viewer directly
  uv run python generate_demo.py --json real_scene_scene_graph.json --open
""")
    parser.add_argument("--json", help="Path to pipeline JSON sidecar")
    parser.add_argument("--rrd", help="Path to .rrd file (for --open)")
    parser.add_argument("--video", help="Path to input video (runs full pipeline)")
    parser.add_argument("--output", default="demo",
                        help="Output name prefix (for --video mode)")
    parser.add_argument("--open", action="store_true",
                        help="Open Rerun viewer after generating")
    args = parser.parse_args()

    json_path = args.json
    rrd_path = args.rrd

    if args.video:
        json_path = run_pipeline_for_demo(args.video, args.output)
        if json_path is None:
            sys.exit(1)
        rrd_path = json_path.replace("_scene_graph.json", ".rrd")

    if json_path:
        create_demo_summary(json_path)

    if args.open:
        if not rrd_path:
            if json_path:
                rrd_path = json_path.replace("_scene_graph.json", ".rrd")
        if rrd_path and os.path.isfile(rrd_path):
            print(f"Opening Rerun viewer: {rrd_path}")
            subprocess.Popen(["uv", "run", "rerun", rrd_path])
        else:
            print(f"Cannot open: {rrd_path} not found.")

    if not json_path and not args.video:
        parser.print_help()


if __name__ == "__main__":
    main()
