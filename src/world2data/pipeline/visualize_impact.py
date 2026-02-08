"""World2Data Pitch Visualization -- Rerun.io "Matrix View"

Can run in two modes:
  1. Demo mode: Animated synthetic door state-change (no pipeline needed)
  2. Live mode: Load real pipeline output with positioned 3D objects

Usage:
  uv run python visualize_impact.py                    # demo mode
  uv run python visualize_impact.py --json output_scene_graph.json  # live mode
"""
import json
import argparse
import numpy as np
import rerun as rr


def visualize_demo():
    """Animated demo: door state change (red=closed -> green=open)."""
    rr.init("World2Data_Pitch", spawn=True)

    rng = np.random.RandomState(7)
    n_wall = 2000

    # Floor
    floor = np.column_stack([
        rng.uniform(-3, 3, n_wall),
        np.zeros(n_wall),
        rng.uniform(-3, 3, n_wall),
    ])
    floor_colors = np.full((n_wall, 3), [180, 160, 140], dtype=np.uint8)

    # Back wall
    wall = np.column_stack([
        rng.uniform(-3, 3, n_wall),
        rng.uniform(0, 3, n_wall),
        np.full(n_wall, -3.0),
    ])
    wall_colors = np.full((n_wall, 3), [200, 200, 210], dtype=np.uint8)

    rr.log("world/geo", rr.Points3D(
        np.vstack([floor, wall]),
        colors=np.vstack([floor_colors, wall_colors]),
        radii=0.02,
    ))

    for t in range(100):
        rr.set_time("frame", sequence=t)

        # Door: red -> green at frame 50
        if t < 50:
            color, label = [220, 40, 40], "Door: CLOSED"
        else:
            color, label = [40, 200, 40], "Door: OPEN"

        rr.log("world/objects/Door_01", rr.Boxes3D(
            centers=[[-1.0, 1.1, -2.83]],
            sizes=[[1.0, 2.2, 0.15]],
            labels=[label], colors=[color],
        ))

        # Table: static blue
        rr.log("world/objects/Table_01", rr.Boxes3D(
            centers=[[1.25, 0.4, -0.5]],
            sizes=[[1.5, 0.8, 1.0]],
            labels=["Table: STATIC"], colors=[[40, 80, 220]],
        ))

        # Cup: vanishes at frame 70
        if t < 70:
            cup_label, cup_color = "Cup: ON_TABLE", [40, 200, 200]
        else:
            cup_label, cup_color = "Cup: VANISHED (picked up)", [200, 200, 40]
        rr.log("world/objects/Cup_01", rr.Boxes3D(
            centers=[[1.075, 0.9, -0.625]],
            sizes=[[0.15, 0.2, 0.15]],
            labels=[cup_label], colors=[cup_color],
        ))

    print("Pitch demo running. Open Rerun viewer.")


def visualize_from_pipeline(json_path):
    """Load real pipeline output with positioned 3D objects."""
    rr.init("World2Data_Live", spawn=True)

    with open(json_path) as f:
        data = json.load(f)

    objects_3d = data.get("objects_3d", [])
    scene_objects = data.get("scene_graph", {}).get("objects", [])
    num_points = data.get("num_points", 0)
    num_cameras = data.get("num_cameras", 0)

    print(f"Loaded: {num_points} points, {num_cameras} cameras, "
          f"{len(objects_3d)} 3D objects, "
          f"{len(scene_objects)} reasoning objects")

    # Color by type
    type_colors = {
        "door": [220, 40, 40],
        "table": [40, 80, 220],
        "chair": [40, 200, 200],
        "cup": [200, 200, 40],
        "window": [200, 100, 40],
        "cabinet": [40, 200, 40],
        "shelf": [140, 80, 200],
        "person": [255, 140, 0],
    }

    # Log 3D objects with actual positions
    for obj in objects_3d:
        entity = obj.get("entity", "Unknown")
        obj_type = obj.get("type", "unknown")
        center = obj.get("center", [0, 0, 0])
        size = obj.get("size", [1, 1, 1])
        initial = obj.get("initial_state", "?")
        final = obj.get("final_state", "?")
        joint = obj.get("component_type", "?")

        color = type_colors.get(obj_type.lower(), [180, 180, 180])
        label = f"{entity} ({obj_type}): {initial} -> {final} [{joint}]"

        rr.log(f"world/objects/{entity}", rr.Boxes3D(
            centers=[center], sizes=[size],
            labels=[label], colors=[color],
        ))

    # Also log reasoning-only objects (without 3D position)
    for obj in scene_objects:
        entity = obj.get("entity", "Unknown")
        obj_type = obj.get("type", "unknown")
        initial = obj.get("initial_state", "?")
        final = obj.get("final_state", "?")
        joint = obj.get("component_type", "?")

        rr.log("info/reasoning", rr.TextLog(
            f"{entity} ({obj_type}): {initial} -> {final} [{joint}]"
        ))

    rr.log("info/summary", rr.TextLog(
        f"Pipeline: {num_points} pts, {num_cameras} cams, "
        f"{len(objects_3d)} 3D objects"
    ))

    print("Live visualization running. Open Rerun viewer.")


def main():
    parser = argparse.ArgumentParser(description="World2Data Pitch Visualization")
    parser.add_argument("--json", type=str, default=None,
                        help="Path to pipeline JSON sidecar (live mode)")
    args = parser.parse_args()

    if args.json:
        visualize_from_pipeline(args.json)
    else:
        visualize_demo()


if __name__ == "__main__":
    main()
