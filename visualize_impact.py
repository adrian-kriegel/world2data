"""World2Data Pitch Visualization — Rerun.io "Matrix View"

Can run in two modes:
  1. Demo mode: Animated synthetic door state-change (no pipeline needed)
  2. Live mode: Load real pipeline output and visualize 3D scene + objects

Usage:
  uv run python visualize_impact.py                    # demo mode
  uv run python visualize_impact.py --json output_scene_graph.json  # live mode
"""
import json
import argparse
import numpy as np
import rerun as rr


def visualize_demo():
    """Animated demo: door state change (red=closed → green=open)."""
    rr.init("World2Data_Pitch", spawn=True)

    # Simulated point cloud (a room-like box)
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

    # Log static geometry
    all_pts = np.vstack([floor, wall])
    all_colors = np.vstack([floor_colors, wall_colors])
    rr.log("world/geo", rr.Points3D(all_pts, colors=all_colors, radii=0.02))

    # Simulation of a "State Change" event for the pitch video
    for t in range(100):
        rr.set_time_sequence("frame", t)

        # Door: turns from Red (Closed) to Green (Open) at frame 50
        if t < 50:
            color = [220, 40, 40]
            label = "Door: CLOSED"
        else:
            color = [40, 200, 40]
            label = "Door: OPEN"

        rr.log(
            "world/objects/Door_01",
            rr.Boxes3D(
                mins=[-1.5, 0, -2.9],
                sizes=[1.0, 2.2, 0.15],
                labels=[label],
                colors=[color],
            ),
        )

        # Table: always blue (static interactable)
        rr.log(
            "world/objects/Table_01",
            rr.Boxes3D(
                mins=[0.5, 0, -1],
                sizes=[1.5, 0.8, 1.0],
                labels=["Table: STATIC"],
                colors=[[40, 80, 220]],
            ),
        )

        # Cup: vanishes at frame 70 (picked up)
        if t < 70:
            rr.log(
                "world/objects/Cup_01",
                rr.Boxes3D(
                    mins=[1.0, 0.8, -0.7],
                    sizes=[0.15, 0.2, 0.15],
                    labels=["Cup: ON_TABLE"],
                    colors=[[40, 200, 200]],
                ),
            )
        else:
            rr.log(
                "world/objects/Cup_01",
                rr.Boxes3D(
                    mins=[1.0, 0.8, -0.7],
                    sizes=[0.15, 0.2, 0.15],
                    labels=["Cup: VANISHED (picked up)"],
                    colors=[[200, 200, 40]],
                ),
            )

    print("Pitch visualization running. Open Rerun viewer to see the demo.")


def visualize_from_pipeline(json_path):
    """Load real pipeline output and visualize in Rerun."""
    rr.init("World2Data_Live", spawn=True)

    with open(json_path) as f:
        data = json.load(f)

    scene_graph = data.get("scene_graph", {})
    num_points = data.get("num_points", 0)
    num_cameras = data.get("num_cameras", 0)

    print(f"Loaded: {num_points} points, {num_cameras} cameras, "
          f"{len(scene_graph.get('objects', []))} objects")

    # Log objects from scene graph as labeled boxes
    for obj in scene_graph.get("objects", []):
        entity = obj.get("entity", "Unknown")
        obj_type = obj.get("type", "unknown")
        initial = obj.get("initial_state", "?")
        final = obj.get("final_state", "?")

        # Color by type
        color_map = {
            "door": [220, 40, 40],
            "table": [40, 80, 220],
            "cup": [40, 200, 200],
            "cabinet": [40, 200, 40],
            "window": [200, 200, 40],
        }
        color = color_map.get(obj_type.lower(), [180, 180, 180])

        label = f"{entity} ({obj_type}): {initial} → {final}"
        rr.log(
            f"world/objects/{entity}",
            rr.Boxes3D(
                mins=[-0.5, -0.5, -0.5],
                sizes=[1, 1, 1],
                labels=[label],
                colors=[color],
            ),
        )

    # Log text summary
    rr.log("info/summary", rr.TextLog(
        f"Pipeline output: {num_points} 3D points, {num_cameras} cameras, "
        f"{len(scene_graph.get('objects', []))} detected objects"
    ))

    print("Live visualization running. Open Rerun viewer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="World2Data Pitch Visualization")
    parser.add_argument("--json", type=str, default=None,
                        help="Path to pipeline JSON sidecar (live mode)")
    args = parser.parse_args()

    if args.json:
        visualize_from_pipeline(args.json)
    else:
        visualize_demo()
