from __future__ import annotations

"""Convert World2Data demo output into protocol-aligned camera/point-cloud layers.

This script reads a demo output directory that contains:
- demo_scene.usda
- demo_scene.ply
- demo_scene_scene_graph.json (optional, for timestamps)

It writes:
- camera_poses.usda
- point_cloud_frames.usda

Point clouds are referenced as external assets (`w2d:pointsAsset`) per frame.
No dense point arrays are authored in-layer.
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_]")


def _require_pxr() -> tuple[Any, Any, Any, Any]:
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover
        raise ImportError("OpenUSD Python bindings are required. Install `usd-core`.") from exc
    return Gf, Sdf, Usd, UsdGeom


def _safe_prim_name(raw: str) -> str:
    value = _SAFE_NAME_RE.sub("_", raw).strip("_")
    if not value:
        value = "run"
    if value[0].isdigit():
        value = f"n_{value}"
    return value


def _read_scene_graph_sidecar(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _read_ply_vertex_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            text = line.strip()
            if text.startswith("element vertex "):
                try:
                    return int(text.split()[-1])
                except Exception:
                    return None
            if text == "end_header":
                break
    return None


def _camera_index_from_name(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1))
    return 0


def _collect_camera_matrices(stage: Any, cameras_root: str = "/World/Cameras") -> list[Any]:
    root = stage.GetPrimAtPath(cameras_root)
    if not root:
        raise ValueError(f"Cameras root not found: {cameras_root}")

    camera_prims = sorted(root.GetChildren(), key=lambda prim: _camera_index_from_name(prim.GetName()))
    matrices: list[Any] = []
    for prim in camera_prims:
        xform_attr = prim.GetAttribute("xformOp:transform")
        if not xform_attr:
            continue
        matrix = xform_attr.Get()
        if matrix is None:
            continue
        matrices.append(matrix)
    if not matrices:
        raise ValueError(f"No camera transforms found under {cameras_root}")
    return matrices


def _derive_timestamps(
    *,
    frame_count: int,
    fallback_fps: float,
    keyframe_timestamps: list[float] | None,
    video_fps: float | None,
) -> list[float]:
    if (
        keyframe_timestamps is not None
        and len(keyframe_timestamps) >= frame_count
        and video_fps is not None
        and video_fps > 1e-9
    ):
        return [float(keyframe_timestamps[i]) / float(video_fps) for i in range(frame_count)]
    fps = max(float(fallback_fps), 1e-6)
    return [float(i) / fps for i in range(frame_count)]


def _infer_timecodes_per_second(
    *,
    timestamps: list[float],
    fallback_fps: float,
) -> float:
    if len(timestamps) >= 2:
        deltas = [
            timestamps[i] - timestamps[i - 1]
            for i in range(1, len(timestamps))
            if timestamps[i] > timestamps[i - 1]
        ]
        if deltas:
            dt = median(deltas)
            if dt > 1e-9:
                return 1.0 / dt
    return max(float(fallback_fps), 1e-6)


def _write_run_provenance(
    *,
    stage: Any,
    run_id: str,
    component: str,
    model_name: str,
    params: dict[str, Any],
) -> None:
    _Gf, Sdf, _Usd, UsdGeom = _require_pxr()

    UsdGeom.Scope.Define(stage, "/World/W2D/Provenance")
    UsdGeom.Scope.Define(stage, "/World/W2D/Provenance/runs")
    run_prim = UsdGeom.Scope.Define(
        stage,
        f"/World/W2D/Provenance/runs/{_safe_prim_name(run_id)}",
    ).GetPrim()
    run_prim.CreateAttribute("w2d:runId", Sdf.ValueTypeNames.String, custom=True).Set(run_id)
    run_prim.CreateAttribute("w2d:component", Sdf.ValueTypeNames.String, custom=True).Set(component)
    run_prim.CreateAttribute("w2d:modelName", Sdf.ValueTypeNames.String, custom=True).Set(model_name)
    run_prim.CreateAttribute("w2d:modelVersion", Sdf.ValueTypeNames.String, custom=True).Set("0.1")
    run_prim.CreateAttribute("w2d:gitCommit", Sdf.ValueTypeNames.String, custom=True).Set("unknown")
    run_prim.CreateAttribute("w2d:timestampIso8601", Sdf.ValueTypeNames.String, custom=True).Set(
        datetime.now(timezone.utc).isoformat()
    )
    run_prim.CreateAttribute("w2d:params", Sdf.ValueTypeNames.String, custom=True).Set(
        json.dumps(params, sort_keys=True)
    )


def write_camera_poses_layer(
    *,
    output_path: Path,
    camera_matrices_world_to_camera: list[Any],
    timestamps: list[float],
    run_id: str,
    timecodes_per_second: float,
) -> None:
    Gf, Sdf, Usd, UsdGeom = _require_pxr()

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(float(timecodes_per_second))
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(float(max(0, len(camera_matrices_world_to_camera) - 1)))

    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)

    UsdGeom.Scope.Define(stage, "/World/W2D")
    UsdGeom.Scope.Define(stage, "/World/W2D/Sensors")
    UsdGeom.Scope.Define(stage, "/World/W2D/Sensors/CameraPoses")
    UsdGeom.Scope.Define(stage, "/World/W2D/Sensors/CameraPoses/Frames")

    for frame_index, matrix in enumerate(camera_matrices_world_to_camera):
        prim = UsdGeom.Scope.Define(
            stage,
            f"/World/W2D/Sensors/CameraPoses/Frames/f_{frame_index:06d}",
        ).GetPrim()
        prim.CreateAttribute("w2d:frameIndex", Sdf.ValueTypeNames.Int, custom=True).Set(frame_index)
        prim.CreateAttribute("w2d:timestampSec", Sdf.ValueTypeNames.Double, custom=True).Set(
            float(timestamps[frame_index])
        )
        prim.CreateAttribute("w2d:rotationMatrix", Sdf.ValueTypeNames.Matrix3d, custom=True).Set(
            Gf.Matrix3d(
                float(matrix[0][0]),
                float(matrix[0][1]),
                float(matrix[0][2]),
                float(matrix[1][0]),
                float(matrix[1][1]),
                float(matrix[1][2]),
                float(matrix[2][0]),
                float(matrix[2][1]),
                float(matrix[2][2]),
            )
        )
        prim.CreateAttribute("w2d:translation", Sdf.ValueTypeNames.Float3, custom=True).Set(
            Gf.Vec3f(
                float(matrix[0][3]),
                float(matrix[1][3]),
                float(matrix[2][3]),
            )
        )
        prim.CreateAttribute("w2d:poseConvention", Sdf.ValueTypeNames.String, custom=True).Set(
            "world_to_camera"
        )
        prim.CreateAttribute("w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True).Set(run_id)

    _write_run_provenance(
        stage=stage,
        run_id=run_id,
        component="reconstruction.camera_poses",
        model_name="world2data_demo_converter",
        params={"frame_count": len(camera_matrices_world_to_camera)},
    )
    stage.GetRootLayer().Save()


def write_point_cloud_frames_layer(
    *,
    output_path: Path,
    frame_count: int,
    timestamps: list[float],
    points_asset_path: str,
    points_format: str,
    points_count: int | None,
    run_id: str,
    timecodes_per_second: float,
) -> None:
    _Gf, Sdf, Usd, UsdGeom = _require_pxr()

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(float(timecodes_per_second))
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(float(max(0, frame_count - 1)))

    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)

    UsdGeom.Scope.Define(stage, "/World/W2D")
    UsdGeom.Scope.Define(stage, "/World/W2D/Reconstruction")
    UsdGeom.Scope.Define(stage, "/World/W2D/Reconstruction/PointCloudFrames")

    for frame_index in range(frame_count):
        prim = UsdGeom.Scope.Define(
            stage,
            f"/World/W2D/Reconstruction/PointCloudFrames/f_{frame_index:06d}",
        ).GetPrim()
        prim.CreateAttribute("w2d:frameIndex", Sdf.ValueTypeNames.Int, custom=True).Set(frame_index)
        prim.CreateAttribute("w2d:timestampSec", Sdf.ValueTypeNames.Double, custom=True).Set(
            float(timestamps[frame_index])
        )
        prim.CreateAttribute("w2d:pointsAsset", Sdf.ValueTypeNames.Asset, custom=True).Set(
            Sdf.AssetPath(points_asset_path)
        )
        prim.CreateAttribute("w2d:pointsFormat", Sdf.ValueTypeNames.String, custom=True).Set(points_format)
        if points_count is not None:
            prim.CreateAttribute("w2d:pointCount", Sdf.ValueTypeNames.Int, custom=True).Set(points_count)
        prim.CreateAttribute("w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True).Set(run_id)

    _write_run_provenance(
        stage=stage,
        run_id=run_id,
        component="reconstruction.point_cloud_frames_index",
        model_name="world2data_demo_converter",
        params={
            "frame_count": frame_count,
            "points_asset": points_asset_path,
            "points_count": points_count,
        },
    )
    stage.GetRootLayer().Save()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert World2Data demo output into protocol-aligned "
            "camera pose + stamped point-cloud index layers."
        )
    )
    parser.add_argument(
        "--demo-output-dir",
        type=Path,
        default=Path("/run/media/adrian/Crucial X9/files_World2data/demo_output_5fps"),
        help="path containing demo_scene.usda/.ply/.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="where protocol layers are written",
    )
    parser.add_argument(
        "--camera-layer-name",
        type=str,
        default="camera_poses.usda",
    )
    parser.add_argument(
        "--point-cloud-layer-name",
        type=str,
        default="point_cloud_frames.usda",
    )
    parser.add_argument(
        "--fallback-fps",
        type=float,
        default=5.0,
        help="used when sidecar timing is unavailable",
    )
    parser.add_argument(
        "--timecodes-per-second",
        type=float,
        default=0.0,
        help="override output stage timeCodesPerSecond (0 = infer)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="provenance run id; default uses UTC timestamp",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    demo_dir = args.demo_output_dir
    scene_path = demo_dir / "demo_scene.usda"
    ply_path = demo_dir / "demo_scene.ply"
    sidecar_path = demo_dir / "demo_scene_scene_graph.json"
    if not scene_path.exists():
        raise FileNotFoundError(f"Missing file: {scene_path}")
    if not ply_path.exists():
        raise FileNotFoundError(f"Missing file: {ply_path}")

    _Gf, _Sdf, Usd, _UsdGeom = _require_pxr()
    stage = Usd.Stage.Open(str(scene_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {scene_path}")

    camera_matrices = _collect_camera_matrices(stage)
    frame_count = len(camera_matrices)

    sidecar = _read_scene_graph_sidecar(sidecar_path)
    keyframe_timestamps = sidecar.get("keyframe_timestamps")
    if not isinstance(keyframe_timestamps, list):
        keyframe_timestamps = None
    video_fps_value = sidecar.get("video_fps")
    video_fps = float(video_fps_value) if isinstance(video_fps_value, (int, float)) else None
    timestamps = _derive_timestamps(
        frame_count=frame_count,
        fallback_fps=float(args.fallback_fps),
        keyframe_timestamps=keyframe_timestamps,
        video_fps=video_fps,
    )

    tps = float(args.timecodes_per_second)
    if tps <= 0.0:
        tps = _infer_timecodes_per_second(timestamps=timestamps, fallback_fps=float(args.fallback_fps))

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_layer = output_dir / args.camera_layer_name
    point_cloud_layer = output_dir / args.point_cloud_layer_name

    # Use a relative asset path from the layer location.
    try:
        points_asset_rel = os.path.relpath(
            str(ply_path.resolve()),
            start=str(point_cloud_layer.parent.resolve()),
        )
    except Exception:
        points_asset_rel = str(ply_path.resolve())

    points_count = _read_ply_vertex_count(ply_path)
    points_format = ply_path.suffix.lstrip(".").lower() or "unknown"

    write_camera_poses_layer(
        output_path=camera_layer,
        camera_matrices_world_to_camera=camera_matrices,
        timestamps=timestamps,
        run_id=run_id,
        timecodes_per_second=tps,
    )
    write_point_cloud_frames_layer(
        output_path=point_cloud_layer,
        frame_count=frame_count,
        timestamps=timestamps,
        points_asset_path=points_asset_rel,
        points_format=points_format,
        points_count=points_count,
        run_id=run_id,
        timecodes_per_second=tps,
    )

    print(f"wrote camera poses layer: {camera_layer}")
    print(f"wrote point-cloud frames layer: {point_cloud_layer}")
    print(f"run_id={run_id}")
    print(f"frame_count={frame_count}")
    print(f"timeCodesPerSecond={tps:.6f}")
    print(f"points_asset={points_asset_rel}")


if __name__ == "__main__":
    main()
