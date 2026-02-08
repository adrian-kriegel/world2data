from __future__ import annotations

"""Particle-filter adapter for protocol layers (YOLO + point cloud + camera + calibration)."""

import argparse
import json
import random
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .model import AABB2D, CameraIntrinsics, CameraPose, Detection2D, FilterResult, FrameContext
from .particle_filter import MultiObjectParticleFilter, ParticleFilterConfig


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_]")
_FRAME_INDEX_RE = re.compile(r"(\d+)$")
_POINTS_ASSET_CACHE: dict[tuple[str, int], tuple[tuple[float, float, float], ...]] = {}


def _safe_prim_name(raw: str) -> str:
    value = _SAFE_NAME_RE.sub("_", raw).strip("_")
    if not value:
        value = "item"
    if value[0].isdigit():
        value = f"n_{value}"
    return value


def _git_commit_or_unknown() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    value = result.stdout.strip()
    return value if value else "unknown"


def _require_pxr() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("OpenUSD Python bindings are required. Install `usd-core`.") from exc
    return Gf, Sdf, Usd, UsdGeom, Vt


@dataclass(frozen=True)
class CameraPoseFrame:
    frame_index: int
    timestamp_s: float
    pose: CameraPose


@dataclass(frozen=True)
class PointCloudFrame:
    frame_index: int
    timestamp_s: float
    points_world: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class ParticleFilterFrameResult:
    frame_index: int
    timestamp_s: float
    filter_result: FilterResult


def _frame_index_from_prim(prim: Any) -> int:
    frame_attr = prim.GetAttribute("w2d:frameIndex")
    if frame_attr and frame_attr.HasAuthoredValueOpinion():
        value = frame_attr.Get()
        if value is not None:
            return int(value)

    name = prim.GetName()
    match = _FRAME_INDEX_RE.search(name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not derive frame index for prim {prim.GetPath()}")


def _timestamp_from_prim(
    prim: Any,
    *,
    frame_index: int,
    default_fps: float,
) -> float:
    timestamp_attr = prim.GetAttribute("w2d:timestampSec")
    if timestamp_attr and timestamp_attr.HasAuthoredValueOpinion():
        value = timestamp_attr.Get()
        if value is not None:
            return float(value)
    return float(frame_index / max(default_fps, 1e-6))


def _matrix3d_to_tuple(matrix3d: Any) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    return (
        (float(matrix3d[0][0]), float(matrix3d[0][1]), float(matrix3d[0][2])),
        (float(matrix3d[1][0]), float(matrix3d[1][1]), float(matrix3d[1][2])),
        (float(matrix3d[2][0]), float(matrix3d[2][1]), float(matrix3d[2][2])),
    )


def _float3_to_tuple(value: Any) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def _float4_to_tuple(value: Any) -> tuple[float, float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _asset_value_to_path_string(value: Any) -> str:
    resolved = str(getattr(value, "resolvedPath", "") or "").strip()
    if resolved:
        return resolved
    authored = str(getattr(value, "path", "") or "").strip()
    if authored:
        return authored
    return str(value).strip()


def _resolve_points_asset_path(
    asset_value: Any,
    *,
    base_dir: Path | None,
) -> Path:
    raw = _asset_value_to_path_string(asset_value)
    if not raw:
        raise ValueError("Point-cloud asset attribute is empty")

    as_path = Path(raw).expanduser()
    candidates: list[Path] = []
    if as_path.is_absolute():
        candidates.append(as_path)
    if base_dir is not None:
        candidates.append((base_dir / raw).expanduser())
    candidates.append(as_path)

    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.exists():
            return resolved

    searched = ", ".join(str(candidate.resolve(strict=False)) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve point-cloud asset path '{raw}'. Tried: {searched}")


def _read_ascii_ply_points(
    *,
    path: Path,
    vertex_count: int,
    x_index: int,
    y_index: int,
    z_index: int,
    max_points: int,
    header_end_offset: int,
) -> tuple[tuple[float, float, float], ...]:
    wanted = vertex_count if max_points <= 0 else min(vertex_count, max_points)
    points: list[tuple[float, float, float]] = []

    with path.open("rb") as f:
        f.seek(header_end_offset)
        for _ in range(vertex_count):
            if len(points) >= wanted:
                break
            line_bytes = f.readline()
            if not line_bytes:
                break
            text = line_bytes.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            parts = text.split()
            needed_index = max(x_index, y_index, z_index)
            if len(parts) <= needed_index:
                continue
            points.append(
                (
                    float(parts[x_index]),
                    float(parts[y_index]),
                    float(parts[z_index]),
                )
            )
    return tuple(points)


def _read_ply_points_xyz(
    path: Path,
    *,
    max_points: int,
) -> tuple[tuple[float, float, float], ...]:
    fmt = ""
    vertex_count = 0
    vertex_properties: list[str] = []
    in_vertex = False
    header_end_offset = 0
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header in {path}")
            stripped = line.decode("ascii", errors="replace").strip()
            if stripped.startswith("format "):
                tokens = stripped.split()
                if len(tokens) >= 2:
                    fmt = tokens[1].strip()
            elif stripped.startswith("element "):
                tokens = stripped.split()
                if len(tokens) >= 3 and tokens[1] == "vertex":
                    in_vertex = True
                    vertex_count = int(tokens[2])
                    vertex_properties = []
                else:
                    in_vertex = False
            elif in_vertex and stripped.startswith("property "):
                tokens = stripped.split()
                if len(tokens) >= 3:
                    vertex_properties.append(tokens[-1])
            elif stripped == "end_header":
                header_end_offset = f.tell()
                break

    if fmt.lower() != "ascii":
        raise ValueError(f"Unsupported PLY format '{fmt}' for {path}; expected ascii")
    if vertex_count <= 0:
        return ()

    try:
        x_index = vertex_properties.index("x")
        y_index = vertex_properties.index("y")
        z_index = vertex_properties.index("z")
    except ValueError as exc:
        raise ValueError(f"PLY file {path} is missing x/y/z vertex properties") from exc

    return _read_ascii_ply_points(
        path=path,
        vertex_count=vertex_count,
        x_index=x_index,
        y_index=y_index,
        z_index=z_index,
        max_points=max_points,
        header_end_offset=header_end_offset,
    )


def _load_points_from_asset(
    path: Path,
    *,
    max_points: int,
) -> tuple[tuple[float, float, float], ...]:
    key = (str(path), int(max_points))
    cached = _POINTS_ASSET_CACHE.get(key)
    if cached is not None:
        return cached

    suffix = path.suffix.lower()
    if suffix == ".ply":
        points = _read_ply_points_xyz(path, max_points=max_points)
    else:
        raise ValueError(f"Unsupported points asset format '{suffix}' for {path}")

    _POINTS_ASSET_CACHE[key] = points
    return points


def _collect_latest_stitched_points_by_track(
    frame_results: Sequence[ParticleFilterFrameResult],
) -> dict[str, tuple[tuple[float, float, float], ...]]:
    stitched_points_by_track: dict[str, tuple[tuple[float, float, float], ...]] = {}
    for frame_result in frame_results:
        for track_id, stitched_points in frame_result.filter_result.stitched_points_by_track.items():
            stitched_points_by_track[track_id] = tuple(stitched_points)
    return stitched_points_by_track


def _write_merged_track_point_cloud_blobs(
    *,
    frame_results: Sequence[ParticleFilterFrameResult],
    output_path: Path,
) -> dict[str, str]:
    stitched_points_by_track = _collect_latest_stitched_points_by_track(frame_results)
    blob_dir = output_path.parent / f"{output_path.stem}_merged_track_point_clouds"
    blob_dir.mkdir(parents=True, exist_ok=True)

    assets_by_track: dict[str, str] = {}
    for track_id, points in sorted(stitched_points_by_track.items(), key=lambda item: item[0]):
        safe = _safe_prim_name(track_id)
        file_path = blob_dir / f"{safe}.npy"
        arr = np.asarray(points, dtype=np.float32)
        if arr.size == 0:
            arr = np.empty((0, 3), dtype=np.float32)
        np.save(file_path, arr, allow_pickle=False)

        try:
            relative = file_path.relative_to(output_path.parent)
            assets_by_track[track_id] = relative.as_posix()
        except Exception:
            assets_by_track[track_id] = str(file_path.resolve())
    return assets_by_track


def _open_composed_stage(
    *,
    calibration_layer: Path,
    camera_poses_layer: Path,
    yolo_layer: Path,
    point_cloud_layer: Path,
) -> Any:
    _Gf, Sdf, Usd, _UsdGeom, _Vt = _require_pxr()

    for path in (calibration_layer, camera_poses_layer, yolo_layer, point_cloud_layer):
        if not path.exists():
            raise FileNotFoundError(f"Missing input layer: {path}")

    root = Sdf.Layer.CreateAnonymous(".usda")
    root.subLayerPaths = [
        str(calibration_layer.resolve()),
        str(camera_poses_layer.resolve()),
        str(yolo_layer.resolve()),
        str(point_cloud_layer.resolve()),
    ]
    stage = Usd.Stage.Open(root)
    if stage is None:
        raise RuntimeError("Could not open composed stage from input layers")
    return stage


def read_calibration_intrinsics(
    stage: Any,
    *,
    calibration_camera_prim: str = "/World/W2D/Sensors/CalibrationCamera",
) -> CameraIntrinsics:
    camera_prim = stage.GetPrimAtPath(calibration_camera_prim)
    if not camera_prim:
        raise ValueError(f"Calibration camera prim not found: {calibration_camera_prim}")

    intrinsic_attr = camera_prim.GetAttribute("w2d:intrinsicMatrix")
    width_attr = camera_prim.GetAttribute("w2d:imageWidth")
    height_attr = camera_prim.GetAttribute("w2d:imageHeight")
    if not intrinsic_attr or not width_attr or not height_attr:
        raise ValueError(
            "Calibration prim must author w2d:intrinsicMatrix, w2d:imageWidth, w2d:imageHeight"
        )

    matrix = intrinsic_attr.Get()
    width = width_attr.Get()
    height = height_attr.Get()
    if matrix is None or width is None or height is None:
        raise ValueError("Calibration intrinsics attributes are missing values")

    fx = float(matrix[0][0])
    fy = float(matrix[1][1])
    cx = float(matrix[0][2])
    cy = float(matrix[1][2])
    distortion_model_attr = camera_prim.GetAttribute("w2d:distortionModel")
    distortion_coeffs_attr = camera_prim.GetAttribute("w2d:distortionCoeffs")
    distortion_model = ""
    if distortion_model_attr and distortion_model_attr.HasAuthoredValueOpinion():
        authored_model = distortion_model_attr.Get()
        if authored_model is not None:
            distortion_model = str(authored_model)
    distortion_coeffs: tuple[float, ...] = ()
    if distortion_coeffs_attr and distortion_coeffs_attr.HasAuthoredValueOpinion():
        authored_coeffs = distortion_coeffs_attr.Get()
        if authored_coeffs is not None:
            distortion_coeffs = tuple(float(value) for value in authored_coeffs)

    return CameraIntrinsics(
        width_px=int(width),
        height_px=int(height),
        fx_px=fx,
        fy_px=fy,
        cx_px=cx,
        cy_px=cy,
        distortion_model=distortion_model,
        distortion_coeffs=distortion_coeffs,
    )


def read_camera_pose_frames(
    stage: Any,
    *,
    camera_poses_frames_root: str = "/World/W2D/Sensors/CameraPoses/Frames",
    default_fps: float = 30.0,
) -> dict[int, CameraPoseFrame]:
    frames_root = stage.GetPrimAtPath(camera_poses_frames_root)
    if not frames_root:
        raise ValueError(f"Camera poses root not found: {camera_poses_frames_root}")

    frames: dict[int, CameraPoseFrame] = {}
    for prim in sorted(frames_root.GetChildren(), key=_frame_index_from_prim):
        frame_index = _frame_index_from_prim(prim)
        timestamp_s = _timestamp_from_prim(prim, frame_index=frame_index, default_fps=default_fps)

        rotation_attr = prim.GetAttribute("w2d:rotationMatrix")
        translation_attr = prim.GetAttribute("w2d:translation")
        if not rotation_attr or not translation_attr:
            raise ValueError(
                f"Camera pose frame {prim.GetPath()} must contain w2d:rotationMatrix and w2d:translation"
            )

        rotation_value = rotation_attr.Get()
        translation_value = translation_attr.Get()
        if rotation_value is None or translation_value is None:
            raise ValueError(f"Camera pose frame {prim.GetPath()} has empty pose values")

        frames[frame_index] = CameraPoseFrame(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            pose=CameraPose(
                rotation=_matrix3d_to_tuple(rotation_value),
                translation=_float3_to_tuple(translation_value),
            ),
        )
    return frames


def read_point_cloud_frames(
    stage: Any,
    *,
    point_cloud_frames_root: str = "/World/W2D/Reconstruction/PointCloudFrames",
    default_fps: float = 30.0,
    points_asset_base_dir: Path | None = None,
    points_asset_max_points: int = 20_000,
) -> dict[int, PointCloudFrame]:
    frames_root = stage.GetPrimAtPath(point_cloud_frames_root)
    if not frames_root:
        raise ValueError(f"Point-cloud frames root not found: {point_cloud_frames_root}")

    frames: dict[int, PointCloudFrame] = {}
    for prim in sorted(frames_root.GetChildren(), key=_frame_index_from_prim):
        frame_index = _frame_index_from_prim(prim)
        timestamp_s = _timestamp_from_prim(prim, frame_index=frame_index, default_fps=default_fps)
        points_attr = prim.GetAttribute("w2d:points")
        points_world: tuple[tuple[float, float, float], ...]
        if points_attr and points_attr.HasAuthoredValueOpinion():
            points_value = points_attr.Get()
            if points_value is None:
                points_value = []
            points_world = tuple(_float3_to_tuple(point) for point in points_value)
        else:
            points_asset_attr = prim.GetAttribute("w2d:pointsAsset")
            if not points_asset_attr or not points_asset_attr.HasAuthoredValueOpinion():
                raise ValueError(
                    f"Point-cloud frame {prim.GetPath()} must contain w2d:points or w2d:pointsAsset"
                )
            points_asset_value = points_asset_attr.Get()
            if points_asset_value is None:
                points_world = ()
            else:
                asset_path = _resolve_points_asset_path(
                    points_asset_value,
                    base_dir=points_asset_base_dir,
                )
                points_world = _load_points_from_asset(
                    asset_path,
                    max_points=max(0, int(points_asset_max_points)),
                )
        frames[frame_index] = PointCloudFrame(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            points_world=points_world,
        )
    return frames


def read_yolo_detection_frames(
    stage: Any,
    *,
    yolo_frames_root: str = "/World/W2D/Observations/YOLO/Frames",
    default_fps: float = 30.0,
) -> dict[int, tuple[float, tuple[Detection2D, ...]]]:
    frames_root = stage.GetPrimAtPath(yolo_frames_root)
    if not frames_root:
        raise ValueError(f"YOLO frames root not found: {yolo_frames_root}")

    yolo_frames: dict[int, tuple[float, tuple[Detection2D, ...]]] = {}
    for prim in sorted(frames_root.GetChildren(), key=_frame_index_from_prim):
        frame_index = _frame_index_from_prim(prim)
        timestamp_s = _timestamp_from_prim(prim, frame_index=frame_index, default_fps=default_fps)
        detections = _read_yolo_detections_from_frame_prim(prim)
        yolo_frames[frame_index] = (timestamp_s, detections)
    return yolo_frames


def _read_yolo_detections_from_frame_prim(prim: Any) -> tuple[Detection2D, ...]:
    detections = _read_yolo_arrays_from_frame_prim(prim)
    if detections:
        return tuple(detections)
    return tuple(_read_yolo_children_from_frame_prim(prim))


def _read_yolo_arrays_from_frame_prim(prim: Any) -> list[Detection2D]:
    labels_attr = prim.GetAttribute("w2d:labels")
    boxes_attr = prim.GetAttribute("w2d:boxesXYXY")
    scores_attr = prim.GetAttribute("w2d:scores")
    if not labels_attr or not boxes_attr:
        return []

    labels = labels_attr.Get()
    boxes = boxes_attr.Get()
    scores = scores_attr.Get() if scores_attr else None
    if labels is None or boxes is None:
        return []

    detections: list[Detection2D] = []
    for index in range(min(len(labels), len(boxes))):
        label = str(labels[index])
        score = float(scores[index]) if scores is not None and index < len(scores) else 1.0
        x_min, y_min, x_max, y_max = _float4_to_tuple(boxes[index])
        try:
            aabb = AABB2D(
                x_min=float(x_min),
                y_min=float(y_min),
                x_max=float(x_max),
                y_max=float(y_max),
            )
        except ValueError:
            continue
        detections.append(Detection2D(label=label, aabb=aabb, confidence=score))
    return detections


def _read_yolo_children_from_frame_prim(prim: Any) -> list[Detection2D]:
    detections: list[Detection2D] = []
    for det_prim in sorted(prim.GetChildren(), key=lambda item: item.GetName()):
        label_attr = det_prim.GetAttribute("w2d:class")
        bbox_attr = det_prim.GetAttribute("w2d:bboxXYXY")
        if not label_attr or not bbox_attr:
            continue

        label = label_attr.Get()
        bbox = bbox_attr.Get()
        if label is None or bbox is None:
            continue
        score_attr = det_prim.GetAttribute("w2d:confidence")
        score = float(score_attr.Get()) if score_attr else 1.0
        x_min, y_min, x_max, y_max = _float4_to_tuple(bbox)
        try:
            aabb = AABB2D(
                x_min=float(x_min),
                y_min=float(y_min),
                x_max=float(x_max),
                y_max=float(y_max),
            )
        except ValueError:
            continue
        detections.append(Detection2D(label=str(label), aabb=aabb, confidence=score))
    return detections


def run_particle_filter_from_stage(
    stage: Any,
    *,
    config: ParticleFilterConfig | None = None,
    seed: int = 7,
    calibration_camera_prim: str = "/World/W2D/Sensors/CalibrationCamera",
    camera_poses_frames_root: str = "/World/W2D/Sensors/CameraPoses/Frames",
    yolo_frames_root: str = "/World/W2D/Observations/YOLO/Frames",
    point_cloud_frames_root: str = "/World/W2D/Reconstruction/PointCloudFrames",
    points_asset_base_dir: Path | None = None,
    points_asset_max_points: int = 20_000,
    debug: bool = False,
    debug_every_n_frames: int = 10,
    early_stop_no_match_frames: int = 0,
    early_stop_no_stitch_frames: int = 0,
) -> list[ParticleFilterFrameResult]:
    tps = float(stage.GetTimeCodesPerSecond() or 30.0)
    if tps <= 0.0:
        tps = 30.0

    intrinsics = read_calibration_intrinsics(
        stage,
        calibration_camera_prim=calibration_camera_prim,
    )
    camera_frames = read_camera_pose_frames(
        stage,
        camera_poses_frames_root=camera_poses_frames_root,
        default_fps=tps,
    )
    yolo_frames = read_yolo_detection_frames(
        stage,
        yolo_frames_root=yolo_frames_root,
        default_fps=tps,
    )
    point_cloud_frames = read_point_cloud_frames(
        stage,
        point_cloud_frames_root=point_cloud_frames_root,
        default_fps=tps,
        points_asset_base_dir=points_asset_base_dir,
        points_asset_max_points=points_asset_max_points,
    )

    tracker = MultiObjectParticleFilter(
        config=config or ParticleFilterConfig(),
        rng=random.Random(seed),
    )
    results: list[ParticleFilterFrameResult] = []

    frame_indices = sorted(set(camera_frames.keys()) | set(yolo_frames.keys()) | set(point_cloud_frames.keys()))
    previous_timestamp = 0.0
    has_previous = False
    no_match_streak = 0
    no_stitch_streak = 0
    processed_frame_count = 0
    debug_stride = max(1, int(debug_every_n_frames))
    for frame_index in frame_indices:
        camera_frame = camera_frames.get(frame_index)
        if camera_frame is None:
            # Cannot project/back-project without camera pose for this frame.
            continue

        yolo_entry = yolo_frames.get(frame_index)
        detections = yolo_entry[1] if yolo_entry is not None else ()
        point_cloud_entry = point_cloud_frames.get(frame_index)
        point_cloud_points = point_cloud_entry.points_world if point_cloud_entry is not None else ()

        timestamp_candidates = [camera_frame.timestamp_s]
        if yolo_entry is not None:
            timestamp_candidates.append(float(yolo_entry[0]))
        if point_cloud_entry is not None:
            timestamp_candidates.append(float(point_cloud_entry.timestamp_s))
        timestamp_s = min(timestamp_candidates)

        if has_previous:
            dt_s = max(1.0 / tps, timestamp_s - previous_timestamp)
        else:
            dt_s = 1.0 / tps
            has_previous = True
        previous_timestamp = timestamp_s

        frame = FrameContext(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            dt_s=dt_s,
            camera_pose=camera_frame.pose,
            camera_intrinsics=intrinsics,
            detections=detections,
            point_cloud_points=point_cloud_points,
        )
        filter_result = tracker.step(frame)
        results.append(
            ParticleFilterFrameResult(
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                filter_result=filter_result,
            )
        )
        processed_frame_count += 1

        diagnostics = dict(filter_result.diagnostics)
        if int(diagnostics.get("matched_existing_tracks", 0)) <= 0 and int(
            diagnostics.get("tracks_before_update", 0)
        ) > 0:
            no_match_streak += 1
        else:
            no_match_streak = 0

        if int(diagnostics.get("stitched_track_count", 0)) <= 0 and int(
            diagnostics.get("tracks_after_update", 0)
        ) > 0:
            no_stitch_streak += 1
        else:
            no_stitch_streak = 0

        if debug and (
            processed_frame_count == 1
            or processed_frame_count % debug_stride == 0
        ):
            print(
                "[pf-adapter] "
                f"frame={frame_index} "
                f"detections={int(diagnostics.get('detections', 0))} "
                f"tracks_before={int(diagnostics.get('tracks_before_update', 0))} "
                f"matched_existing={int(diagnostics.get('matched_existing_tracks', 0))} "
                f"spawned={int(diagnostics.get('spawned_tracks', 0))} "
                f"spawned_from_cloud={int(diagnostics.get('spawned_from_cloud_tracks', 0))} "
                f"spawned_fallback={int(diagnostics.get('spawned_fallback_tracks', 0))} "
                f"spawned_cloud_point_matches={int(diagnostics.get('spawned_cloud_point_matches', 0))} "
                f"removed={int(diagnostics.get('removed_tracks', 0))} "
                f"tracks_after={int(diagnostics.get('tracks_after_update', 0))} "
                f"stitched_tracks={int(diagnostics.get('stitched_track_count', 0))} "
                f"stitched_points={int(diagnostics.get('stitched_point_count', 0))} "
                f"no_match_streak={no_match_streak} "
                f"no_stitch_streak={no_stitch_streak}",
                flush=True,
            )

        early_stop_reason = ""
        if (
            int(early_stop_no_match_frames) > 0
            and no_match_streak >= int(early_stop_no_match_frames)
        ):
            early_stop_reason = (
                f"matched_existing_tracks=0 for {no_match_streak} consecutive frames"
            )
        if (
            not early_stop_reason
            and int(early_stop_no_stitch_frames) > 0
            and no_stitch_streak >= int(early_stop_no_stitch_frames)
        ):
            early_stop_reason = (
                f"stitched_track_count=0 for {no_stitch_streak} consecutive frames"
            )
        if early_stop_reason:
            if debug:
                print(
                    "[pf-adapter] early-stop "
                    f"frame={frame_index} reason={early_stop_reason}",
                    flush=True,
                )
            break
    return results


def particle_filter_results_to_stage(
    *,
    frame_results: Sequence[ParticleFilterFrameResult],
    run_id: str,
    model_name: str = "world2data_particle_filter",
    model_version: str = "0.1",
    params: dict[str, Any] | None = None,
    git_commit: str | None = None,
    time_codes_per_second: float = 30.0,
    merged_point_cloud_assets_by_track: dict[str, str] | None = None,
) -> Any:
    Gf, Sdf, Usd, UsdGeom, Vt = _require_pxr()

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(float(time_codes_per_second))

    frame_indices = [frame.frame_index for frame in frame_results]
    if frame_indices:
        stage.SetStartTimeCode(float(min(frame_indices)))
        stage.SetEndTimeCode(float(max(frame_indices)))
    else:
        stage.SetStartTimeCode(0.0)
        stage.SetEndTimeCode(0.0)

    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)

    UsdGeom.Scope.Define(stage, "/World/W2D")
    entities_scope = UsdGeom.Scope.Define(stage, "/World/W2D/Entities").GetPrim()
    UsdGeom.Scope.Define(stage, "/World/W2D/Entities/Objects")
    UsdGeom.Scope.Define(stage, "/World/W2D/Reconstruction")
    merged_cloud_scope = UsdGeom.Scope.Define(
        stage,
        "/World/W2D/Reconstruction/MergedTrackPointClouds",
    ).GetPrim()
    tracks_scope = UsdGeom.Scope.Define(stage, "/World/W2D/Tracks").GetPrim()
    pf_scope = UsdGeom.Scope.Define(stage, "/World/W2D/Tracks/ParticleFilter").GetPrim()
    UsdGeom.Scope.Define(stage, "/World/W2D/Provenance")
    UsdGeom.Scope.Define(stage, "/World/W2D/Provenance/runs")

    tracks_rel = pf_scope.CreateRelationship("w2d:tracks", custom=True)
    entities_rel = entities_scope.CreateRelationship("w2d:objects", custom=True)
    merged_clouds_rel = merged_cloud_scope.CreateRelationship("w2d:mergedPointClouds", custom=True)

    track_samples: dict[str, list[tuple[int, Any]]] = {}
    track_labels: dict[str, str] = {}
    track_histories: dict[str, dict[str, int]] = {}
    track_stitched_points: dict[str, tuple[tuple[float, float, float], ...]] = {}
    for frame_result in frame_results:
        result = frame_result.filter_result
        for track_id, estimate in result.estimates.items():
            track_samples.setdefault(track_id, []).append((frame_result.frame_index, estimate))
            track_labels.setdefault(track_id, estimate.label)
        for track_id, label_counts in result.track_label_counts_by_track.items():
            track_histories[track_id] = dict(label_counts)
        for track_id, stitched_points in result.stitched_points_by_track.items():
            track_stitched_points[track_id] = tuple(stitched_points)

    for track_id in sorted(track_samples.keys()):
        safe = _safe_prim_name(track_id)
        label = track_labels.get(track_id, "unknown")
        entity_path = f"/World/W2D/Entities/Objects/{safe}"
        track_path = f"/World/W2D/Tracks/ParticleFilter/{safe}"
        merged_cloud_path = f"/World/W2D/Reconstruction/MergedTrackPointClouds/{safe}"

        entity_prim = UsdGeom.Scope.Define(stage, entity_path).GetPrim()
        entity_prim.CreateAttribute("w2d:uid", Sdf.ValueTypeNames.String, custom=True).Set(track_id)
        entity_prim.CreateAttribute("w2d:class", Sdf.ValueTypeNames.String, custom=True).Set(label)
        entity_prim.CreateAttribute(
            "w2d:producedByRunId",
            Sdf.ValueTypeNames.String,
            custom=True,
        ).Set(run_id)
        entity_prim.CreateRelationship("w2d:track", custom=True).AddTarget(track_path)
        entity_prim.CreateRelationship("w2d:mergedPointCloud", custom=True).AddTarget(
            merged_cloud_path
        )
        entities_rel.AddTarget(entity_prim.GetPath())

        track_prim = UsdGeom.Xform.Define(stage, track_path).GetPrim()
        xformable = UsdGeom.Xformable(track_prim)
        translate_op = xformable.AddTranslateOp()
        bbox_attr = track_prim.CreateAttribute(
            "w2d:meanBoundingBox",
            Sdf.ValueTypeNames.Float3,
            custom=True,
        )
        particle_count_attr = track_prim.CreateAttribute(
            "w2d:particleCount",
            Sdf.ValueTypeNames.Int,
            custom=True,
        )
        track_prim.CreateAttribute("w2d:trackId", Sdf.ValueTypeNames.String, custom=True).Set(
            track_id
        )
        track_prim.CreateAttribute("w2d:class", Sdf.ValueTypeNames.String, custom=True).Set(
            label
        )
        track_prim.CreateAttribute(
            "w2d:producedByRunId",
            Sdf.ValueTypeNames.String,
            custom=True,
        ).Set(run_id)

        class_history = track_histories.get(track_id, {label: 1})
        track_prim.CreateAttribute(
            "w2d:classHistoryJson",
            Sdf.ValueTypeNames.String,
            custom=True,
        ).Set(json.dumps(class_history, sort_keys=True))
        track_prim.CreateRelationship("w2d:mergedPointCloud", custom=True).AddTarget(
            merged_cloud_path
        )

        merged_points_prim = UsdGeom.Scope.Define(stage, merged_cloud_path).GetPrim()
        stitched_points = track_stitched_points.get(track_id, ())
        merged_points_prim.CreateAttribute("w2d:trackId", Sdf.ValueTypeNames.String, custom=True).Set(
            track_id
        )
        merged_points_prim.CreateAttribute("w2d:class", Sdf.ValueTypeNames.String, custom=True).Set(
            label
        )
        merged_points_prim.CreateAttribute(
            "w2d:producedByRunId",
            Sdf.ValueTypeNames.String,
            custom=True,
        ).Set(run_id)
        merged_points_prim.CreateAttribute(
            "w2d:pointCount",
            Sdf.ValueTypeNames.Int,
            custom=True,
        ).Set(int(len(stitched_points)))
        merged_points_prim.CreateAttribute(
            "w2d:pointsFormat",
            Sdf.ValueTypeNames.String,
            custom=True,
        ).Set("npy")
        if merged_point_cloud_assets_by_track is not None and track_id in merged_point_cloud_assets_by_track:
            merged_points_prim.CreateAttribute(
                "w2d:pointsAsset",
                Sdf.ValueTypeNames.Asset,
                custom=True,
            ).Set(Sdf.AssetPath(str(merged_point_cloud_assets_by_track[track_id])))
        merged_clouds_rel.AddTarget(merged_points_prim.GetPath())

        for frame_index, estimate in sorted(track_samples[track_id], key=lambda item: item[0]):
            time_code = float(frame_index)
            translate_op.Set(
                Gf.Vec3d(
                    float(estimate.position[0]),
                    float(estimate.position[1]),
                    float(estimate.position[2]),
                ),
                time_code,
            )
            bbox_attr.Set(
                Gf.Vec3f(
                    float(estimate.bounding_box[0]),
                    float(estimate.bounding_box[1]),
                    float(estimate.bounding_box[2]),
                ),
                time_code,
            )
            particle_count_attr.Set(int(estimate.particle_count), time_code)

        tracks_rel.AddTarget(track_prim.GetPath())

    pf_scope.CreateAttribute("w2d:component", Sdf.ValueTypeNames.String, custom=True).Set(
        "tracking.particle_filter"
    )
    pf_scope.CreateAttribute("w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True).Set(
        run_id
    )
    pf_scope.CreateAttribute("w2d:trackCount", Sdf.ValueTypeNames.Int, custom=True).Set(
        len(track_samples)
    )
    pf_scope.CreateAttribute("w2d:processedFrameCount", Sdf.ValueTypeNames.Int, custom=True).Set(
        len(frame_results)
    )
    pf_scope.CreateAttribute(
        "w2d:mergedPointCloudTrackCount",
        Sdf.ValueTypeNames.Int,
        custom=True,
    ).Set(len(track_stitched_points))
    merged_cloud_scope.CreateAttribute("w2d:component", Sdf.ValueTypeNames.String, custom=True).Set(
        "tracking.particle_filter.merged_point_clouds"
    )
    merged_cloud_scope.CreateAttribute(
        "w2d:producedByRunId",
        Sdf.ValueTypeNames.String,
        custom=True,
    ).Set(run_id)
    merged_cloud_scope.CreateAttribute("w2d:trackCount", Sdf.ValueTypeNames.Int, custom=True).Set(
        len(track_stitched_points)
    )

    run_prim = UsdGeom.Scope.Define(
        stage, f"/World/W2D/Provenance/runs/{_safe_prim_name(run_id)}"
    ).GetPrim()
    run_prim.CreateAttribute("w2d:runId", Sdf.ValueTypeNames.String, custom=True).Set(run_id)
    run_prim.CreateAttribute("w2d:component", Sdf.ValueTypeNames.String, custom=True).Set(
        "tracking.particle_filter"
    )
    run_prim.CreateAttribute("w2d:modelName", Sdf.ValueTypeNames.String, custom=True).Set(
        model_name
    )
    run_prim.CreateAttribute("w2d:modelVersion", Sdf.ValueTypeNames.String, custom=True).Set(
        model_version
    )
    run_prim.CreateAttribute("w2d:gitCommit", Sdf.ValueTypeNames.String, custom=True).Set(
        git_commit if git_commit is not None else _git_commit_or_unknown()
    )
    run_prim.CreateAttribute(
        "w2d:timestampIso8601",
        Sdf.ValueTypeNames.String,
        custom=True,
    ).Set(datetime.now(timezone.utc).isoformat())
    run_prim.CreateAttribute("w2d:params", Sdf.ValueTypeNames.String, custom=True).Set(
        json.dumps(params or {}, sort_keys=True)
    )

    return stage


def write_particle_filter_tracks_usd(
    *,
    calibration_layer: Path,
    camera_poses_layer: Path,
    yolo_layer: Path,
    point_cloud_layer: Path,
    output_path: Path,
    config: ParticleFilterConfig | None = None,
    seed: int = 7,
    run_id: str,
    time_codes_per_second: float = 0.0,
    calibration_camera_prim: str = "/World/W2D/Sensors/CalibrationCamera",
    camera_poses_frames_root: str = "/World/W2D/Sensors/CameraPoses/Frames",
    yolo_frames_root: str = "/World/W2D/Observations/YOLO/Frames",
    point_cloud_frames_root: str = "/World/W2D/Reconstruction/PointCloudFrames",
    points_asset_max_points: int = 20_000,
    debug: bool = False,
    debug_every_n_frames: int = 10,
    early_stop_no_match_frames: int = 0,
    early_stop_no_stitch_frames: int = 0,
) -> None:
    stage = _open_composed_stage(
        calibration_layer=calibration_layer,
        camera_poses_layer=camera_poses_layer,
        yolo_layer=yolo_layer,
        point_cloud_layer=point_cloud_layer,
    )
    frame_results = run_particle_filter_from_stage(
        stage,
        config=config,
        seed=seed,
        calibration_camera_prim=calibration_camera_prim,
        camera_poses_frames_root=camera_poses_frames_root,
        yolo_frames_root=yolo_frames_root,
        point_cloud_frames_root=point_cloud_frames_root,
        points_asset_base_dir=point_cloud_layer.resolve().parent,
        points_asset_max_points=points_asset_max_points,
        debug=debug,
        debug_every_n_frames=debug_every_n_frames,
        early_stop_no_match_frames=early_stop_no_match_frames,
        early_stop_no_stitch_frames=early_stop_no_stitch_frames,
    )
    tps = float(time_codes_per_second)
    if tps <= 0.0:
        tps = float(stage.GetTimeCodesPerSecond() or 30.0)

    effective_config = config or ParticleFilterConfig()
    params = {
        "calibration_layer": str(calibration_layer),
        "camera_poses_layer": str(camera_poses_layer),
        "yolo_layer": str(yolo_layer),
        "point_cloud_layer": str(point_cloud_layer),
        "points_asset_max_points": int(points_asset_max_points),
        "assignment_workers": int(effective_config.assignment_workers),
        "particle_score_workers": int(effective_config.particle_score_workers),
        "debug": bool(debug),
        "debug_every_n_frames": int(debug_every_n_frames),
        "early_stop_no_match_frames": int(early_stop_no_match_frames),
        "early_stop_no_stitch_frames": int(early_stop_no_stitch_frames),
        "seed": int(seed),
    }
    merged_assets_by_track = _write_merged_track_point_cloud_blobs(
        frame_results=frame_results,
        output_path=output_path,
    )
    params["merged_point_cloud_blob_count"] = int(len(merged_assets_by_track))
    params["merged_point_cloud_blob_format"] = "npy"
    params["merged_point_cloud_blob_dir"] = f"{output_path.stem}_merged_track_point_clouds"
    out_stage = particle_filter_results_to_stage(
        frame_results=frame_results,
        run_id=run_id,
        model_name="world2data_particle_filter",
        model_version="0.1",
        params=params,
        time_codes_per_second=tps,
        merged_point_cloud_assets_by_track=merged_assets_by_track,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_stage.GetRootLayer().Export(str(output_path))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read protocol layers (calibration, camera poses, YOLO, point cloud), "
            "run particle filtering, and write a tracks layer."
        )
    )
    parser.add_argument("--calibration-layer", type=Path, required=True)
    parser.add_argument("--camera-poses-layer", type=Path, required=True)
    parser.add_argument("--yolo-layer", type=Path, required=True)
    parser.add_argument("--point-cloud-layer", type=Path, required=True)
    parser.add_argument(
        "--output-usd",
        type=Path,
        default=Path("outputs/particle_tracks.usda"),
        help="output OpenUSD tracking layer",
    )
    parser.add_argument("--run-id", type=str, default="", help="provenance run id")
    parser.add_argument("--seed", type=int, default=7, help="rng seed for particle filter")
    parser.add_argument(
        "--particles-per-track",
        type=int,
        default=96,
        help="particle count per track",
    )
    parser.add_argument(
        "--min-assignment-iou",
        type=float,
        default=0.10,
        help="minimum IoU for primary assignment",
    )
    parser.add_argument(
        "--point-cloud-backend",
        type=str,
        default="torch",
        help="point-cloud backend to use: torch, gpu, cuda, pcl, or numpy",
    )
    parser.add_argument(
        "--timecodes-per-second",
        type=float,
        default=0.0,
        help="output stage timeCodesPerSecond; 0 means infer from input stage",
    )
    parser.add_argument(
        "--points-asset-max-points",
        type=int,
        default=20_000,
        help=(
            "for point-cloud frames that reference w2d:pointsAsset (e.g. PLY), "
            "load at most this many points per frame; 0 means load all"
        ),
    )
    parser.add_argument(
        "--assignment-workers",
        type=int,
        default=1,
        help="worker threads for assignment candidate building/predict; 0 means auto by CPU count",
    )
    parser.add_argument(
        "--particle-score-workers",
        type=int,
        default=1,
        help="worker threads for per-track particle scoring; 0 means auto by CPU count",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print per-frame tracking/stitching diagnostics while running",
    )
    parser.add_argument(
        "--debug-every-n-frames",
        type=int,
        default=10,
        help="when --debug is set, print one diagnostic line every N processed frames",
    )
    parser.add_argument(
        "--early-stop-no-match-frames",
        type=int,
        default=0,
        help=(
            "early-stop when matched_existing_tracks stays at 0 for N consecutive processed frames; "
            "0 disables"
        ),
    )
    parser.add_argument(
        "--early-stop-no-stitch-frames",
        type=int,
        default=0,
        help=(
            "early-stop when stitched_track_count stays at 0 for N consecutive processed frames; "
            "0 disables"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config = ParticleFilterConfig(
        particles_per_track=int(args.particles_per_track),
        min_assignment_iou=float(args.min_assignment_iou),
        point_cloud_backend=str(args.point_cloud_backend),
        assignment_workers=int(args.assignment_workers),
        particle_score_workers=int(args.particle_score_workers),
    )
    write_particle_filter_tracks_usd(
        calibration_layer=args.calibration_layer,
        camera_poses_layer=args.camera_poses_layer,
        yolo_layer=args.yolo_layer,
        point_cloud_layer=args.point_cloud_layer,
        output_path=args.output_usd,
        config=config,
        seed=int(args.seed),
        run_id=run_id,
        time_codes_per_second=float(args.timecodes_per_second),
        points_asset_max_points=int(args.points_asset_max_points),
        debug=bool(args.debug),
        debug_every_n_frames=int(args.debug_every_n_frames),
        early_stop_no_match_frames=int(args.early_stop_no_match_frames),
        early_stop_no_stitch_frames=int(args.early_stop_no_stitch_frames),
    )
    print(f"wrote particle tracks layer: {args.output_usd}")
    print(f"run_id={run_id}")


if __name__ == "__main__":
    main()
