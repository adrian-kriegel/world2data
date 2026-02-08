from __future__ import annotations

"""Open3D surface-reconstruction adapter from PF stitched point clouds."""

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_]")
_FRAME_INDEX_RE = re.compile(r"(\d+)$")


def _safe_prim_name(raw: str) -> str:
    value = _SAFE_NAME_RE.sub("_", raw).strip("_")
    if not value:
        value = "item"
    if value[0].isdigit():
        value = f"n_{value}"
    return value


def _frame_index_from_prim(prim: Any) -> int:
    frame_attr = prim.GetAttribute("w2d:frameIndex")
    if frame_attr and frame_attr.HasAuthoredValueOpinion():
        value = frame_attr.Get()
        if value is not None:
            return int(value)
    match = _FRAME_INDEX_RE.search(prim.GetName())
    if match:
        return int(match.group(1))
    return 0


def _require_pxr() -> tuple[Any, Any, Any, Any]:
    try:
        from pxr import Sdf, Usd, UsdGeom, Vt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("OpenUSD Python bindings are required. Install `usd-core`.") from exc
    return Sdf, Usd, UsdGeom, Vt


def _require_open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover
        raise ImportError("open3d is required for mesh reconstruction. Install `open3d`.") from exc
    return o3d


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


def _asset_value_to_path_string(value: Any) -> str:
    resolved = str(getattr(value, "resolvedPath", "") or "").strip()
    if resolved:
        return resolved
    authored = str(getattr(value, "path", "") or "").strip()
    if authored:
        return authored
    return str(value).strip()


def _resolve_points_asset_path(asset_value: Any, *, base_dir: Path | None) -> Path:
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


def _asset_path_relative_to_layer(*, layer_path: Path, asset_path: Path) -> str:
    relative = Path(
        str(asset_path.resolve().relative_to(asset_path.resolve().anchor))
    )
    del relative
    rel = Path(
        __import__("os").path.relpath(str(asset_path.resolve()), str(layer_path.resolve().parent))
    )
    return rel.as_posix()


def _to_float3_array(points_value: Any) -> np.ndarray:
    if points_value is None:
        return np.empty((0, 3), dtype=np.float64)
    points = np.asarray(
        [[float(point[0]), float(point[1]), float(point[2])] for point in points_value],
        dtype=np.float64,
    )
    if points.ndim != 2 or points.shape[1] != 3:
        return np.empty((0, 3), dtype=np.float64)
    return points


def _to_color_array(colors_value: Any, *, expected_count: int) -> np.ndarray | None:
    if colors_value is None:
        return None
    colors = np.asarray(
        [[float(value[0]), float(value[1]), float(value[2])] for value in colors_value],
        dtype=np.float64,
    )
    if colors.ndim != 2 or colors.shape[1] != 3:
        return None
    if len(colors) == expected_count:
        return np.clip(colors, 0.0, 1.0)
    if len(colors) == 1 and expected_count > 0:
        return np.repeat(colors, expected_count, axis=0)
    return None


def _stable_downsample(points: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    indices = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int64)
    return points[indices], colors[indices]


def _hash_points_and_colors(points: np.ndarray, colors: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(np.asarray(points, dtype=np.float32).tobytes())
    hasher.update(np.asarray(colors, dtype=np.float32).tobytes())
    return hasher.hexdigest()


def _deterministic_track_color(track_id: str) -> np.ndarray:
    digest = hashlib.sha256(track_id.encode("utf-8")).digest()
    rgb = np.asarray([digest[0], digest[1], digest[2]], dtype=np.float64) / 255.0
    return np.clip(0.2 + 0.8 * rgb, 0.0, 1.0)


def _nearest_neighbor_indices(source_points: np.ndarray, query_points: np.ndarray) -> np.ndarray:
    if len(source_points) == 0 or len(query_points) == 0:
        return np.empty((0,), dtype=np.int64)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(source_points)
        _distances, indices = tree.query(query_points, k=1)
        return np.asarray(indices, dtype=np.int64)
    except Exception:
        chunk = 2048
        indices = np.empty((len(query_points),), dtype=np.int64)
        for start in range(0, len(query_points), chunk):
            stop = min(start + chunk, len(query_points))
            q = query_points[start:stop]
            deltas = q[:, None, :] - source_points[None, :, :]
            distances = np.linalg.norm(deltas, axis=2)
            indices[start:stop] = np.argmin(distances, axis=1)
        return indices


def _mean_edge_length(vertices: np.ndarray, triangles: np.ndarray) -> float:
    if len(vertices) == 0 or len(triangles) == 0:
        return 0.0
    tri = triangles.astype(np.int64)
    v0 = vertices[tri[:, 0]]
    v1 = vertices[tri[:, 1]]
    v2 = vertices[tri[:, 2]]
    l01 = np.linalg.norm(v0 - v1, axis=1)
    l12 = np.linalg.norm(v1 - v2, axis=1)
    l20 = np.linalg.norm(v2 - v0, axis=1)
    return float(np.mean(np.concatenate([l01, l12, l20])))


def _estimate_normal_radius(points: np.ndarray, *, explicit_radius_m: float) -> float:
    if explicit_radius_m > 0.0:
        return float(explicit_radius_m)
    if len(points) == 0:
        return 0.05
    bounds = np.ptp(points, axis=0)
    diag = float(np.linalg.norm(bounds))
    return max(diag * 0.02, 1e-3)


@dataclass(frozen=True)
class TrackPointCloud:
    track_id: str
    label: str
    safe_name: str
    points: np.ndarray
    colors: np.ndarray | None


@dataclass(frozen=True)
class MeshArtifact:
    track_id: str
    label: str
    safe_name: str
    point_count: int
    vertex_count: int
    face_count: int
    reconstruction_method: str
    poisson_depth: int
    mesh_surface_area_m2: float
    mean_edge_length: float
    source_points_sha256: str
    open3d_version: str
    points_asset_path: Path
    mesh_asset_path: Path


def read_pf_stitched_track_clouds(
    stage: Any,
    *,
    merged_cloud_root: str = "/World/W2D/Reconstruction/MergedTrackPointClouds",
) -> list[TrackPointCloud]:
    root = stage.GetPrimAtPath(merged_cloud_root)
    if not root:
        raise ValueError(f"PF stitched cloud root not found: {merged_cloud_root}")

    tracks: list[TrackPointCloud] = []
    for prim in sorted(root.GetChildren(), key=lambda item: item.GetName()):
        points_attr = prim.GetAttribute("points")
        points_value = points_attr.Get() if points_attr else None
        points = _to_float3_array(points_value)
        if len(points) == 0:
            continue

        track_id_attr = prim.GetAttribute("w2d:trackId")
        label_attr = prim.GetAttribute("w2d:class")
        track_id = str(track_id_attr.Get()) if track_id_attr and track_id_attr.Get() is not None else prim.GetName()
        label = str(label_attr.Get()) if label_attr and label_attr.Get() is not None else "unknown"
        safe = _safe_prim_name(track_id)

        colors_value = None
        for attr_name in ("primvars:displayColor", "displayColor"):
            color_attr = prim.GetAttribute(attr_name)
            if color_attr and color_attr.HasAuthoredValueOpinion():
                colors_value = color_attr.Get()
                if colors_value is not None:
                    break
        colors = _to_color_array(colors_value, expected_count=len(points))

        tracks.append(
            TrackPointCloud(
                track_id=track_id,
                label=label,
                safe_name=safe,
                points=points,
                colors=colors,
            )
        )
    if not tracks:
        raise ValueError(f"No stitched PF points found under {merged_cloud_root}")
    return tracks


def _collect_color_source_from_point_cloud_frames(
    stage: Any,
    *,
    point_cloud_frames_root: str = "/World/W2D/Reconstruction/PointCloudFrames",
    points_asset_base_dir: Path | None = None,
    max_points: int = 500_000,
) -> tuple[np.ndarray, np.ndarray]:
    o3d = _require_open3d()
    frames_root = stage.GetPrimAtPath(point_cloud_frames_root)
    if not frames_root:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )

    seen_assets: set[str] = set()
    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    total = 0

    for frame_prim in sorted(frames_root.GetChildren(), key=_frame_index_from_prim):
        asset_attr = frame_prim.GetAttribute("w2d:pointsAsset")
        if not asset_attr or not asset_attr.HasAuthoredValueOpinion():
            continue
        asset_value = asset_attr.Get()
        if asset_value is None:
            continue
        path = _resolve_points_asset_path(asset_value, base_dir=points_asset_base_dir)
        key = str(path.resolve())
        if key in seen_assets:
            continue
        seen_assets.add(key)

        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points, dtype=np.float64)
        colors = np.asarray(pcd.colors, dtype=np.float64)
        if len(points) == 0 or len(colors) == 0 or len(points) != len(colors):
            continue

        if max_points > 0:
            remaining = max_points - total
            if remaining <= 0:
                break
            if len(points) > remaining:
                points, colors = _stable_downsample(points, colors, remaining)

        point_chunks.append(points)
        color_chunks.append(np.clip(colors, 0.0, 1.0))
        total += len(points)

    if not point_chunks:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )
    return (np.vstack(point_chunks), np.vstack(color_chunks))


def _materialize_colored_track_clouds(
    tracks: list[TrackPointCloud],
    *,
    source_points: np.ndarray,
    source_colors: np.ndarray,
    require_color: bool,
) -> list[TrackPointCloud]:
    result: list[TrackPointCloud] = []
    has_source = len(source_points) > 0 and len(source_points) == len(source_colors)
    for track in tracks:
        colors = track.colors
        if colors is None or len(colors) != len(track.points):
            if has_source:
                indices = _nearest_neighbor_indices(source_points, track.points)
                colors = source_colors[indices]
            elif require_color:
                raise ValueError(
                    f"Track '{track.track_id}' has no colors and no color source was available. "
                    "Provide a point-cloud layer with colored w2d:pointsAsset inputs or disable strict color enforcement."
                )
            else:
                colors = np.repeat(
                    _deterministic_track_color(track.track_id)[None, :],
                    len(track.points),
                    axis=0,
                )
        result.append(
            TrackPointCloud(
                track_id=track.track_id,
                label=track.label,
                safe_name=track.safe_name,
                points=track.points,
                colors=np.clip(colors, 0.0, 1.0),
            )
        )
    return result


def _to_open3d_point_cloud(o3d: Any, points: np.ndarray, colors: np.ndarray) -> Any:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _transfer_colors_to_mesh(mesh: Any, *, source_points: np.ndarray, source_colors: np.ndarray, o3d: Any) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        return
    indices = _nearest_neighbor_indices(source_points, vertices)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(source_colors[indices], 0.0, 1.0))


def _reconstruct_mesh_from_track_points(
    *,
    track: TrackPointCloud,
    poisson_depth: int,
    density_trim_quantile: float,
    normal_radius_m: float,
    normal_max_nn: int,
    normal_orient_k: int,
) -> tuple[Any, str]:
    o3d = _require_open3d()
    assert track.colors is not None
    pcd = _to_open3d_point_cloud(o3d, track.points, track.colors)

    radius = _estimate_normal_radius(track.points, explicit_radius_m=normal_radius_m)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(radius),
            max_nn=max(8, int(normal_max_nn)),
        )
    )
    if len(track.points) >= max(8, int(normal_orient_k)):
        pcd.orient_normals_consistent_tangent_plane(max(8, int(normal_orient_k)))

    mesh = None
    method = "poisson"
    densities_np: np.ndarray | None = None
    try:
        mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=max(6, int(poisson_depth)),
            linear_fit=False,
        )
        if len(mesh_poisson.vertices) > 0 and len(mesh_poisson.triangles) > 0:
            mesh = mesh_poisson
            densities_np = np.asarray(densities, dtype=np.float64)
    except Exception:
        mesh = None

    if mesh is not None and densities_np is not None and len(densities_np) == len(mesh.vertices):
        trim_q = float(np.clip(density_trim_quantile, 0.0, 0.49))
        if trim_q > 0.0:
            threshold = float(np.quantile(densities_np, trim_q))
            mesh.remove_vertices_by_mask(densities_np < threshold)

    if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        method = "ball_pivoting"
        distances = np.asarray(pcd.compute_nearest_neighbor_distance(), dtype=np.float64)
        mean_distance = float(np.mean(distances)) if len(distances) > 0 else 0.01
        r1 = max(mean_distance * 1.5, 1e-4)
        r2 = max(mean_distance * 3.0, r1 * 1.5)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector([r1, r2]),
        )
        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            raise ValueError(f"Could not reconstruct a mesh for track '{track.track_id}'")

    _transfer_colors_to_mesh(mesh, source_points=track.points, source_colors=track.colors, o3d=o3d)
    mesh.compute_vertex_normals()
    return mesh, method


def _write_colored_point_cloud_asset(*, track: TrackPointCloud, output_path: Path) -> None:
    o3d = _require_open3d()
    assert track.colors is not None
    pcd = _to_open3d_point_cloud(o3d, track.points, track.colors)
    ok = o3d.io.write_point_cloud(
        str(output_path),
        pcd,
        write_ascii=False,
        compressed=True,
    )
    if not ok:
        raise RuntimeError(f"Failed writing point cloud asset: {output_path}")


def _write_colored_mesh_asset(*, mesh: Any, output_path: Path) -> None:
    o3d = _require_open3d()
    ok = o3d.io.write_triangle_mesh(
        str(output_path),
        mesh,
        write_ascii=False,
        compressed=True,
        write_vertex_normals=True,
        write_vertex_colors=True,
    )
    if not ok:
        raise RuntimeError(f"Failed writing mesh asset: {output_path}")


def reconstruct_track_mesh_assets(
    *,
    tracks: list[TrackPointCloud],
    external_recon_dir: Path,
    poisson_depth: int = 11,
    density_trim_quantile: float = 0.01,
    normal_radius_m: float = 0.0,
    normal_max_nn: int = 48,
    normal_orient_k: int = 24,
    min_track_points: int = 50,
) -> list[MeshArtifact]:
    o3d = _require_open3d()
    stitched_dir = external_recon_dir / "stitched"
    meshes_dir = external_recon_dir / "meshes"
    stitched_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[MeshArtifact] = []
    for track in tracks:
        if len(track.points) < max(3, int(min_track_points)):
            continue
        assert track.colors is not None

        points_asset_path = stitched_dir / f"{track.safe_name}_points_colored.ply"
        mesh_asset_path = meshes_dir / f"{track.safe_name}_mesh_colored.ply"
        _write_colored_point_cloud_asset(track=track, output_path=points_asset_path)

        mesh, method = _reconstruct_mesh_from_track_points(
            track=track,
            poisson_depth=poisson_depth,
            density_trim_quantile=density_trim_quantile,
            normal_radius_m=normal_radius_m,
            normal_max_nn=normal_max_nn,
            normal_orient_k=normal_orient_k,
        )
        _write_colored_mesh_asset(mesh=mesh, output_path=mesh_asset_path)

        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        triangles = np.asarray(mesh.triangles, dtype=np.int64)
        try:
            surface_area = float(mesh.get_surface_area())
        except Exception:
            surface_area = 0.0

        artifacts.append(
            MeshArtifact(
                track_id=track.track_id,
                label=track.label,
                safe_name=track.safe_name,
                point_count=int(len(track.points)),
                vertex_count=int(len(vertices)),
                face_count=int(len(triangles)),
                reconstruction_method=method,
                poisson_depth=int(poisson_depth),
                mesh_surface_area_m2=surface_area,
                mean_edge_length=_mean_edge_length(vertices, triangles),
                source_points_sha256=_hash_points_and_colors(track.points, track.colors),
                open3d_version=str(getattr(o3d, "__version__", "unknown")),
                points_asset_path=points_asset_path,
                mesh_asset_path=mesh_asset_path,
            )
        )
    return artifacts


def mesh_artifacts_to_stage(
    *,
    artifacts: list[MeshArtifact],
    output_layer_path: Path,
    run_id: str,
    model_name: str,
    model_version: str,
    params: dict[str, Any],
    git_commit: str | None = None,
    time_codes_per_second: float = 30.0,
    start_time_code: float = 0.0,
    end_time_code: float = 0.0,
) -> Any:
    Sdf, Usd, UsdGeom, _Vt = _require_pxr()

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(float(time_codes_per_second))
    stage.SetStartTimeCode(float(start_time_code))
    stage.SetEndTimeCode(float(end_time_code))

    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)

    UsdGeom.Scope.Define(stage, "/World/W2D")
    UsdGeom.Scope.Define(stage, "/World/W2D/Reconstruction")
    stitched_scope = UsdGeom.Scope.Define(
        stage,
        "/World/W2D/Reconstruction/StitchedTrackPointClouds",
    ).GetPrim()
    meshes_scope = UsdGeom.Scope.Define(stage, "/World/W2D/Reconstruction/Meshes").GetPrim()
    UsdGeom.Scope.Define(stage, "/World/W2D/Provenance")
    UsdGeom.Scope.Define(stage, "/World/W2D/Provenance/runs")

    stitched_scope.CreateAttribute("w2d:component", Sdf.ValueTypeNames.String, custom=True).Set(
        "reconstruction.surface_mesh.open3d.stitched_track_point_clouds"
    )
    stitched_scope.CreateAttribute("w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True).Set(
        run_id
    )
    stitched_scope.CreateAttribute("w2d:trackCount", Sdf.ValueTypeNames.Int, custom=True).Set(
        int(len(artifacts))
    )

    meshes_scope.CreateAttribute("w2d:component", Sdf.ValueTypeNames.String, custom=True).Set(
        "reconstruction.surface_mesh.open3d"
    )
    meshes_scope.CreateAttribute("w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True).Set(
        run_id
    )
    meshes_scope.CreateAttribute("w2d:meshCount", Sdf.ValueTypeNames.Int, custom=True).Set(
        int(len(artifacts))
    )

    for artifact in sorted(artifacts, key=lambda item: item.track_id):
        stitched_path = f"/World/W2D/Reconstruction/StitchedTrackPointClouds/{artifact.safe_name}"
        mesh_path = f"/World/W2D/Reconstruction/Meshes/{artifact.safe_name}"
        track_path = f"/World/W2D/Tracks/ParticleFilter/{artifact.safe_name}"

        stitched_prim = UsdGeom.Scope.Define(stage, stitched_path).GetPrim()
        stitched_prim.CreateAttribute("w2d:trackId", Sdf.ValueTypeNames.String, custom=True).Set(
            artifact.track_id
        )
        stitched_prim.CreateAttribute("w2d:class", Sdf.ValueTypeNames.String, custom=True).Set(
            artifact.label
        )
        stitched_prim.CreateAttribute("w2d:pointsFormat", Sdf.ValueTypeNames.String, custom=True).Set(
            "ply"
        )
        stitched_prim.CreateAttribute("w2d:hasColor", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
        stitched_prim.CreateAttribute("w2d:colorEncoding", Sdf.ValueTypeNames.String, custom=True).Set(
            "rgb_f32"
        )
        stitched_prim.CreateAttribute("w2d:pointCount", Sdf.ValueTypeNames.Int, custom=True).Set(
            artifact.point_count
        )
        stitched_prim.CreateAttribute("w2d:pointsSha256", Sdf.ValueTypeNames.String, custom=True).Set(
            artifact.source_points_sha256
        )
        stitched_prim.CreateAttribute("w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True).Set(
            run_id
        )
        stitched_asset_rel = _asset_path_relative_to_layer(
            layer_path=output_layer_path,
            asset_path=artifact.points_asset_path,
        )
        stitched_prim.CreateAttribute("w2d:pointsAsset", Sdf.ValueTypeNames.Asset, custom=True).Set(
            Sdf.AssetPath(stitched_asset_rel)
        )

        mesh_prim = UsdGeom.Scope.Define(stage, mesh_path).GetPrim()
        mesh_prim.CreateAttribute("w2d:trackId", Sdf.ValueTypeNames.String, custom=True).Set(
            artifact.track_id
        )
        mesh_prim.CreateAttribute("w2d:class", Sdf.ValueTypeNames.String, custom=True).Set(
            artifact.label
        )
        mesh_prim.CreateAttribute("w2d:meshFormat", Sdf.ValueTypeNames.String, custom=True).Set("ply")
        mesh_prim.CreateAttribute("w2d:meshHasVertexColors", Sdf.ValueTypeNames.Bool, custom=True).Set(
            True
        )
        mesh_prim.CreateAttribute("w2d:sourcePointsHaveColor", Sdf.ValueTypeNames.Bool, custom=True).Set(
            True
        )
        mesh_prim.CreateAttribute("w2d:vertexCount", Sdf.ValueTypeNames.Int, custom=True).Set(
            artifact.vertex_count
        )
        mesh_prim.CreateAttribute("w2d:faceCount", Sdf.ValueTypeNames.Int, custom=True).Set(
            artifact.face_count
        )
        mesh_prim.CreateAttribute("w2d:reconstructionMethod", Sdf.ValueTypeNames.String, custom=True).Set(
            artifact.reconstruction_method
        )
        mesh_prim.CreateAttribute("w2d:poissonDepth", Sdf.ValueTypeNames.Int, custom=True).Set(
            artifact.poisson_depth
        )
        mesh_prim.CreateAttribute("w2d:meanEdgeLength", Sdf.ValueTypeNames.Float, custom=True).Set(
            float(artifact.mean_edge_length)
        )
        mesh_prim.CreateAttribute("w2d:meshSurfaceAreaM2", Sdf.ValueTypeNames.Float, custom=True).Set(
            float(artifact.mesh_surface_area_m2)
        )
        mesh_prim.CreateAttribute("w2d:sourcePointsSha256", Sdf.ValueTypeNames.String, custom=True).Set(
            artifact.source_points_sha256
        )
        mesh_prim.CreateAttribute("w2d:open3dVersion", Sdf.ValueTypeNames.String, custom=True).Set(
            artifact.open3d_version
        )
        mesh_prim.CreateAttribute("w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True).Set(
            run_id
        )
        mesh_asset_rel = _asset_path_relative_to_layer(
            layer_path=output_layer_path,
            asset_path=artifact.mesh_asset_path,
        )
        mesh_prim.CreateAttribute("w2d:meshAsset", Sdf.ValueTypeNames.Asset, custom=True).Set(
            Sdf.AssetPath(mesh_asset_rel)
        )
        mesh_prim.CreateRelationship("w2d:sourceStitchedPointCloud", custom=True).AddTarget(
            stitched_prim.GetPath()
        )
        mesh_prim.CreateRelationship("w2d:track", custom=True).AddTarget(track_path)

    run_prim = UsdGeom.Scope.Define(
        stage,
        f"/World/W2D/Provenance/runs/{_safe_prim_name(run_id)}",
    ).GetPrim()
    run_prim.CreateAttribute("w2d:runId", Sdf.ValueTypeNames.String, custom=True).Set(run_id)
    run_prim.CreateAttribute("w2d:component", Sdf.ValueTypeNames.String, custom=True).Set(
        "reconstruction.surface_mesh.open3d"
    )
    run_prim.CreateAttribute("w2d:modelName", Sdf.ValueTypeNames.String, custom=True).Set(model_name)
    run_prim.CreateAttribute("w2d:modelVersion", Sdf.ValueTypeNames.String, custom=True).Set(model_version)
    run_prim.CreateAttribute("w2d:gitCommit", Sdf.ValueTypeNames.String, custom=True).Set(
        git_commit if git_commit is not None else _git_commit_or_unknown()
    )
    run_prim.CreateAttribute("w2d:timestampIso8601", Sdf.ValueTypeNames.String, custom=True).Set(
        datetime.now(timezone.utc).isoformat()
    )
    run_prim.CreateAttribute("w2d:params", Sdf.ValueTypeNames.String, custom=True).Set(
        json.dumps(params, sort_keys=True)
    )

    return stage


def _open_composed_stage(*, tracks_layer: Path, point_cloud_layer: Path | None = None) -> Any:
    _Sdf, Usd, _UsdGeom, _Vt = _require_pxr()
    if not tracks_layer.exists():
        raise FileNotFoundError(f"Missing PF tracks layer: {tracks_layer}")
    if point_cloud_layer is not None and not point_cloud_layer.exists():
        raise FileNotFoundError(f"Missing point-cloud layer: {point_cloud_layer}")

    Sdf, _Usd, _UsdGeom2, _Vt2 = _require_pxr()
    root = Sdf.Layer.CreateAnonymous(".usda")
    sublayers = [str(tracks_layer.resolve())]
    if point_cloud_layer is not None:
        sublayers.append(str(point_cloud_layer.resolve()))
    root.subLayerPaths = sublayers
    stage = Usd.Stage.Open(root)
    if stage is None:
        raise RuntimeError("Could not open composed stage for mesh reconstruction")
    return stage


def write_open3d_mesh_layer(
    *,
    tracks_layer: Path,
    output_path: Path,
    run_id: str,
    point_cloud_layer: Path | None = None,
    external_recon_dir: Path | None = None,
    merged_cloud_root: str = "/World/W2D/Reconstruction/MergedTrackPointClouds",
    point_cloud_frames_root: str = "/World/W2D/Reconstruction/PointCloudFrames",
    require_color: bool = True,
    color_source_max_points: int = 500_000,
    poisson_depth: int = 11,
    density_trim_quantile: float = 0.01,
    normal_radius_m: float = 0.0,
    normal_max_nn: int = 48,
    normal_orient_k: int = 24,
    min_track_points: int = 50,
    time_codes_per_second: float = 0.0,
    debug: bool = False,
) -> None:
    stage = _open_composed_stage(tracks_layer=tracks_layer, point_cloud_layer=point_cloud_layer)

    tracks = read_pf_stitched_track_clouds(stage, merged_cloud_root=merged_cloud_root)
    if debug:
        print(f"[mesh-adapter] stitched tracks: {len(tracks)}")

    source_points = np.empty((0, 3), dtype=np.float64)
    source_colors = np.empty((0, 3), dtype=np.float64)
    if point_cloud_layer is not None:
        source_points, source_colors = _collect_color_source_from_point_cloud_frames(
            stage,
            point_cloud_frames_root=point_cloud_frames_root,
            points_asset_base_dir=point_cloud_layer.resolve().parent,
            max_points=max(0, int(color_source_max_points)),
        )
        if debug:
            print(
                "[mesh-adapter] source color points: "
                f"{len(source_points)} from {point_cloud_frames_root}",
                flush=True,
            )

    colored_tracks = _materialize_colored_track_clouds(
        tracks,
        source_points=source_points,
        source_colors=source_colors,
        require_color=require_color,
    )
    if debug:
        colored_count = sum(1 for track in colored_tracks if track.colors is not None)
        print(f"[mesh-adapter] colored tracks: {colored_count}", flush=True)

    destination_external = (
        external_recon_dir
        if external_recon_dir is not None
        else output_path.resolve().parent / "external" / "recon"
    )
    artifacts = reconstruct_track_mesh_assets(
        tracks=colored_tracks,
        external_recon_dir=destination_external,
        poisson_depth=poisson_depth,
        density_trim_quantile=density_trim_quantile,
        normal_radius_m=normal_radius_m,
        normal_max_nn=normal_max_nn,
        normal_orient_k=normal_orient_k,
        min_track_points=min_track_points,
    )
    if not artifacts:
        raise ValueError(
            "No mesh artifacts were produced. "
            "Tracks may contain too few points; lower --min-track-points."
        )
    if debug:
        print(f"[mesh-adapter] reconstructed meshes: {len(artifacts)}", flush=True)

    tps = float(time_codes_per_second)
    if tps <= 0.0:
        tps = float(stage.GetTimeCodesPerSecond() or 30.0)
    start_time = float(stage.GetStartTimeCode())
    end_time = float(stage.GetEndTimeCode())

    params = {
        "tracks_layer": str(tracks_layer),
        "point_cloud_layer": str(point_cloud_layer) if point_cloud_layer is not None else "",
        "merged_cloud_root": merged_cloud_root,
        "point_cloud_frames_root": point_cloud_frames_root,
        "external_recon_dir": str(destination_external),
        "require_color": bool(require_color),
        "color_source_max_points": int(color_source_max_points),
        "poisson_depth": int(poisson_depth),
        "density_trim_quantile": float(density_trim_quantile),
        "normal_radius_m": float(normal_radius_m),
        "normal_max_nn": int(normal_max_nn),
        "normal_orient_k": int(normal_orient_k),
        "min_track_points": int(min_track_points),
        "mesh_count": int(len(artifacts)),
    }
    out_stage = mesh_artifacts_to_stage(
        artifacts=artifacts,
        output_layer_path=output_path,
        run_id=run_id,
        model_name="world2data_open3d_surface_reconstruction",
        model_version="0.1",
        params=params,
        time_codes_per_second=tps,
        start_time_code=start_time,
        end_time_code=end_time,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_stage.GetRootLayer().Export(str(output_path))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read PF stitched track point clouds and generate high-fidelity colored Open3D meshes "
            "with a protocol-aligned mesh layer."
        )
    )
    parser.add_argument("--tracks-layer", type=Path, required=True, help="particle-filter tracks layer")
    parser.add_argument(
        "--point-cloud-layer",
        type=Path,
        default=None,
        help="optional point-cloud frames layer used as color source for stitched points",
    )
    parser.add_argument(
        "--output-usd",
        type=Path,
        default=Path("outputs/mesh_reconstruction.usda"),
        help="output OpenUSD mesh layer",
    )
    parser.add_argument(
        "--external-recon-dir",
        type=Path,
        default=None,
        help="directory where stitched colored point clouds and meshes are written",
    )
    parser.add_argument("--run-id", type=str, default="", help="provenance run id")
    parser.add_argument(
        "--merged-cloud-root",
        type=str,
        default="/World/W2D/Reconstruction/MergedTrackPointClouds",
        help="PF stitched cloud scope path",
    )
    parser.add_argument(
        "--point-cloud-frames-root",
        type=str,
        default="/World/W2D/Reconstruction/PointCloudFrames",
        help="point-cloud frames scope path for color-source assets",
    )
    parser.add_argument(
        "--allow-uncolored-fallback",
        action="store_true",
        help="if set, tracks without color are assigned deterministic fallback colors",
    )
    parser.add_argument(
        "--color-source-max-points",
        type=int,
        default=500_000,
        help="maximum colored points to load from source point-cloud assets for color transfer",
    )
    parser.add_argument(
        "--poisson-depth",
        type=int,
        default=11,
        help="Open3D Poisson depth (higher = more detail)",
    )
    parser.add_argument(
        "--density-trim-quantile",
        type=float,
        default=0.01,
        help="Poisson low-density vertex removal quantile in [0, 0.49]",
    )
    parser.add_argument(
        "--normal-radius-m",
        type=float,
        default=0.0,
        help="normal-estimation radius in meters; 0 means auto",
    )
    parser.add_argument(
        "--normal-max-nn",
        type=int,
        default=48,
        help="max neighbors for normal estimation",
    )
    parser.add_argument(
        "--normal-orient-k",
        type=int,
        default=24,
        help="k for orient_normals_consistent_tangent_plane",
    )
    parser.add_argument(
        "--min-track-points",
        type=int,
        default=50,
        help="minimum stitched points required to reconstruct a mesh for a track",
    )
    parser.add_argument(
        "--timecodes-per-second",
        type=float,
        default=0.0,
        help="output stage timeCodesPerSecond; 0 means infer from input stage",
    )
    parser.add_argument("--debug", action="store_true", help="print reconstruction diagnostics")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_open3d_mesh_layer(
        tracks_layer=args.tracks_layer,
        output_path=args.output_usd,
        run_id=run_id,
        point_cloud_layer=args.point_cloud_layer,
        external_recon_dir=args.external_recon_dir,
        merged_cloud_root=str(args.merged_cloud_root),
        point_cloud_frames_root=str(args.point_cloud_frames_root),
        require_color=not bool(args.allow_uncolored_fallback),
        color_source_max_points=int(args.color_source_max_points),
        poisson_depth=int(args.poisson_depth),
        density_trim_quantile=float(args.density_trim_quantile),
        normal_radius_m=float(args.normal_radius_m),
        normal_max_nn=int(args.normal_max_nn),
        normal_orient_k=int(args.normal_orient_k),
        min_track_points=int(args.min_track_points),
        time_codes_per_second=float(args.timecodes_per_second),
        debug=bool(args.debug),
    )
    print(f"wrote mesh layer: {args.output_usd}")
    print(f"run_id={run_id}")


if __name__ == "__main__":
    main()
