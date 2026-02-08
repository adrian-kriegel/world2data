from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from world2data.mesh_adapter import write_open3d_mesh_layer


def _sphere_points_with_colors(*, count: int = 900) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    xyz = rng.normal(size=(count, 3))
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    xyz = xyz / norms
    radius = 0.4 + 0.03 * rng.normal(size=(count, 1))
    points = np.asarray([0.2, -0.1, 2.0], dtype=np.float64)[None, :] + xyz * radius
    colors = np.clip((points - points.min(axis=0)) / np.maximum(np.ptp(points, axis=0), 1e-9), 0.0, 1.0)
    return points, colors


def _write_pf_tracks_layer_with_colored_stitched_cloud(path: Path) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(30.0)
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(2.0)

    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    UsdGeom.Scope.Define(stage, "/World/W2D")
    UsdGeom.Scope.Define(stage, "/World/W2D/Tracks")
    UsdGeom.Scope.Define(stage, "/World/W2D/Tracks/ParticleFilter")
    UsdGeom.Scope.Define(stage, "/World/W2D/Reconstruction")
    UsdGeom.Scope.Define(stage, "/World/W2D/Reconstruction/MergedTrackPointClouds")

    track_id = "track-1"
    safe = "track_1"
    track_prim = UsdGeom.Xform.Define(stage, f"/World/W2D/Tracks/ParticleFilter/{safe}").GetPrim()
    track_prim.CreateAttribute("w2d:trackId", Sdf.ValueTypeNames.String, custom=True).Set(track_id)
    track_prim.CreateAttribute("w2d:class", Sdf.ValueTypeNames.String, custom=True).Set("cup")

    points, colors = _sphere_points_with_colors()
    merged_prim = UsdGeom.Points.Define(
        stage,
        f"/World/W2D/Reconstruction/MergedTrackPointClouds/{safe}",
    ).GetPrim()
    merged_schema = UsdGeom.Points(merged_prim)
    merged_schema.CreatePointsAttr(
        Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points])
    )
    merged_schema.CreateDisplayColorAttr(
        Vt.Vec3fArray([Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])) for c in colors])
    )
    merged_prim.CreateAttribute("w2d:trackId", Sdf.ValueTypeNames.String, custom=True).Set(track_id)
    merged_prim.CreateAttribute("w2d:class", Sdf.ValueTypeNames.String, custom=True).Set("cup")

    stage.GetRootLayer().Save()


def test_mesh_adapter_writes_protocol_aligned_layer_and_assets(tmp_path: Path) -> None:
    pytest.importorskip("pxr")
    open3d = pytest.importorskip("open3d")
    from pxr import Usd

    tracks_layer = tmp_path / "particle_tracks.usda"
    _write_pf_tracks_layer_with_colored_stitched_cloud(tracks_layer)

    output_layer = tmp_path / "mesh_layer.usda"
    external_recon_dir = tmp_path / "external" / "recon"
    write_open3d_mesh_layer(
        tracks_layer=tracks_layer,
        output_path=output_layer,
        run_id="mesh-run-1",
        external_recon_dir=external_recon_dir,
        poisson_depth=8,
        min_track_points=40,
        require_color=True,
    )

    stage = Usd.Stage.Open(str(output_layer))
    assert stage

    mesh_prim = stage.GetPrimAtPath("/World/W2D/Reconstruction/Meshes/track_1")
    assert mesh_prim
    assert mesh_prim.GetAttribute("w2d:meshHasVertexColors").Get() is True
    assert mesh_prim.GetAttribute("w2d:sourcePointsHaveColor").Get() is True
    assert int(mesh_prim.GetAttribute("w2d:vertexCount").Get() or 0) > 0
    assert int(mesh_prim.GetAttribute("w2d:faceCount").Get() or 0) > 0
    assert mesh_prim.GetRelationship("w2d:sourceStitchedPointCloud").GetTargets()

    stitched_prim = stage.GetPrimAtPath("/World/W2D/Reconstruction/StitchedTrackPointClouds/track_1")
    assert stitched_prim
    assert stitched_prim.GetAttribute("w2d:hasColor").Get() is True
    assert int(stitched_prim.GetAttribute("w2d:pointCount").Get() or 0) > 0

    mesh_asset = mesh_prim.GetAttribute("w2d:meshAsset").Get()
    points_asset = stitched_prim.GetAttribute("w2d:pointsAsset").Get()
    assert mesh_asset is not None
    assert points_asset is not None

    mesh_asset_path = (output_layer.parent / str(mesh_asset.path)).resolve()
    points_asset_path = (output_layer.parent / str(points_asset.path)).resolve()
    assert mesh_asset_path.exists()
    assert points_asset_path.exists()

    mesh = open3d.io.read_triangle_mesh(str(mesh_asset_path))
    point_cloud = open3d.io.read_point_cloud(str(points_asset_path))
    assert len(np.asarray(mesh.vertices)) > 0
    assert len(np.asarray(mesh.vertex_colors)) > 0
    assert len(np.asarray(point_cloud.points)) > 0
    assert len(np.asarray(point_cloud.colors)) > 0
