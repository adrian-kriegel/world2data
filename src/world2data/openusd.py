from __future__ import annotations

"""OpenUSD export helpers for particle-filter estimates."""

import re
from collections.abc import Mapping

from .model import TrackEstimate

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_]")


def _require_pxr() -> tuple[object, object, object]:
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "OpenUSD Python bindings are required. Add `openusd` to your Python environment."
        ) from exc
    return Gf, Sdf, Usd, UsdGeom


def _safe_name(raw: str) -> str:
    value = _SAFE_NAME_RE.sub("_", raw).strip("_")
    if not value:
        value = "item"
    if value[0].isdigit():
        value = f"n_{value}"
    return value


def track_estimates_to_stage(
    estimates: Mapping[str, TrackEstimate],
    *,
    frame_index: int | None = None,
) -> object:
    """Build an in-memory OpenUSD stage containing per-track centroid + mean box state."""
    Gf, Sdf, Usd, UsdGeom = _require_pxr()

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    UsdGeom.Xform.Define(stage, "/World")
    centroids_scope = UsdGeom.Scope.Define(stage, "/World/ParticleCentroids").GetPrim()
    rel = centroids_scope.CreateRelationship("world2data:itemCentroids", custom=False)

    if frame_index is not None:
        stage.SetStartTimeCode(float(frame_index))
        stage.SetEndTimeCode(float(frame_index))
        centroids_scope.CreateAttribute("world2data:frameIndex", Sdf.ValueTypeNames.Int).Set(
            frame_index
        )

    for track_id in sorted(estimates.keys()):
        estimate = estimates[track_id]
        prim_path = f"/World/ParticleCentroids/{_safe_name(track_id)}"
        prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*estimate.position))

        prim.CreateAttribute("world2data:trackId", Sdf.ValueTypeNames.String).Set(estimate.track_id)
        prim.CreateAttribute("world2data:label", Sdf.ValueTypeNames.String).Set(estimate.label)
        prim.CreateAttribute("world2data:boundingBox", Sdf.ValueTypeNames.Float3).Set(
            Gf.Vec3f(*estimate.bounding_box)
        )
        prim.CreateAttribute("world2data:mass", Sdf.ValueTypeNames.Float).Set(estimate.mass)
        prim.CreateAttribute("world2data:velocity", Sdf.ValueTypeNames.Float3).Set(
            Gf.Vec3f(*estimate.velocity)
        )
        prim.CreateAttribute("world2data:particleCount", Sdf.ValueTypeNames.Int).Set(
            estimate.particle_count
        )
        rel.AddTarget(prim.GetPath())

    return stage


def track_estimates_to_usda(
    estimates: Mapping[str, TrackEstimate],
    *,
    frame_index: int | None = None,
) -> str:
    stage = track_estimates_to_stage(estimates, frame_index=frame_index)
    return stage.GetRootLayer().ExportToString()
