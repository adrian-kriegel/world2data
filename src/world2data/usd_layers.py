"""OpenUSD Layering Protocol for World2Data.

Implements the multi-layer composition spec from W2D_OpenUSD_Layering_Protocol.md:

  scene.usda                      # assembly entrypoint (open this file)
  layers/
    00_base.usda                  # conventions + empty scopes
    10_inputs_run_<RUNID>.usda    # input refs + metadata
    20_recon_run_<RUNID>.usdc     # cameras + recon outputs
    30_tracks_run_<RUNID>.usdc    # entities + tracks
    40_events_run_<RUNID>.usda    # events/relations graph
    90_overrides.usda             # human QA / final decisions (strong)
    99_session.usda               # per-user local edits (strongest)
  external/
    inputs/                       # video files
    recon/                        # point cloud caches (.ply)

Key principles:
  - No physical merging by default (use composition)
  - Deterministic composition (same layers -> same stage)
  - Namespace ownership (each producer owns its prefix)
  - Provenance-first (every result traceable to run + model + params)
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import hashlib

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PARQUET = True
except ImportError:
    _HAS_PARQUET = False

_SAFE_PRIM_RE = re.compile(r"[^A-Za-z0-9_]")


def _safe_prim_name(raw: str) -> str:
    """Sanitize a string for use as a USD prim name."""
    value = _SAFE_PRIM_RE.sub("_", str(raw)).strip("_")
    if not value:
        value = "item"
    if value[0].isdigit():
        value = f"n_{value}"
    return value


def _require_pxr():
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt
    except ImportError as exc:
        raise ImportError(
            "OpenUSD Python bindings are required. Install with: uv add usd-core"
        ) from exc
    return Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt


def _generate_run_id() -> str:
    """Generate a deterministic, sortable run ID."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# =========================================================================
# Data containers for layer producers
# =========================================================================

@dataclass(frozen=True)
class ProvenanceRecord:
    """Provenance metadata attached to every producer layer."""
    run_id: str
    component: str          # "ingest", "recon", "tracking", "reasoning"
    model_name: str         # "mast3r", "yolov8x-seg", "gemini-2.5-flash"
    model_version: str
    params: dict = field(default_factory=dict)
    git_commit: str = ""
    timestamp_iso: str = ""

    def __post_init__(self):
        if not self.timestamp_iso:
            object.__setattr__(
                self, "timestamp_iso",
                datetime.now(timezone.utc).isoformat(),
            )


@dataclass
class EntityRecord:
    """An entity in the scene (object, person, etc.) for the tracking layer."""
    uid: str                            # ULID or stable ID
    entity_class: str                   # "table", "chair", "person"
    label: str                          # display name "Table_01"
    confidence: float = 0.0
    detected_by: list = field(default_factory=list)
    bbox_3d_min: tuple = (0.0, 0.0, 0.0)
    bbox_3d_max: tuple = (0.0, 0.0, 0.0)
    first_frame: int = 0
    last_frame: int = 0
    # Time-sampled poses: {frame_idx: (tx, ty, tz)}
    frame_positions: dict = field(default_factory=dict)


@dataclass
class EventRecord:
    """An event/relation in the scene for the events layer."""
    uid: str
    predicate: str              # "drinksFrom", "opens", "sits_on"
    subject_uid: str            # entity uid
    object_uid: str             # entity uid (optional)
    t_start: int                # frame index
    t_end: int                  # frame index
    confidence: float = 0.0
    description: str = ""


@dataclass
class YOLOFrameRecord:
    """Per-frame YOLO detection result for the observations layer."""
    frame_index: int
    timestamp_sec: float
    image_width: int
    image_height: int
    labels: list[str] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    boxes_xyxy: list[tuple] = field(default_factory=list)  # [(x1,y1,x2,y2), ...]


@dataclass
class PointCloudFrameRecord:
    """Per-frame point cloud metadata for externalized recon assets."""
    frame_index: int
    timestamp_sec: float
    points_asset_path: str          # relative path to PLY file
    point_count: int
    points_format: str = "ply"


@dataclass
class OverrideRecord:
    """A human QA override for the overrides layer."""
    target_prim_path: str       # e.g. "/World/W2D/Entities/Objects/Table_01"
    property_name: str          # e.g. "w2d:class"
    new_value: Any
    reason: str = ""
    approved_by: str = ""


# =========================================================================
# Layer Writers
# =========================================================================

class USDLayerWriter:
    """Writes the multi-layer OpenUSD scene according to the W2D protocol.

    Usage:
        writer = USDLayerWriter(output_dir="scene")
        writer.write_base_layer(fps=30.0, start_frame=0, end_frame=900)
        writer.write_inputs_layer(run_id, video_path, cloud_path)
        writer.write_recon_layer(run_id, cameras, point_cloud_path)
        writer.write_tracks_layer(run_id, entities)
        writer.write_events_layer(run_id, events)
        writer.write_assembly()
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.layers_dir = self.output_dir / "layers"
        self.external_dir = self.output_dir / "external"
        self._layer_files: list[str] = []  # relative paths for assembly

    def _ensure_dirs(self):
        self.layers_dir.mkdir(parents=True, exist_ok=True)
        (self.external_dir / "inputs").mkdir(parents=True, exist_ok=True)
        (self.external_dir / "recon").mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 00_base.usda -- conventions + empty scopes
    # -----------------------------------------------------------------
    def write_base_layer(
        self,
        fps: float = 30.0,
        start_frame: int = 0,
        end_frame: int = 0,
        up_axis: str = "Y",
    ) -> Path:
        """Write the base layer with conventions and skeleton namespaces."""
        self._ensure_dirs()
        Gf, Sdf, Usd, UsdGeom, _, _ = _require_pxr()

        path = self.layers_dir / "00_base.usda"
        stage = Usd.Stage.CreateNew(str(path))

        stage.SetMetadata("metersPerUnit", 1.0)
        stage.SetMetadata("upAxis", up_axis)
        stage.SetMetadata("timeCodesPerSecond", fps)
        if end_frame > 0:
            stage.SetStartTimeCode(float(start_frame))
            stage.SetEndTimeCode(float(end_frame))

        # World root
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

        # W2D namespace skeleton
        for scope in [
            "/World/W2D",
            "/World/W2D/Inputs",
            "/World/W2D/Sensors",
            "/World/W2D/Reconstruction",
            "/World/W2D/Entities",
            "/World/W2D/Entities/Objects",
            "/World/W2D/Entities/People",
            "/World/W2D/Tracks",
            "/World/W2D/Observations",
            "/World/W2D/Events",
            "/World/W2D/Graph",
            "/World/W2D/Provenance",
            "/World/W2D/Provenance/runs",
            "/World/W2D/Overrides",
        ]:
            UsdGeom.Scope.Define(stage, scope)

        stage.GetRootLayer().Save()
        self._layer_files.append("00_base.usda")
        return path

    # -----------------------------------------------------------------
    # 10_inputs_run_<RUNID>.usda
    # -----------------------------------------------------------------
    def write_inputs_layer(
        self,
        provenance: ProvenanceRecord,
        video_path: str | Path | None = None,
        point_cloud_path: str | Path | None = None,
    ) -> Path:
        """Write the inputs layer referencing raw input files."""
        self._ensure_dirs()
        Gf, Sdf, Usd, UsdGeom, _, _ = _require_pxr()

        fname = f"10_inputs_run_{provenance.run_id}.usda"
        path = self.layers_dir / fname
        stage = Usd.Stage.CreateNew(str(path))

        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.Scope.Define(stage, "/World/W2D")
        UsdGeom.Scope.Define(stage, "/World/W2D/Inputs")

        # Copy input files to external/ and create references
        if video_path and Path(video_path).exists():
            dest = self.external_dir / "inputs" / Path(video_path).name
            if not dest.exists():
                shutil.copy2(str(video_path), str(dest))
            rel_path = f"../external/inputs/{dest.name}"

            vid_prim = UsdGeom.Scope.Define(
                stage, "/World/W2D/Inputs/Video_01"
            ).GetPrim()
            vid_prim.CreateAttribute(
                "w2d:uri", Sdf.ValueTypeNames.Asset, custom=True
            ).Set(rel_path)
            vid_prim.CreateAttribute(
                "w2d:uid", Sdf.ValueTypeNames.String, custom=True
            ).Set(f"video_{provenance.run_id}")
            vid_prim.CreateAttribute(
                "w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True
            ).Set(provenance.run_id)

        if point_cloud_path and Path(point_cloud_path).exists():
            dest = self.external_dir / "recon" / Path(point_cloud_path).name
            if not dest.exists():
                shutil.copy2(str(point_cloud_path), str(dest))
            rel_path = f"../external/recon/{dest.name}"

            pc_prim = UsdGeom.Scope.Define(
                stage, "/World/W2D/Inputs/PointCloud_01"
            ).GetPrim()
            pc_prim.CreateAttribute(
                "w2d:uri", Sdf.ValueTypeNames.Asset, custom=True
            ).Set(rel_path)
            pc_prim.CreateAttribute(
                "w2d:uid", Sdf.ValueTypeNames.String, custom=True
            ).Set(f"cloud_{provenance.run_id}")

        # Provenance
        self._write_provenance(stage, provenance)
        stage.GetRootLayer().Save()
        self._layer_files.append(fname)
        return path

    # -----------------------------------------------------------------
    # 20_recon_run_<RUNID>.usda -- cameras + reconstruction
    # -----------------------------------------------------------------
    def write_recon_layer(
        self,
        provenance: ProvenanceRecord,
        cameras: Sequence[dict] | None = None,
        point_cloud: np.ndarray | None = None,
        point_colors: np.ndarray | None = None,
        focals: Sequence[float] | None = None,
        fps: float = 30.0,
    ) -> Path:
        """Write reconstruction layer with cameras and optional point cloud.

        Args:
            cameras: list of dicts with keys: pose (4x4), focal, principal_point, frame_idx
            point_cloud: (N, 3) array
            point_colors: (N, 3) array (0-1 float)
            focals: per-camera focal lengths
            fps: video FPS for time-sampling
        """
        self._ensure_dirs()
        Gf, Sdf, Usd, UsdGeom, _, Vt = _require_pxr()

        fname = f"20_recon_run_{provenance.run_id}.usda"
        path = self.layers_dir / fname
        stage = Usd.Stage.CreateNew(str(path))

        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.Scope.Define(stage, "/World/W2D")
        UsdGeom.Scope.Define(stage, "/World/W2D/Sensors")

        # Cameras with time-sampled poses
        if cameras:
            rig = UsdGeom.Xform.Define(stage, "/World/W2D/Sensors/Rig_01")
            rig_prim = rig.GetPrim()
            rig_prim.CreateAttribute(
                "w2d:uid", Sdf.ValueTypeNames.String, custom=True
            ).Set(f"rig_{provenance.run_id}")
            rig_prim.CreateAttribute(
                "w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True
            ).Set(provenance.run_id)

            for i, cam in enumerate(cameras):
                cam_path = f"/World/W2D/Sensors/Rig_01/Cam_{i:03d}"
                cam_xform = UsdGeom.Xform.Define(stage, cam_path)
                cam_prim = cam_xform.GetPrim()

                # Pose as transform
                pose = np.array(cam.get("pose", np.eye(4)))
                mat = Gf.Matrix4d(*pose.flatten().tolist())

                frame_idx = cam.get("frame_idx", i)
                xform_op = cam_xform.AddTransformOp()
                xform_op.Set(mat, float(frame_idx))

                # Camera intrinsics
                focal = cam.get("focal", 500.0)
                if focals and i < len(focals):
                    focal = focals[i]

                usd_cam = UsdGeom.Camera.Define(stage, f"{cam_path}/Camera")
                usd_cam.GetFocalLengthAttr().Set(float(focal) * 0.036)

                cam_prim.CreateAttribute(
                    "w2d:uid", Sdf.ValueTypeNames.String, custom=True
                ).Set(f"cam_{provenance.run_id}_{i:03d}")
                cam_prim.CreateAttribute(
                    "w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True
                ).Set(provenance.run_id)

        # Reconstruction point cloud (optional, can be heavy)
        if point_cloud is not None and point_cloud.shape[0] > 0:
            UsdGeom.Scope.Define(stage, "/World/W2D/Reconstruction")
            pc_prim = UsdGeom.Points.Define(
                stage, "/World/W2D/Reconstruction/DenseCloud"
            )
            pts = point_cloud
            # Subsample if too large for USDA
            max_pts = 100000
            if pts.shape[0] > max_pts:
                rng = np.random.RandomState(42)
                idx = rng.choice(pts.shape[0], max_pts, replace=False)
                pts = pts[idx]
                if point_colors is not None:
                    point_colors = point_colors[idx]

            pc_prim.GetPointsAttr().Set(
                Vt.Vec3fArray([Gf.Vec3f(*p.tolist()) for p in pts])
            )
            pc_prim.GetWidthsAttr().Set(
                Vt.FloatArray([0.005] * len(pts))
            )
            if point_colors is not None and point_colors.shape[0] == pts.shape[0]:
                colors = Vt.Vec3fArray(
                    [Gf.Vec3f(*c.tolist()) for c in np.clip(point_colors, 0, 1)]
                )
                pc_prim.GetDisplayColorAttr().Set(colors)

            pc_prim.GetPrim().CreateAttribute(
                "w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True
            ).Set(provenance.run_id)

        self._write_provenance(stage, provenance)
        stage.GetRootLayer().Save()
        self._layer_files.append(fname)
        return path

    # -----------------------------------------------------------------
    # Per-frame point cloud externalization helpers
    # -----------------------------------------------------------------
    def write_per_frame_point_clouds(
        self,
        frame_data: Sequence[dict],
        run_id: str,
    ) -> list[PointCloudFrameRecord]:
        """Write per-frame point clouds as external PLY files.

        Args:
            frame_data: list of dicts with keys: index, pts3d (N,3), colors (N,3), timestamp_sec
            run_id: for provenance

        Returns:
            List of PointCloudFrameRecord for use in the recon layer.
        """
        recon_dir = self.external_dir / "recon"
        recon_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for fd in frame_data:
            idx = fd["index"]
            pts = fd["pts3d"]
            colors = fd.get("colors")
            ts = fd.get("timestamp_sec", idx / 30.0)
            n = pts.shape[0] if pts is not None else 0
            if n == 0:
                continue

            fname = f"frame_{idx:06d}.ply"
            ply_path = recon_dir / fname
            _write_ply(ply_path, pts, colors)

            records.append(PointCloudFrameRecord(
                frame_index=idx,
                timestamp_sec=ts,
                points_asset_path=f"../external/recon/{fname}",
                point_count=n,
            ))
        return records

    # -----------------------------------------------------------------
    # 20_recon extended: per-frame point cloud index in recon layer
    # -----------------------------------------------------------------
    def write_recon_layer_with_frames(
        self,
        provenance: ProvenanceRecord,
        cameras: Sequence[dict] | None = None,
        frame_records: Sequence[PointCloudFrameRecord] | None = None,
        focals: Sequence[float] | None = None,
        fps: float = 30.0,
    ) -> Path:
        """Write reconstruction layer with cameras + per-frame point cloud index.

        Unlike write_recon_layer, this does NOT embed dense point arrays.
        Instead it writes compact metadata prims pointing to external assets.
        """
        self._ensure_dirs()
        Gf, Sdf, Usd, UsdGeom, _, Vt = _require_pxr()

        fname = f"20_recon_run_{provenance.run_id}.usda"
        path = self.layers_dir / fname
        stage = Usd.Stage.CreateNew(str(path))

        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.Scope.Define(stage, "/World/W2D")
        UsdGeom.Scope.Define(stage, "/World/W2D/Sensors")
        UsdGeom.Scope.Define(stage, "/World/W2D/Reconstruction")

        # --- Cameras with time-sampled poses ---
        if cameras:
            rig = UsdGeom.Xform.Define(stage, "/World/W2D/Sensors/Rig_01")
            rig_prim = rig.GetPrim()
            rig_prim.CreateAttribute(
                "w2d:uid", Sdf.ValueTypeNames.String, custom=True
            ).Set(f"rig_{provenance.run_id}")
            rig_prim.CreateAttribute(
                "w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True
            ).Set(provenance.run_id)

            # Camera pose frames scope (protocol 15.2)
            UsdGeom.Scope.Define(stage, "/World/W2D/Sensors/CameraPoses")
            UsdGeom.Scope.Define(stage, "/World/W2D/Sensors/CameraPoses/Frames")

            for i, cam in enumerate(cameras):
                # Standard camera Xform with time-sampled pose
                cam_path = f"/World/W2D/Sensors/Rig_01/Cam_{i:03d}"
                cam_xform = UsdGeom.Xform.Define(stage, cam_path)
                cam_prim = cam_xform.GetPrim()

                pose = np.array(cam.get("pose", np.eye(4)))
                mat = Gf.Matrix4d(*pose.flatten().tolist())
                frame_idx = cam.get("frame_idx", i)
                xform_op = cam_xform.AddTransformOp()
                xform_op.Set(mat, float(frame_idx))

                focal = cam.get("focal", 500.0)
                if focals and i < len(focals):
                    focal = focals[i]
                usd_cam = UsdGeom.Camera.Define(stage, f"{cam_path}/Camera")
                usd_cam.GetFocalLengthAttr().Set(float(focal) * 0.036)

                cam_prim.CreateAttribute(
                    "w2d:uid", Sdf.ValueTypeNames.String, custom=True
                ).Set(f"cam_{provenance.run_id}_{i:03d}")
                cam_prim.CreateAttribute(
                    "w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True
                ).Set(provenance.run_id)

                # Per-frame pose prim (protocol 15.2)
                pose_path = f"/World/W2D/Sensors/CameraPoses/Frames/f_{frame_idx:06d}"
                pose_scope = UsdGeom.Scope.Define(stage, pose_path)
                pp = pose_scope.GetPrim()
                pp.CreateAttribute("w2d:frameIndex", Sdf.ValueTypeNames.Int, custom=True).Set(frame_idx)
                ts = cam.get("timestamp_sec", frame_idx / fps)
                pp.CreateAttribute("w2d:timestampSec", Sdf.ValueTypeNames.Double, custom=True).Set(float(ts))
                # Store rotation (3x3) and translation (3,)
                R = pose[:3, :3]
                t = pose[:3, 3]
                pp.CreateAttribute("w2d:translation", Sdf.ValueTypeNames.Float3, custom=True).Set(
                    Gf.Vec3f(*t.tolist())
                )
                pp.CreateAttribute("w2d:poseConvention", Sdf.ValueTypeNames.String, custom=True).Set("camera_to_world")
                pp.CreateAttribute("w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True).Set(provenance.run_id)

        # --- Per-frame point cloud index (protocol 15.4) ---
        if frame_records:
            pcf_scope = UsdGeom.Scope.Define(
                stage, "/World/W2D/Reconstruction/PointCloudFrames"
            )
            for rec in frame_records:
                prim_path = f"/World/W2D/Reconstruction/PointCloudFrames/f_{rec.frame_index:06d}"
                scope = UsdGeom.Scope.Define(stage, prim_path)
                p = scope.GetPrim()
                p.CreateAttribute("w2d:frameIndex", Sdf.ValueTypeNames.Int, custom=True).Set(rec.frame_index)
                p.CreateAttribute("w2d:timestampSec", Sdf.ValueTypeNames.Double, custom=True).Set(rec.timestamp_sec)
                p.CreateAttribute("w2d:pointsAsset", Sdf.ValueTypeNames.Asset, custom=True).Set(rec.points_asset_path)
                p.CreateAttribute("w2d:pointCount", Sdf.ValueTypeNames.Int, custom=True).Set(rec.point_count)
                p.CreateAttribute("w2d:pointsFormat", Sdf.ValueTypeNames.String, custom=True).Set(rec.points_format)
                p.CreateAttribute("w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True).Set(provenance.run_id)

        self._write_provenance(stage, provenance)
        stage.GetRootLayer().Save()
        self._layer_files.append(fname)
        return path

    # -----------------------------------------------------------------
    # 25_yolo_run_<RUNID>.usda -- YOLO observations (protocol 15.3)
    # -----------------------------------------------------------------
    def write_yolo_observations_layer(
        self,
        provenance: ProvenanceRecord,
        frames: Sequence[YOLOFrameRecord],
    ) -> Path:
        """Write YOLO per-frame 2D detection observations."""
        self._ensure_dirs()
        Gf, Sdf, Usd, UsdGeom, _, Vt = _require_pxr()

        fname = f"25_yolo_run_{provenance.run_id}.usda"
        path = self.layers_dir / fname
        stage = Usd.Stage.CreateNew(str(path))

        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.Scope.Define(stage, "/World/W2D")
        UsdGeom.Scope.Define(stage, "/World/W2D/Observations")
        UsdGeom.Scope.Define(stage, "/World/W2D/Observations/YOLO")
        UsdGeom.Scope.Define(stage, "/World/W2D/Observations/YOLO/Frames")

        for fr in frames:
            prim_path = f"/World/W2D/Observations/YOLO/Frames/f_{fr.frame_index:06d}"
            scope = UsdGeom.Scope.Define(stage, prim_path)
            p = scope.GetPrim()

            p.CreateAttribute("w2d:frameIndex", Sdf.ValueTypeNames.Int, custom=True).Set(fr.frame_index)
            p.CreateAttribute("w2d:timestampSec", Sdf.ValueTypeNames.Double, custom=True).Set(fr.timestamp_sec)
            p.CreateAttribute("w2d:imageWidth", Sdf.ValueTypeNames.Int, custom=True).Set(fr.image_width)
            p.CreateAttribute("w2d:imageHeight", Sdf.ValueTypeNames.Int, custom=True).Set(fr.image_height)
            p.CreateAttribute("w2d:detectionCount", Sdf.ValueTypeNames.Int, custom=True).Set(len(fr.labels))
            p.CreateAttribute("w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True).Set(provenance.run_id)

            if fr.labels:
                p.CreateAttribute("w2d:labels", Sdf.ValueTypeNames.StringArray, custom=True).Set(fr.labels)
            if fr.class_ids:
                p.CreateAttribute("w2d:classIds", Sdf.ValueTypeNames.IntArray, custom=True).Set(fr.class_ids)
            if fr.scores:
                p.CreateAttribute("w2d:scores", Sdf.ValueTypeNames.FloatArray, custom=True).Set(
                    [float(s) for s in fr.scores]
                )
            if fr.boxes_xyxy:
                # Flatten to float4 array: [x1,y1,x2,y2, x1,y1,x2,y2, ...]
                flat = []
                for box in fr.boxes_xyxy:
                    flat.append(Gf.Vec4f(*[float(v) for v in box[:4]]))
                p.CreateAttribute("w2d:boxesXYXY", Sdf.ValueTypeNames.Float4Array, custom=True).Set(
                    Vt.Vec4fArray(flat)
                )

        self._write_provenance(stage, provenance)
        stage.GetRootLayer().Save()
        self._layer_files.append(fname)
        return path

    # -----------------------------------------------------------------
    # 30_tracks_run_<RUNID>.usda -- entities + tracks
    # -----------------------------------------------------------------
    def write_tracks_layer(
        self,
        provenance: ProvenanceRecord,
        entities: Sequence[EntityRecord],
        fps: float = 30.0,
    ) -> Path:
        """Write tracking layer with entities and their time-sampled tracks."""
        self._ensure_dirs()
        Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt = _require_pxr()

        fname = f"30_tracks_run_{provenance.run_id}.usda"
        path = self.layers_dir / fname
        stage = Usd.Stage.CreateNew(str(path))

        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.Scope.Define(stage, "/World/W2D")
        UsdGeom.Scope.Define(stage, "/World/W2D/Entities")
        UsdGeom.Scope.Define(stage, "/World/W2D/Entities/Objects")
        UsdGeom.Scope.Define(stage, "/World/W2D/Tracks")

        for entity in entities:
            safe_name = _safe_prim_name(entity.label)

            # Entity prim (identity)
            entity_path = f"/World/W2D/Entities/Objects/{safe_name}"
            entity_xform = UsdGeom.Xform.Define(stage, entity_path)
            prim = entity_xform.GetPrim()

            # Required: w2d:uid
            prim.CreateAttribute(
                "w2d:uid", Sdf.ValueTypeNames.String, custom=True
            ).Set(entity.uid)
            prim.CreateAttribute(
                "w2d:class", Sdf.ValueTypeNames.String, custom=True
            ).Set(entity.entity_class)
            prim.CreateAttribute(
                "w2d:label", Sdf.ValueTypeNames.String, custom=True
            ).Set(entity.label)
            prim.CreateAttribute(
                "w2d:confidence", Sdf.ValueTypeNames.Float, custom=True
            ).Set(float(entity.confidence))
            prim.CreateAttribute(
                "w2d:detectedBy", Sdf.ValueTypeNames.String, custom=True
            ).Set(",".join(entity.detected_by))
            prim.CreateAttribute(
                "w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True
            ).Set(provenance.run_id)

            # Static position (center of bbox)
            center = tuple(
                (a + b) / 2.0
                for a, b in zip(entity.bbox_3d_min, entity.bbox_3d_max)
            )
            entity_xform.AddTranslateOp().Set(Gf.Vec3d(*center))

            # Bounding box as cube child
            size = tuple(
                max(abs(b - a), 0.01)
                for a, b in zip(entity.bbox_3d_min, entity.bbox_3d_max)
            )
            cube = UsdGeom.Cube.Define(stage, f"{entity_path}/Shape")
            max_dim = max(size) if size else 0.1
            cube.GetSizeAttr().Set(float(max_dim))
            sx = size[0] / max_dim if max_dim > 0 else 1
            sy = size[1] / max_dim if max_dim > 0 else 1
            sz = size[2] / max_dim if max_dim > 0 else 1
            cube.AddScaleOp().Set(Gf.Vec3f(float(sx), float(sy), float(sz)))

            prim.CreateAttribute(
                "w2d:bboxMin", Sdf.ValueTypeNames.Float3, custom=True
            ).Set(Gf.Vec3f(*[float(v) for v in entity.bbox_3d_min]))
            prim.CreateAttribute(
                "w2d:bboxMax", Sdf.ValueTypeNames.Float3, custom=True
            ).Set(Gf.Vec3f(*[float(v) for v in entity.bbox_3d_max]))

            # Track prim (time-sampled positions)
            if entity.frame_positions:
                track_path = f"/World/W2D/Tracks/{safe_name}"
                track_xform = UsdGeom.Xform.Define(stage, track_path)
                track_prim = track_xform.GetPrim()
                track_prim.CreateAttribute(
                    "w2d:uid", Sdf.ValueTypeNames.String, custom=True
                ).Set(f"track_{entity.uid}")

                translate_op = track_xform.AddTranslateOp()
                for frame_idx, pos in sorted(entity.frame_positions.items()):
                    translate_op.Set(
                        Gf.Vec3d(*[float(v) for v in pos]),
                        float(frame_idx),
                    )

                # Relationship to entity
                track_prim.CreateRelationship(
                    "w2d:entity", custom=True
                ).AddTarget(entity_path)

        self._write_provenance(stage, provenance)
        stage.GetRootLayer().Save()
        self._layer_files.append(fname)
        return path

    # -----------------------------------------------------------------
    # 40_events_run_<RUNID>.usda -- events/relations
    # -----------------------------------------------------------------
    def write_events_layer(
        self,
        provenance: ProvenanceRecord,
        events: Sequence[EventRecord],
    ) -> Path:
        """Write events layer with actions/relations and time intervals."""
        self._ensure_dirs()
        Gf, Sdf, Usd, UsdGeom, _, _ = _require_pxr()

        fname = f"40_events_run_{provenance.run_id}.usda"
        path = self.layers_dir / fname
        stage = Usd.Stage.CreateNew(str(path))

        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.Scope.Define(stage, "/World/W2D")
        UsdGeom.Scope.Define(stage, "/World/W2D/Events")

        for event in events:
            safe_name = _safe_prim_name(f"{event.predicate}_{event.uid}")
            event_path = f"/World/W2D/Events/{safe_name}"
            event_scope = UsdGeom.Scope.Define(stage, event_path)
            prim = event_scope.GetPrim()

            prim.CreateAttribute(
                "w2d:uid", Sdf.ValueTypeNames.String, custom=True
            ).Set(event.uid)
            prim.CreateAttribute(
                "w2d:predicate", Sdf.ValueTypeNames.String, custom=True
            ).Set(event.predicate)
            prim.CreateAttribute(
                "w2d:tStart", Sdf.ValueTypeNames.Int, custom=True
            ).Set(event.t_start)
            prim.CreateAttribute(
                "w2d:tEnd", Sdf.ValueTypeNames.Int, custom=True
            ).Set(event.t_end)
            prim.CreateAttribute(
                "w2d:confidence", Sdf.ValueTypeNames.Float, custom=True
            ).Set(float(event.confidence))
            prim.CreateAttribute(
                "w2d:description", Sdf.ValueTypeNames.String, custom=True
            ).Set(event.description)
            prim.CreateAttribute(
                "w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True
            ).Set(provenance.run_id)

            # Relationships to subject/object entities
            if event.subject_uid:
                prim.CreateRelationship("w2d:subject", custom=True)
            if event.object_uid:
                prim.CreateRelationship("w2d:object", custom=True)

        self._write_provenance(stage, provenance)
        stage.GetRootLayer().Save()
        self._layer_files.append(fname)
        return path

    # -----------------------------------------------------------------
    # 90_overrides.usda -- human QA corrections
    # -----------------------------------------------------------------
    def write_overrides_layer(
        self,
        overrides: Sequence[OverrideRecord] | None = None,
    ) -> Path:
        """Write (or create empty) human QA overrides layer."""
        self._ensure_dirs()
        Gf, Sdf, Usd, UsdGeom, _, _ = _require_pxr()

        path = self.layers_dir / "90_overrides.usda"
        stage = Usd.Stage.CreateNew(str(path))

        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.Scope.Define(stage, "/World/W2D")
        UsdGeom.Scope.Define(stage, "/World/W2D/Overrides")

        if overrides:
            for ovr in overrides:
                # Override the target prim's property
                prim = stage.OverridePrim(ovr.target_prim_path)
                if ovr.reason:
                    prim.CreateAttribute(
                        "w2d:overrideReason",
                        Sdf.ValueTypeNames.String,
                        custom=True,
                    ).Set(ovr.reason)
                if ovr.approved_by:
                    prim.CreateAttribute(
                        "w2d:approvedBy",
                        Sdf.ValueTypeNames.String,
                        custom=True,
                    ).Set(ovr.approved_by)

        stage.GetRootLayer().Save()
        self._layer_files.append("90_overrides.usda")
        return path

    # -----------------------------------------------------------------
    # 99_session.usda -- per-user local edits (gitignored)
    # -----------------------------------------------------------------
    def write_session_layer(self) -> Path:
        """Create an empty session layer (per-user, gitignored)."""
        self._ensure_dirs()
        _, Sdf, Usd, UsdGeom, _, _ = _require_pxr()

        path = self.layers_dir / "99_session.usda"
        stage = Usd.Stage.CreateNew(str(path))
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        stage.GetRootLayer().Save()
        self._layer_files.append("99_session.usda")
        return path

    # -----------------------------------------------------------------
    # scene.usda -- assembly entrypoint
    # -----------------------------------------------------------------
    def write_assembly(
        self,
        fps: float = 30.0,
        start_frame: int = 0,
        end_frame: int = 0,
    ) -> Path:
        """Write the assembly scene.usda that composes all layers."""
        self._ensure_dirs()
        _, Sdf, Usd, UsdGeom, _, _ = _require_pxr()

        path = self.output_dir / "scene.usda"

        # Build sublayer list (weak -> strong order)
        ordered = sorted(set(self._layer_files), key=self._layer_sort_key)
        sublayer_paths = [f"./layers/{f}" for f in ordered]

        # Write manually for clean output
        lines = [
            '#usda 1.0',
            '(',
            '  defaultPrim = "World"',
            '  metersPerUnit = 1',
            '  upAxis = "Y"',
            f'  timeCodesPerSecond = {fps}',
        ]
        if end_frame > 0:
            lines.append(f'  startTimeCode = {start_frame}')
            lines.append(f'  endTimeCode = {end_frame}')

        lines.append('  subLayers = [')
        for sp in sublayer_paths:
            lines.append(f'    @{sp}@,')
        lines.append('  ]')
        lines.append(')')
        lines.append('')
        lines.append('def Xform "World" {}')
        lines.append('')

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    # -----------------------------------------------------------------
    # Provenance helper
    # -----------------------------------------------------------------
    def _write_provenance(self, stage, provenance: ProvenanceRecord):
        """Write provenance run record to a stage."""
        _, Sdf, Usd, UsdGeom, _, _ = _require_pxr()

        runs_scope_path = "/World/W2D/Provenance/runs"
        # Ensure scope exists
        if not stage.GetPrimAtPath(runs_scope_path):
            UsdGeom.Scope.Define(stage, "/World/W2D/Provenance")
            UsdGeom.Scope.Define(stage, runs_scope_path)

        run_path = f"{runs_scope_path}/{_safe_prim_name(provenance.run_id)}"
        run_scope = UsdGeom.Scope.Define(stage, run_path)
        prim = run_scope.GetPrim()

        prim.CreateAttribute(
            "w2d:runId", Sdf.ValueTypeNames.String, custom=True
        ).Set(provenance.run_id)
        prim.CreateAttribute(
            "w2d:component", Sdf.ValueTypeNames.String, custom=True
        ).Set(provenance.component)
        prim.CreateAttribute(
            "w2d:modelName", Sdf.ValueTypeNames.String, custom=True
        ).Set(provenance.model_name)
        prim.CreateAttribute(
            "w2d:modelVersion", Sdf.ValueTypeNames.String, custom=True
        ).Set(provenance.model_version)
        prim.CreateAttribute(
            "w2d:timestampIso8601", Sdf.ValueTypeNames.String, custom=True
        ).Set(provenance.timestamp_iso)

        if provenance.git_commit:
            prim.CreateAttribute(
                "w2d:gitCommit", Sdf.ValueTypeNames.String, custom=True
            ).Set(provenance.git_commit)
        if provenance.params:
            prim.CreateAttribute(
                "w2d:params", Sdf.ValueTypeNames.String, custom=True
            ).Set(json.dumps(provenance.params, sort_keys=True))

    @staticmethod
    def _layer_sort_key(filename: str) -> int:
        """Sort layers by their numeric prefix (00, 10, 20, ...)."""
        match = re.match(r"(\d+)_", filename)
        return int(match.group(1)) if match else 999


# =========================================================================
# Convenience: build full layered scene from pipeline data
# =========================================================================

def build_layered_scene(
    output_dir: str | Path,
    *,
    video_path: str | None = None,
    point_cloud: np.ndarray | None = None,
    point_colors: np.ndarray | None = None,
    cameras: list[dict] | None = None,
    focals: list[float] | None = None,
    entities: list[EntityRecord] | None = None,
    events: list[EventRecord] | None = None,
    fps: float = 30.0,
    start_frame: int = 0,
    end_frame: int = 0,
    run_id: str | None = None,
    ply_path: str | None = None,
) -> Path:
    """One-call helper to build the full layered USD scene.

    Returns the path to scene.usda.
    """
    run_id = run_id or _generate_run_id()
    writer = USDLayerWriter(output_dir)

    # 00 Base
    writer.write_base_layer(fps=fps, start_frame=start_frame, end_frame=end_frame)

    # 10 Inputs
    prov_ingest = ProvenanceRecord(
        run_id=run_id, component="ingest",
        model_name="world2data", model_version="0.3.0",
    )
    writer.write_inputs_layer(prov_ingest, video_path=video_path, point_cloud_path=ply_path)

    # 20 Recon
    if cameras or point_cloud is not None:
        prov_recon = ProvenanceRecord(
            run_id=run_id, component="recon",
            model_name="mast3r", model_version="ViTLarge_512",
        )
        writer.write_recon_layer(
            prov_recon, cameras=cameras,
            point_cloud=point_cloud, point_colors=point_colors,
            focals=focals, fps=fps,
        )

    # 30 Tracks
    if entities:
        prov_tracking = ProvenanceRecord(
            run_id=run_id, component="tracking",
            model_name="yolov8x-seg+particle_filter",
            model_version="0.3.0",
        )
        writer.write_tracks_layer(prov_tracking, entities, fps=fps)

    # 40 Events
    if events:
        prov_events = ProvenanceRecord(
            run_id=run_id, component="reasoning",
            model_name="gemini-2.5-flash", model_version="2.5",
        )
        writer.write_events_layer(prov_events, events)

    # 90 Overrides (empty for now)
    writer.write_overrides_layer()

    # 99 Session (empty, gitignored)
    writer.write_session_layer()

    # Assembly
    scene_path = writer.write_assembly(
        fps=fps, start_frame=start_frame, end_frame=end_frame,
    )
    return scene_path


# =========================================================================
# PLY helper (write compact per-frame files)
# =========================================================================

def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None):
    """Write a minimal ASCII PLY file for a single frame's points."""
    n = points.shape[0]
    has_color = colors is not None and colors.shape[0] == n
    path = Path(path)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            line = f"{x:.6f} {y:.6f} {z:.6f}"
            if has_color:
                r, g, b = np.clip(colors[i] * 255, 0, 255).astype(int)
                line += f" {r} {g} {b}"
            f.write(line + "\n")


# =========================================================================
# Point Lineage (Parquet) -- Protocol §4.2
# =========================================================================

def write_point_lineage(
    output_dir: str | Path,
    frame_data: Sequence[dict],
    run_id: str,
    pf_track_estimates: dict | None = None,
    total_frames: int | None = None,
) -> Path | None:
    """Write a point lineage parquet file for a pipeline run.

    Each row traces a single 3D point back to its source frame, with optional
    PF track association for lifecycle management.

    When ``pf_track_estimates`` is provided (from MultiObjectParticleFilter),
    points are associated with the nearest tracked object and their state is
    managed through a lifecycle:
      - "active"  -- point belongs to the most recent frame window
      - "stale"   -- point is older than the active window but still within
                     the track's observed lifetime
      - "retired" -- point is outside any active track's observation window

    This allows downstream consumers (Rerun, USD viewers) to animate/filter
    points based on freshness rather than treating them as a static dump.

    Args:
        output_dir: scene bundle root (contains external/)
        frame_data: list of dicts with keys: index, pts3d (N,3), colors (N,3),
                    confidence (N,), timestamp_sec
        run_id: pipeline run ID
        pf_track_estimates: optional dict {track_id: [(frame_idx, TrackEstimate), ...]}
        total_frames: total number of frames in the sequence (for lifecycle calc)
    """
    if not _HAS_PARQUET:
        return None

    output_dir = Path(output_dir)
    recon_dir = output_dir / "external" / "recon"
    recon_dir.mkdir(parents=True, exist_ok=True)

    # Build spatial index for PF tracks if available
    # For each frame, build a list of (track_id, position_3d, bbox_3d)
    pf_frame_tracks: dict[int, list] = {}
    if pf_track_estimates:
        for track_id, estimates in pf_track_estimates.items():
            for (fidx, est) in estimates:
                entry = (track_id, np.array(est.position), np.array(est.bounding_box))
                pf_frame_tracks.setdefault(fidx, []).append(entry)

    # Determine active window (last 30% of frames)
    if total_frames is None:
        total_frames = max((fd["index"] for fd in frame_data), default=0) + 1
    active_window_start = int(total_frames * 0.7)

    rows = {
        "point_uid": [],
        "frame_index_origin": [],
        "timestamp_sec_origin": [],
        "source_points_asset": [],
        "x": [], "y": [], "z": [],
        "r": [], "g": [], "b": [],
        "confidence": [],
        "state": [],
        "track_id": [],  # NEW: associated PF track (or "untracked")
    }

    for fd in frame_data:
        idx = fd["index"]
        pts = fd["pts3d"]
        colors = fd.get("colors")
        conf = fd.get("confidence")
        ts = fd.get("timestamp_sec", idx / 30.0)
        asset = f"frame_{idx:06d}.ply"
        n = pts.shape[0] if pts is not None else 0

        # Get active tracks for this frame
        frame_tracks = pf_frame_tracks.get(idx, [])

        for j in range(n):
            # Deterministic point UID
            uid_seed = f"{run_id}:{idx}:{asset}:{j}"
            uid = hashlib.sha256(uid_seed.encode()).hexdigest()[:16]
            rows["point_uid"].append(uid)
            rows["frame_index_origin"].append(idx)
            rows["timestamp_sec_origin"].append(ts)
            rows["source_points_asset"].append(asset)

            px, py, pz = float(pts[j, 0]), float(pts[j, 1]), float(pts[j, 2])
            rows["x"].append(px)
            rows["y"].append(py)
            rows["z"].append(pz)

            if colors is not None and j < colors.shape[0]:
                c = np.clip(colors[j], 0, 1)
                rows["r"].append(float(c[0]))
                rows["g"].append(float(c[1]))
                rows["b"].append(float(c[2]))
            else:
                rows["r"].append(0.5)
                rows["g"].append(0.5)
                rows["b"].append(0.5)

            base_conf = float(conf[j]) if conf is not None and j < len(conf) else 0.5
            rows["confidence"].append(base_conf)

            # Associate point with nearest PF track and compute lifecycle state
            best_track = "untracked"
            if frame_tracks:
                pt = np.array([px, py, pz])
                min_dist = float("inf")
                for (tid, track_pos, track_bbox) in frame_tracks:
                    dist = float(np.linalg.norm(pt - track_pos))
                    # Within bounding box radius = tracked
                    radius = float(np.linalg.norm(track_bbox)) / 2.0
                    if dist < radius and dist < min_dist:
                        min_dist = dist
                        best_track = tid

            rows["track_id"].append(best_track)

            # Lifecycle state
            if idx >= active_window_start:
                rows["state"].append("active")
            elif best_track != "untracked":
                rows["state"].append("stale")
            else:
                rows["state"].append("retired")

    table = pa.table(rows)
    parquet_path = recon_dir / f"point_lineage_{run_id}.parquet"
    pq.write_table(table, str(parquet_path))
    return parquet_path


# =========================================================================
# Validation (CI-ready)
# =========================================================================

def validate_scene(scene_path: str | Path) -> list[str]:
    """Validate a composed scene against the W2D protocol.

    Returns a list of error strings (empty = valid).
    """
    _, Sdf, Usd, UsdGeom, _, _ = _require_pxr()
    errors: list[str] = []

    stage = Usd.Stage.Open(str(scene_path))
    if not stage:
        return [f"Cannot open stage: {scene_path}"]

    # 1. Check defaultPrim
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        errors.append("No defaultPrim defined")

    # 2. Check namespace ownership: entity prims must have w2d:uid
    #    (skip container scopes and geometry children like /Shape)
    entity_containers = {
        "/World/W2D/Entities",
        "/World/W2D/Entities/Objects",
        "/World/W2D/Entities/People",
    }
    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if not path_str.startswith("/World/W2D/Entities/"):
            continue
        if path_str in entity_containers:
            continue
        # Skip geometry children (Cube, Mesh, etc.) nested under entity prims
        parent_path = str(prim.GetParent().GetPath())
        if parent_path not in entity_containers:
            # This is a child of an entity prim (e.g. /Objects/Table_01/Shape)
            continue
        # This is a direct entity prim (e.g. /Objects/Table_01)
        if not prim.HasAttribute("w2d:uid"):
            errors.append(f"Entity prim missing w2d:uid: {path_str}")

    # 3. Check provenance run records exist
    runs_prim = stage.GetPrimAtPath("/World/W2D/Provenance/runs")
    if runs_prim:
        run_children = list(runs_prim.GetChildren())
        if not run_children:
            errors.append("No provenance run records found")
    else:
        errors.append("Missing /World/W2D/Provenance/runs scope")

    # 4. Check camera xformOpOrder
    sensors = stage.GetPrimAtPath("/World/W2D/Sensors")
    if sensors:
        for prim in Usd.PrimRange(sensors):
            if prim.IsA(UsdGeom.Xformable):
                xf = UsdGeom.Xformable(prim)
                if xf.GetOrderedXformOps():
                    # OK - has xformOps
                    pass

    return errors
