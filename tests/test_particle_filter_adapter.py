from __future__ import annotations

from pathlib import Path

import pytest

from world2data.calibration import CalibrationResult, write_calibration_usda
from world2data.model import AABB2D, CameraIntrinsics
from world2data.particle_filter import ParticleFilterConfig
from world2data.particle_filter_adapter import write_particle_filter_tracks_usd
from world2data.vision import camera_pose_from_world_position, project_box_to_image_aabb
from world2data.yolo_adapter import (
    YoloFrameDetections,
    YoloRawDetection,
    yolo_observations_to_stage,
)


def _write_camera_poses_layer(
    output_path: Path,
    *,
    frame_count: int,
    fps: float,
) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(float(fps))
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(float(frame_count - 1))

    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    UsdGeom.Scope.Define(stage, "/World/W2D")
    UsdGeom.Scope.Define(stage, "/World/W2D/Sensors")
    UsdGeom.Scope.Define(stage, "/World/W2D/Sensors/CameraPoses")
    UsdGeom.Scope.Define(stage, "/World/W2D/Sensors/CameraPoses/Frames")

    for frame_index in range(frame_count):
        camera_x = -0.4 + 0.2 * frame_index
        pose = camera_pose_from_world_position((camera_x, 0.0, 0.0))
        prim = UsdGeom.Scope.Define(
            stage,
            f"/World/W2D/Sensors/CameraPoses/Frames/f_{frame_index:06d}",
        ).GetPrim()
        prim.CreateAttribute("w2d:frameIndex", Sdf.ValueTypeNames.Int, custom=True).Set(
            int(frame_index)
        )
        prim.CreateAttribute("w2d:timestampSec", Sdf.ValueTypeNames.Double, custom=True).Set(
            float(frame_index / fps)
        )
        prim.CreateAttribute(
            "w2d:rotationMatrix",
            Sdf.ValueTypeNames.Matrix3d,
            custom=True,
        ).Set(
            Gf.Matrix3d(
                pose.rotation[0][0], pose.rotation[0][1], pose.rotation[0][2],
                pose.rotation[1][0], pose.rotation[1][1], pose.rotation[1][2],
                pose.rotation[2][0], pose.rotation[2][1], pose.rotation[2][2],
            )
        )
        prim.CreateAttribute("w2d:translation", Sdf.ValueTypeNames.Float3, custom=True).Set(
            Gf.Vec3f(
                float(pose.translation[0]),
                float(pose.translation[1]),
                float(pose.translation[2]),
            )
        )
        prim.CreateAttribute(
            "w2d:poseConvention",
            Sdf.ValueTypeNames.String,
            custom=True,
        ).Set("world_to_camera")

    stage.GetRootLayer().Save()


def _write_point_cloud_layer(
    output_path: Path,
    *,
    frame_count: int,
    fps: float,
) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(float(fps))
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(float(frame_count - 1))

    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    UsdGeom.Scope.Define(stage, "/World/W2D")
    UsdGeom.Scope.Define(stage, "/World/W2D/Reconstruction")
    UsdGeom.Scope.Define(stage, "/World/W2D/Reconstruction/PointCloudFrames")

    object_center = (0.1, -0.1, 8.1)
    offsets = (
        (-0.2, -0.2, -0.2),
        (0.2, -0.2, -0.2),
        (-0.2, 0.2, -0.2),
        (0.2, 0.2, -0.2),
        (-0.2, -0.2, 0.2),
        (0.2, -0.2, 0.2),
        (-0.2, 0.2, 0.2),
        (0.2, 0.2, 0.2),
    )
    points = [
        Gf.Vec3f(
            float(object_center[0] + dx),
            float(object_center[1] + dy),
            float(object_center[2] + dz),
        )
        for dx, dy, dz in offsets
    ]

    for frame_index in range(frame_count):
        prim = UsdGeom.Scope.Define(
            stage,
            f"/World/W2D/Reconstruction/PointCloudFrames/f_{frame_index:06d}",
        ).GetPrim()
        prim.CreateAttribute("w2d:frameIndex", Sdf.ValueTypeNames.Int, custom=True).Set(
            int(frame_index)
        )
        prim.CreateAttribute("w2d:timestampSec", Sdf.ValueTypeNames.Double, custom=True).Set(
            float(frame_index / fps)
        )
        prim.CreateAttribute(
            "w2d:points",
            Sdf.ValueTypeNames.Float3Array,
            custom=True,
        ).Set(Vt.Vec3fArray(points))

    stage.GetRootLayer().Save()


def _write_yolo_layer(
    output_path: Path,
    *,
    intrinsics: CameraIntrinsics,
    frame_count: int,
    fps: float,
) -> None:
    frames: list[YoloFrameDetections] = []
    center = (0.1, -0.1, 8.1)
    size = (0.6, 0.6, 0.6)

    for frame_index in range(frame_count):
        camera_x = -0.4 + 0.2 * frame_index
        pose = camera_pose_from_world_position((camera_x, 0.0, 0.0))
        aabb = project_box_to_image_aabb(center, size, intrinsics, pose)
        assert isinstance(aabb, AABB2D)
        frames.append(
            YoloFrameDetections(
                frame_index=frame_index,
                timestamp_s=float(frame_index / fps),
                image_width=intrinsics.width_px,
                image_height=intrinsics.height_px,
                detections=(
                    YoloRawDetection(
                        class_id=41,
                        label="cup",
                        confidence=0.9,
                        bbox_xyxy=(aabb.x_min, aabb.y_min, aabb.x_max, aabb.y_max),
                    ),
                ),
            )
        )

    stage = yolo_observations_to_stage(
        detections=tuple(frames),
        run_id="yolo-run-1",
        model_name="yolov8n.pt",
        model_version="8.4.12",
        params={"frame_step": 1},
        git_commit="deadbeef",
        time_codes_per_second=fps,
        video_uri="../external/inputs/demo.mp4",
    )
    stage.GetRootLayer().Export(str(output_path))


def test_particle_filter_adapter_consumes_protocol_layers_and_writes_tracks(tmp_path: Path) -> None:
    pytest.importorskip("pxr")
    from pxr import Usd

    frame_count = 5
    fps = 30.0
    intrinsics = CameraIntrinsics(
        width_px=640,
        height_px=480,
        fx_px=620.0,
        fy_px=620.0,
        cx_px=320.0,
        cy_px=240.0,
    )

    calibration_layer = tmp_path / "calibration.usda"
    write_calibration_usda(
        result=CalibrationResult(
            image_size=(intrinsics.width_px, intrinsics.height_px),
            camera_matrix=[
                [intrinsics.fx_px, 0.0, intrinsics.cx_px],
                [0.0, intrinsics.fy_px, intrinsics.cy_px],
                [0.0, 0.0, 1.0],
            ],
            distortion_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
            distortion_model="opencv_plumb_bob",
            reprojection_error_px=0.1,
            frames_sampled=frame_count,
            frames_used=frame_count,
        ),
        output_path=calibration_layer,
        run_id="calib-run-1",
        model_version="4.13.0",
        params={},
    )

    camera_poses_layer = tmp_path / "camera_poses.usda"
    _write_camera_poses_layer(
        camera_poses_layer,
        frame_count=frame_count,
        fps=fps,
    )

    point_cloud_layer = tmp_path / "point_cloud.usda"
    _write_point_cloud_layer(
        point_cloud_layer,
        frame_count=frame_count,
        fps=fps,
    )

    yolo_layer = tmp_path / "yolo.usda"
    _write_yolo_layer(
        yolo_layer,
        intrinsics=intrinsics,
        frame_count=frame_count,
        fps=fps,
    )

    output_layer = tmp_path / "tracks.usda"
    write_particle_filter_tracks_usd(
        calibration_layer=calibration_layer,
        camera_poses_layer=camera_poses_layer,
        yolo_layer=yolo_layer,
        point_cloud_layer=point_cloud_layer,
        output_path=output_layer,
        config=ParticleFilterConfig(
            particles_per_track=64,
            initial_depth_m=8.0,
            min_assignment_iou=0.05,
            point_cloud_backend="numpy",
        ),
        seed=17,
        run_id="pf-run-1",
    )

    stage = Usd.Stage.Open(str(output_layer))
    assert stage
    tracks_root = stage.GetPrimAtPath("/World/W2D/Tracks/ParticleFilter")
    assert tracks_root
    assert len(tracks_root.GetChildren()) >= 1

    track_prim = tracks_root.GetChildren()[0]
    assert track_prim.GetAttribute("w2d:trackId").Get() is not None
    assert track_prim.GetAttribute("w2d:meanBoundingBox").GetTimeSamples()
    assert track_prim.GetAttribute("xformOp:translate").GetTimeSamples()
    merged_rel = track_prim.GetRelationship("w2d:mergedPointCloud")
    assert merged_rel
    merged_targets = merged_rel.GetTargets()
    assert merged_targets
    merged_cloud_prim = stage.GetPrimAtPath(str(merged_targets[0]))
    assert merged_cloud_prim
    assert merged_cloud_prim.GetAttribute("w2d:trackId").Get() is not None
    assert int(merged_cloud_prim.GetAttribute("w2d:pointCount").Get() or 0) >= 1
    merged_points = merged_cloud_prim.GetAttribute("points").Get()
    assert merged_points is not None
    assert len(merged_points) >= 1

    entity_root = stage.GetPrimAtPath("/World/W2D/Entities/Objects")
    assert entity_root
    assert len(entity_root.GetChildren()) >= 1

    run_prim = stage.GetPrimAtPath("/World/W2D/Provenance/runs/pf_run_1")
    assert run_prim
    assert run_prim.GetAttribute("w2d:component").Get() == "tracking.particle_filter"
