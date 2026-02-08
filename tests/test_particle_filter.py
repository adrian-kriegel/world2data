from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import pytest

from world2data.model import AABB2D, CameraIntrinsics, Detection2D, FrameContext, Particle, ParticleState
from world2data.point_cloud import NumpyPointCloudOps
from world2data.particle_filter import (
    BackProjectionIoUCost,
    ConstantVelocityMotionModel,
    IoUPointCloudCost,
    MultiObjectParticleFilter,
    ParticleFilterConfig,
    PointCloudAlignmentCost,
    SingleObjectParticleFilter,
    build_resampler,
)
from world2data.vision import camera_pose_from_world_position, project_box_to_image_aabb, world_to_camera


@dataclass(frozen=True)
class SceneBox:
    label: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        width_px=640,
        height_px=480,
        fx_px=620.0,
        fy_px=620.0,
        cx_px=320.0,
        cy_px=240.0,
    )


def _scene_boxes() -> tuple[SceneBox, ...]:
    return (
        SceneBox(label="box", center=(-1.8, 0.1, 8.3), size=(0.9, 1.0, 0.8)),
        SceneBox(label="box", center=(-0.3, -0.2, 8.6), size=(1.1, 1.1, 1.0)),
        SceneBox(label="box", center=(1.4, 0.2, 8.0), size=(0.8, 0.9, 0.9)),
        SceneBox(label="crate", center=(2.4, -0.4, 8.9), size=(1.2, 1.0, 1.1)),
    )


def _cube_points(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    *,
    steps_per_axis: int = 5,
) -> tuple[tuple[float, float, float], ...]:
    cx, cy, cz = center
    sx, sy, sz = size
    xs = [cx - sx * 0.5 + sx * i / (steps_per_axis - 1) for i in range(steps_per_axis)]
    ys = [cy - sy * 0.5 + sy * i / (steps_per_axis - 1) for i in range(steps_per_axis)]
    zs = [cz - sz * 0.5 + sz * i / (steps_per_axis - 1) for i in range(steps_per_axis)]

    points: list[tuple[float, float, float]] = []
    for x in xs:
        for y in ys:
            for z in zs:
                if (
                    abs(x - (cx - sx * 0.5)) < 1e-9
                    or abs(x - (cx + sx * 0.5)) < 1e-9
                    or abs(y - (cy - sy * 0.5)) < 1e-9
                    or abs(y - (cy + sy * 0.5)) < 1e-9
                    or abs(z - (cz - sz * 0.5)) < 1e-9
                    or abs(z - (cz + sz * 0.5)) < 1e-9
                ):
                    points.append((x, y, z))
    return tuple(points)


def _visible_sparse_subset(
    points: tuple[tuple[float, float, float], ...],
    camera_x: float,
    *,
    keep_probability: float,
    rng: random.Random,
) -> tuple[tuple[float, float, float], ...]:
    pose = camera_pose_from_world_position((camera_x, 0.0, 0.0))
    visible: list[tuple[float, float, float]] = []
    for point in points:
        camera_point = world_to_camera(point, pose)
        if camera_point[2] <= 0.0:
            continue
        if rng.random() <= keep_probability:
            visible.append(point)
    return tuple(visible)


def _mock_sparse_point_cloud(
    scene: tuple[SceneBox, ...],
    camera_x: float,
    rng: random.Random,
) -> tuple[tuple[float, float, float], ...]:
    sampled: list[tuple[float, float, float]] = []
    for box in scene:
        points = _cube_points(box.center, box.size, steps_per_axis=6)
        sampled.extend(_visible_sparse_subset(points, camera_x, keep_probability=0.18, rng=rng))
    return tuple(sampled)


def _mock_yolo_from_scene(
    scene: tuple[SceneBox, ...],
    intrinsics: CameraIntrinsics,
    camera_x: float,
    rng: random.Random,
    *,
    label_overrides_by_index: dict[int, str] | None = None,
    dropped_indices: set[int] | None = None,
) -> tuple[Detection2D, ...]:
    pose = camera_pose_from_world_position((camera_x, 0.0, 0.0))
    detections: list[Detection2D] = []
    for index, box in enumerate(scene):
        if dropped_indices and index in dropped_indices:
            continue
        aabb = project_box_to_image_aabb(box.center, box.size, intrinsics, pose)
        if aabb is None:
            continue

        # Small detector jitter to avoid a perfect synthetic oracle.
        dx = rng.uniform(-0.5, 0.5)
        dy = rng.uniform(-0.5, 0.5)
        jittered = AABB2D(
            x_min=max(0.0, aabb.x_min + dx),
            y_min=max(0.0, aabb.y_min + dy),
            x_max=min(float(intrinsics.width_px), aabb.x_max + dx),
            y_max=min(float(intrinsics.height_px), aabb.y_max + dy),
        )

        if jittered.x_max <= jittered.x_min or jittered.y_max <= jittered.y_min:
            continue
        label = (
            label_overrides_by_index[index]
            if label_overrides_by_index and index in label_overrides_by_index
            else box.label
        )
        detections.append(Detection2D(label=label, aabb=jittered, confidence=0.95))
    return tuple(detections)


def _frame(
    frame_index: int,
    camera_x: float,
    intrinsics: CameraIntrinsics,
    detections: tuple[Detection2D, ...],
    point_cloud_points: tuple[tuple[float, float, float], ...] = (),
) -> FrameContext:
    return FrameContext(
        frame_index=frame_index,
        timestamp_s=frame_index / 30.0,
        dt_s=1.0 / 30.0,
        camera_pose=camera_pose_from_world_position((camera_x, 0.0, 0.0)),
        camera_intrinsics=intrinsics,
        detections=detections,
        point_cloud_points=point_cloud_points,
    )


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _match_tracks_to_scene(result_estimates, scene: tuple[SceneBox, ...]) -> dict[str, SceneBox]:
    matches: dict[str, SceneBox] = {}
    used_indices: set[int] = set()

    for track_id, estimate in result_estimates.items():
        best_index = -1
        best_distance = float("inf")
        for index, box in enumerate(scene):
            if index in used_indices:
                continue
            if box.label != estimate.label:
                continue
            distance = _distance(estimate.position, box.center)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        assert best_index >= 0
        used_indices.add(best_index)
        matches[track_id] = scene[best_index]

    return matches


def _track_id_for_label(result, label: str) -> str:
    track_ids = [
        track_id
        for track_id, estimate in result.estimates.items()
        if estimate.label == label
    ]
    assert len(track_ids) == 1
    return track_ids[0]


def test_particle_filter_tracks_static_scene_with_moving_camera() -> None:
    intrinsics = _intrinsics()
    scene = _scene_boxes()
    rng = random.Random(123)

    tracker = MultiObjectParticleFilter(
        motion_model=ConstantVelocityMotionModel(
            position_process_std_m=0.008,
            velocity_process_std_mps=0.005,
            bbox_process_std_m=0.004,
            mass_process_std_kg=0.001,
        ),
        cost_function=BackProjectionIoUCost(),
        config=ParticleFilterConfig(
            particles_per_track=128,
            initial_depth_m=8.5,
            min_assignment_iou=0.08,
            resampler="systematic",
        ),
        rng=rng,
    )

    result = None
    frame_count = 18
    for frame_index in range(frame_count):
        camera_x = -1.2 + (2.4 * frame_index / (frame_count - 1))
        detections = _mock_yolo_from_scene(scene, intrinsics, camera_x, rng)
        result = tracker.step(_frame(frame_index, camera_x, intrinsics, detections))

    assert result is not None
    assert len(result.estimates) == len(scene)

    matched = _match_tracks_to_scene(result.estimates, scene)
    for track_id, scene_box in matched.items():
        error = _distance(result.estimates[track_id].position, scene_box.center)
        assert error < 0.70

    # Weights are normalized independently per object filter.
    for particles in result.particles_by_track.values():
        weight_sum = sum(particle.weight for particle in particles)
        assert weight_sum == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("resampler", ["systematic", "stratified", "multinomial"])
def test_resampler_is_selectable(resampler: str) -> None:
    intrinsics = _intrinsics()
    scene = _scene_boxes()[:2]
    rng = random.Random(999)

    tracker = MultiObjectParticleFilter(
        config=ParticleFilterConfig(
            particles_per_track=64,
            initial_depth_m=8.0,
            min_assignment_iou=0.05,
            resampler=resampler,
        ),
        rng=rng,
    )

    result = None
    for frame_index in range(8):
        camera_x = -0.6 + (1.2 * frame_index / 7.0)
        detections = _mock_yolo_from_scene(scene, intrinsics, camera_x, rng)
        result = tracker.step(_frame(frame_index, camera_x, intrinsics, detections))

    assert result is not None
    assert len(result.estimates) == 2
    for estimate in result.estimates.values():
        assert estimate.particle_count == 64


def test_back_projection_iou_cost_prefers_correct_alignment() -> None:
    intrinsics = _intrinsics()
    pose = camera_pose_from_world_position((0.0, 0.0, 0.0))

    true_center = (0.3, -0.1, 8.0)
    true_size = (1.0, 0.9, 0.8)
    detection_aabb = project_box_to_image_aabb(true_center, true_size, intrinsics, pose)
    assert detection_aabb is not None

    frame = FrameContext(
        frame_index=0,
        timestamp_s=0.0,
        dt_s=1.0 / 30.0,
        camera_pose=pose,
        camera_intrinsics=intrinsics,
        detections=(Detection2D(label="box", aabb=detection_aabb, confidence=1.0),),
    )

    aligned_particle = Particle(
        particle_id="p-1",
        track_id="box-1",
        label="box",
        state=ParticleState(
            position=true_center,
            bounding_box=true_size,
            mass=1.0,
            velocity=(0.0, 0.0, 0.0),
        ),
        weight=1.0,
        age=0,
    )
    shifted_particle = aligned_particle.evolved(
        state=ParticleState(
            position=(true_center[0] + 1.5, true_center[1], true_center[2]),
            bounding_box=true_size,
            mass=1.0,
            velocity=(0.0, 0.0, 0.0),
        )
    )

    cost = BackProjectionIoUCost()
    detection = frame.detections[0]
    aligned_score = cost.score(aligned_particle, detection, frame)
    shifted_score = cost.score(shifted_particle, detection, frame)

    assert aligned_score > 0.95
    assert shifted_score < aligned_score


def test_spawn_prefers_point_cloud_depth_for_new_track() -> None:
    intrinsics = _intrinsics()
    pose = camera_pose_from_world_position((0.0, 0.0, 0.0))
    center = (0.2, -0.1, 3.2)
    size = (0.8, 0.8, 0.8)
    aabb = project_box_to_image_aabb(center, size, intrinsics, pose)
    assert aabb is not None

    detection = Detection2D(label="box", aabb=aabb, confidence=1.0)
    points = _cube_points(center, size, steps_per_axis=6)
    frame = _frame(
        frame_index=0,
        camera_x=0.0,
        intrinsics=intrinsics,
        detections=(detection,),
        point_cloud_points=points,
    )

    tracker = MultiObjectParticleFilter(
        config=ParticleFilterConfig(
            particles_per_track=96,
            # Deliberately wrong fallback depth to verify spawn uses matched cloud points.
            initial_depth_m=12.0,
            point_cloud_backend="numpy",
            min_assignment_iou=0.05,
        ),
        rng=random.Random(2026),
    )
    result = tracker.step(frame)

    assert len(result.estimates) == 1
    estimate = next(iter(result.estimates.values()))
    assert abs(estimate.position[2] - center[2]) < 0.9
    assert int(result.diagnostics.get("spawned_from_cloud_tracks", 0)) == 1
    assert int(result.diagnostics.get("spawned_fallback_tracks", 0)) == 0


def test_track_stitched_point_cloud_is_per_item_and_accumulates() -> None:
    intrinsics = _intrinsics()
    scene = _scene_boxes()[:3]
    rng = random.Random(321)

    tracker = MultiObjectParticleFilter(
        motion_model=ConstantVelocityMotionModel(
            position_process_std_m=0.008,
            velocity_process_std_mps=0.005,
            bbox_process_std_m=0.004,
            mass_process_std_kg=0.001,
        ),
        cost_function=IoUPointCloudCost(
            point_cloud_cost=PointCloudAlignmentCost(
                distance_sigma_m=0.18,
                min_observed_points=3,
                bootstrap_alignment=0.5,
            ),
            alignment_influence=1.2,
            zero_alignment_threshold=0.02,
        ),
        config=ParticleFilterConfig(
            particles_per_track=96,
            initial_depth_m=8.3,
            min_assignment_iou=0.08,
            stitched_voxel_size_m=0.04,
            resampler="systematic",
        ),
        rng=rng,
    )

    result = None
    frame_count = 20
    for frame_index in range(frame_count):
        camera_x = -1.2 + (2.4 * frame_index / (frame_count - 1))
        detections = _mock_yolo_from_scene(scene, intrinsics, camera_x, rng)
        point_cloud_points = _mock_sparse_point_cloud(scene, camera_x, rng)
        result = tracker.step(
            _frame(
                frame_index,
                camera_x,
                intrinsics,
                detections,
                point_cloud_points=point_cloud_points,
            )
        )

    assert result is not None
    assert len(result.stitched_points_by_track) == len(scene)
    for track_points in result.stitched_points_by_track.values():
        assert len(track_points) > 8

    # Stitched clouds are independent per track item.
    stitched_keys = sorted(result.stitched_points_by_track.keys())
    assert len(stitched_keys) == len(set(stitched_keys))


def test_track_stitching_uses_icp_registration_for_drifted_observations() -> None:
    intrinsics = _intrinsics()
    pose = camera_pose_from_world_position((0.0, 0.0, 0.0))
    center = (0.0, 0.0, 8.0)
    size = (1.0, 1.0, 1.0)

    detection_aabb = project_box_to_image_aabb(center, size, intrinsics, pose)
    assert detection_aabb is not None
    detection = Detection2D(label="box", aabb=detection_aabb, confidence=1.0)

    particle = Particle(
        particle_id="p-0",
        track_id="box-1",
        label="box",
        state=ParticleState(
            position=center,
            bounding_box=size,
            mass=1.0,
            velocity=(0.0, 0.0, 0.0),
        ),
        weight=1.0,
        age=0,
    )

    tracker = SingleObjectParticleFilter(
        track_id="box-1",
        label="box",
        particles=[particle],
        motion_model=ConstantVelocityMotionModel(
            position_process_std_m=0.0,
            velocity_process_std_mps=0.0,
            bbox_process_std_m=0.0,
            mass_process_std_kg=0.0,
        ),
        cost_function=BackProjectionIoUCost(),
        resampler=build_resampler("systematic"),
        rng=random.Random(1234),
        stitched_voxel_size_m=0.01,
        max_stitched_points=5000,
        point_cloud_ops=NumpyPointCloudOps(),
        stitched_icp_max_iterations=25,
        stitched_icp_tolerance_m=1e-6,
        stitched_icp_max_correspondence_distance_m=0.8,
        stitched_icp_min_correspondence_ratio=0.5,
        stitched_icp_max_mean_error_m=0.02,
    )

    base_cloud = _cube_points(center, size, steps_per_axis=5)
    drifted_cloud = tuple(
        (x + 0.25, y - 0.08, z + 0.06)
        for x, y, z in base_cloud
    )

    frame0 = FrameContext(
        frame_index=0,
        timestamp_s=0.0,
        dt_s=1.0 / 30.0,
        camera_pose=pose,
        camera_intrinsics=intrinsics,
        detections=(detection,),
        point_cloud_points=base_cloud,
    )
    frame1 = FrameContext(
        frame_index=1,
        timestamp_s=1.0 / 30.0,
        dt_s=1.0 / 30.0,
        camera_pose=pose,
        camera_intrinsics=intrinsics,
        detections=(detection,),
        point_cloud_points=drifted_cloud,
    )

    tracker.update(frame0, detection)
    tracker.update(frame1, detection)

    stitched = np.asarray(tracker.stitched_points, dtype=np.float64)
    assert len(stitched) > 0
    centroid = np.mean(stitched, axis=0)
    assert centroid[0] == pytest.approx(center[0], abs=0.08)
    assert centroid[1] == pytest.approx(center[1], abs=0.08)
    assert centroid[2] == pytest.approx(center[2], abs=0.08)


def test_iou_point_cloud_cost_can_zero_on_misalignment() -> None:
    intrinsics = _intrinsics()
    pose = camera_pose_from_world_position((0.0, 0.0, 0.0))

    near_center = (0.0, 0.0, 8.0)
    far_center = (2.0, 0.0, 8.0)
    size = (1.0, 1.0, 1.0)

    detection_box = project_box_to_image_aabb(far_center, size, intrinsics, pose)
    assert detection_box is not None
    detection = Detection2D(label="box", aabb=detection_box, confidence=1.0)

    frame = FrameContext(
        frame_index=0,
        timestamp_s=0.0,
        dt_s=1.0 / 30.0,
        camera_pose=pose,
        camera_intrinsics=intrinsics,
        detections=(detection,),
        point_cloud_points=_cube_points(far_center, size, steps_per_axis=5),
    )

    particle = Particle(
        particle_id="p-0",
        track_id="box-1",
        label="box",
        state=ParticleState(
            position=far_center,
            bounding_box=size,
            mass=1.0,
            velocity=(0.0, 0.0, 0.0),
        ),
    )

    stitched_misaligned = _cube_points(near_center, size, steps_per_axis=5)
    iou_only = BackProjectionIoUCost().score(particle, detection, frame)
    assert iou_only > 0.9

    combined = IoUPointCloudCost(
        point_cloud_cost=PointCloudAlignmentCost(
            distance_sigma_m=0.06,
            min_observed_points=3,
            bootstrap_alignment=0.5,
        ),
        alignment_influence=1.2,
        zero_alignment_threshold=0.05,
    )
    combined_score = combined.score(
        particle,
        detection,
        frame,
        stitched_track_points=stitched_misaligned,
    )

    assert combined_score == 0.0


def test_label_flicker_updates_track_label_database_without_losing_track() -> None:
    intrinsics = _intrinsics()
    scene = (
        SceneBox(label="mug", center=(-0.7, -0.1, 8.2), size=(0.8, 0.9, 0.8)),
        SceneBox(label="book", center=(1.0, 0.2, 8.5), size=(1.1, 0.9, 0.4)),
    )
    rng = random.Random(7788)

    tracker = MultiObjectParticleFilter(
        motion_model=ConstantVelocityMotionModel(
            position_process_std_m=0.007,
            velocity_process_std_mps=0.004,
            bbox_process_std_m=0.003,
            mass_process_std_kg=0.001,
        ),
        cost_function=BackProjectionIoUCost(),
        config=ParticleFilterConfig(
            particles_per_track=96,
            initial_depth_m=8.3,
            min_assignment_iou=0.08,
            min_label_flicker_assignment_iou=0.32,
            min_label_flicker_particle_support_ratio=0.55,
            label_flicker_iou_margin=0.06,
            resampler="systematic",
        ),
        rng=rng,
    )

    result = None
    frame_count = 16
    for frame_index in range(frame_count):
        camera_x = -0.9 + (1.8 * frame_index / (frame_count - 1))
        label_override = {0: "remote"} if frame_index in {6, 7, 8} else None
        detections = _mock_yolo_from_scene(
            scene,
            intrinsics,
            camera_x,
            rng,
            label_overrides_by_index=label_override,
        )
        result = tracker.step(_frame(frame_index, camera_x, intrinsics, detections))

    assert result is not None
    assert len(result.estimates) == 2
    mug_track_id = _track_id_for_label(result, "mug")
    mug_counts = result.track_label_counts_by_track[mug_track_id]
    assert mug_counts.get("mug", 0) > 0
    assert mug_counts.get("remote", 0) > 0

    mug_estimate = result.estimates[mug_track_id]
    assert _distance(mug_estimate.position, scene[0].center) < 0.75


def test_adjacent_objects_missing_one_does_not_relabel_to_other_object_class() -> None:
    intrinsics = _intrinsics()
    scene = (
        SceneBox(label="mug", center=(-0.2, 0.0, 8.3), size=(0.7, 0.8, 0.7)),
        SceneBox(label="remote", center=(0.45, 0.02, 8.35), size=(0.8, 0.4, 0.5)),
    )
    rng = random.Random(9912)

    tracker = MultiObjectParticleFilter(
        motion_model=ConstantVelocityMotionModel(
            position_process_std_m=0.007,
            velocity_process_std_mps=0.004,
            bbox_process_std_m=0.003,
            mass_process_std_kg=0.001,
        ),
        cost_function=BackProjectionIoUCost(),
        config=ParticleFilterConfig(
            particles_per_track=96,
            initial_depth_m=8.3,
            min_assignment_iou=0.08,
            min_label_flicker_assignment_iou=0.34,
            min_label_flicker_particle_support_ratio=0.60,
            label_flicker_iou_margin=0.12,
            resampler="systematic",
        ),
        rng=rng,
    )

    mug_track_id = ""
    remote_track_id = ""
    before_missing: dict[str, dict[str, int]] = {}
    during_missing: dict[str, dict[str, int]] = {}

    frame_count = 12
    for frame_index in range(frame_count):
        camera_x = -0.7 + (1.4 * frame_index / (frame_count - 1))
        dropped = {0} if frame_index == 7 else None
        detections = _mock_yolo_from_scene(
            scene,
            intrinsics,
            camera_x,
            rng,
            dropped_indices=dropped,
        )
        result = tracker.step(_frame(frame_index, camera_x, intrinsics, detections))

        if frame_index == 6:
            mug_track_id = _track_id_for_label(result, "mug")
            remote_track_id = _track_id_for_label(result, "remote")
            before_missing = {
                track_id: dict(label_counts)
                for track_id, label_counts in result.track_label_counts_by_track.items()
            }
        if frame_index == 7:
            during_missing = {
                track_id: dict(label_counts)
                for track_id, label_counts in result.track_label_counts_by_track.items()
            }

    assert mug_track_id
    assert remote_track_id
    assert during_missing[mug_track_id].get("mug", 0) == before_missing[mug_track_id].get("mug", 0)
    assert during_missing[mug_track_id].get("remote", 0) == before_missing[mug_track_id].get("remote", 0)
    assert during_missing[remote_track_id].get("remote", 0) > before_missing[remote_track_id].get("remote", 0)
