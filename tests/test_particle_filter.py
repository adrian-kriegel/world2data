from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pytest

from world2data.model import AABB2D, CameraIntrinsics, Detection2D, FrameContext, Particle, ParticleState
from world2data.particle_filter import (
    BackProjectionIoUCost,
    ConstantVelocityMotionModel,
    MultiObjectParticleFilter,
    ParticleFilterConfig,
)
from world2data.vision import camera_pose_from_world_position, project_box_to_image_aabb


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


def _mock_yolo_from_scene(
    scene: tuple[SceneBox, ...],
    intrinsics: CameraIntrinsics,
    camera_x: float,
    rng: random.Random,
) -> tuple[Detection2D, ...]:
    pose = camera_pose_from_world_position((camera_x, 0.0, 0.0))
    detections: list[Detection2D] = []
    for box in scene:
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
        detections.append(Detection2D(label=box.label, aabb=jittered, confidence=0.95))
    return tuple(detections)


def _frame(
    frame_index: int,
    camera_x: float,
    intrinsics: CameraIntrinsics,
    detections: tuple[Detection2D, ...],
) -> FrameContext:
    return FrameContext(
        frame_index=frame_index,
        timestamp_s=frame_index / 30.0,
        dt_s=1.0 / 30.0,
        camera_pose=camera_pose_from_world_position((camera_x, 0.0, 0.0)),
        camera_intrinsics=intrinsics,
        detections=detections,
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
