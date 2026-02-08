from __future__ import annotations

"""Multi-object particle filter with separated motion, cost, and resampling components."""

import random
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
import os
from typing import Mapping, Sequence

import numpy as np

from .model import (
    CostFunction,
    Detection2D,
    FilterResult,
    FrameContext,
    MotionModel,
    Particle,
    ParticleState,
    Resampler,
    TrackEstimate,
)
from .point_cloud import PointCloudOps, create_point_cloud_ops
from .vision import back_project_pixel, project_box_to_image_aabb


@dataclass(frozen=True)
class ParticleFilterConfig:
    particles_per_track: int = 96
    initial_depth_m: float = 8.0
    initial_mass_kg: float = 1.0
    initial_position_std_m: float = 0.25
    initial_velocity_std_mps: float = 0.05
    initial_bbox_std_m: float = 0.05
    initial_mass_std_kg: float = 0.05
    min_assignment_iou: float = 0.10
    max_missed_frames: int = 8
    resampler: str = "systematic"
    stitched_voxel_size_m: float = 0.03
    max_stitched_points_per_track: int = 5000
    stitched_icp_max_iterations: int = 20
    stitched_icp_tolerance_m: float = 1e-3
    stitched_icp_max_correspondence_distance_m: float = 0.30
    stitched_icp_min_correspondence_ratio: float = 0.25
    stitched_icp_max_mean_error_m: float = 0.12
    point_cloud_backend: str = "pcl"
    require_point_cloud_backend: bool = False
    particle_support_iou_threshold: float = 0.20
    min_label_flicker_assignment_iou: float = 0.35
    min_label_flicker_particle_support_ratio: float = 0.60
    label_flicker_iou_margin: float = 0.08
    spawn_depth_quantile_trim: float = 0.10
    assignment_workers: int = 1
    particle_score_workers: int = 1


def _normalize_worker_count(worker_count: int) -> int:
    if worker_count > 0:
        return int(worker_count)
    return max(1, int(os.cpu_count() or 1))


class ConstantVelocityMotionModel(MotionModel):
    def __init__(
        self,
        *,
        position_process_std_m: float = 0.02,
        velocity_process_std_mps: float = 0.01,
        bbox_process_std_m: float = 0.005,
        mass_process_std_kg: float = 0.002,
    ) -> None:
        self._position_std = position_process_std_m
        self._velocity_std = velocity_process_std_mps
        self._bbox_std = bbox_process_std_m
        self._mass_std = mass_process_std_kg

    def propagate(self, particle: Particle, frame: FrameContext, rng: random.Random) -> Particle:
        dt = max(0.0, frame.dt_s)

        pos = np.asarray(particle.state.position, dtype=np.float64)
        vel = np.asarray(particle.state.velocity, dtype=np.float64)
        bbox = np.asarray(particle.state.bounding_box, dtype=np.float64)
        mass = float(particle.state.mass)

        vel_noise = np.asarray(
            [rng.gauss(0.0, self._velocity_std) for _ in range(3)],
            dtype=np.float64,
        )
        vel_next = vel + vel_noise

        pos_noise = np.asarray(
            [rng.gauss(0.0, self._position_std) for _ in range(3)],
            dtype=np.float64,
        )
        pos_next = pos + vel_next * dt + pos_noise

        bbox_noise = np.asarray(
            [rng.gauss(0.0, self._bbox_std) for _ in range(3)],
            dtype=np.float64,
        )
        bbox_next = np.maximum(1e-3, bbox + bbox_noise)

        mass_next = max(1e-6, mass + rng.gauss(0.0, self._mass_std))

        return particle.evolved(
            state=ParticleState(
                position=(float(pos_next[0]), float(pos_next[1]), float(pos_next[2])),
                bounding_box=(
                    float(bbox_next[0]),
                    float(bbox_next[1]),
                    float(bbox_next[2]),
                ),
                mass=mass_next,
                velocity=(float(vel_next[0]), float(vel_next[1]), float(vel_next[2])),
            ),
            age=particle.age + 1,
        )


class BackProjectionIoUCost(CostFunction):
    """Likelihood from 2D IoU between back-projected particle box and detection box."""

    def score(
        self,
        particle: Particle,
        detection: Detection2D,
        frame: FrameContext,
        *,
        stitched_track_points: Sequence[tuple[float, float, float]] | None = None,
    ) -> float:
        del stitched_track_points
        projected = project_box_to_image_aabb(
            center_world=particle.state.position,
            size_xyz=particle.state.bounding_box,
            intrinsics=frame.camera_intrinsics,
            pose=frame.camera_pose,
        )
        if projected is None:
            return 0.0

        iou = projected.iou(detection.aabb)
        return max(0.0, iou * max(0.0, detection.confidence))


class PointCloudAlignmentCost:
    """Track-level point-cloud alignment term for particle scoring."""

    def __init__(
        self,
        *,
        point_cloud_ops: PointCloudOps | None = None,
        distance_sigma_m: float = 0.15,
        min_observed_points: int = 4,
        bootstrap_alignment: float = 0.5,
        coverage_target_points: int = 24,
    ) -> None:
        self._point_cloud_ops = (
            point_cloud_ops
            if point_cloud_ops is not None
            else create_point_cloud_ops(requested_backend="pcl").backend
        )
        self._distance_sigma_m = max(1e-6, distance_sigma_m)
        self._min_observed_points = max(1, min_observed_points)
        self._bootstrap_alignment = float(np.clip(bootstrap_alignment, 0.0, 1.0))
        self._coverage_target_points = max(1, coverage_target_points)

    def crop_points_for_particle(
        self,
        particle: Particle,
        point_cloud_points: Sequence[tuple[float, float, float]],
    ) -> np.ndarray:
        return self._point_cloud_ops.crop_aabb(
            point_cloud_points,
            center=particle.state.position,
            size_xyz=particle.state.bounding_box,
        )

    def alignment_score(
        self,
        particle: Particle,
        frame: FrameContext,
        *,
        stitched_track_points: Sequence[tuple[float, float, float]] | None = None,
    ) -> tuple[float, bool]:
        """Returns (alignment_score in [0,1], has_point_cloud_signal)."""
        if not frame.point_cloud_points:
            return (self._bootstrap_alignment, False)

        observed = self.crop_points_for_particle(particle, frame.point_cloud_points)
        if len(observed) < self._min_observed_points:
            return (0.0, True)

        if stitched_track_points is None or len(stitched_track_points) == 0:
            return (self._bootstrap_alignment, True)

        stitched = np.asarray(stitched_track_points, dtype=np.float64)
        if stitched.ndim != 2 or stitched.shape[1] != 3:
            return (0.0, True)

        # nearest-neighbor alignment (observed -> stitched)
        nearest = self._point_cloud_ops.nearest_neighbor_distances(observed, stitched)
        if nearest.size == 0:
            return (0.0, True)
        mean_distance = float(np.mean(nearest))
        similarity = float(np.exp(-(mean_distance**2) / (2.0 * self._distance_sigma_m**2)))

        coverage = min(1.0, len(observed) / float(self._coverage_target_points))
        alignment = similarity * coverage
        return (float(np.clip(alignment, 0.0, 1.0)), True)


class IoUPointCloudCost(CostFunction):
    """Combined score: IoU term (positive-only) + point-cloud alignment term."""

    def __init__(
        self,
        *,
        point_cloud_ops: PointCloudOps | None = None,
        iou_cost: BackProjectionIoUCost | None = None,
        point_cloud_cost: PointCloudAlignmentCost | None = None,
        alignment_baseline: float = 0.5,
        alignment_influence: float = 1.0,
        zero_alignment_threshold: float = 0.05,
    ) -> None:
        selected_ops = (
            point_cloud_ops
            if point_cloud_ops is not None
            else create_point_cloud_ops(requested_backend="pcl").backend
        )
        self._iou_cost = iou_cost or BackProjectionIoUCost()
        self._point_cloud_cost = point_cloud_cost or PointCloudAlignmentCost(
            point_cloud_ops=selected_ops
        )
        self._alignment_baseline = float(np.clip(alignment_baseline, 0.0, 1.0))
        self._alignment_influence = alignment_influence
        self._zero_alignment_threshold = float(np.clip(zero_alignment_threshold, 0.0, 1.0))

    @property
    def point_cloud_cost(self) -> PointCloudAlignmentCost:
        return self._point_cloud_cost

    def score(
        self,
        particle: Particle,
        detection: Detection2D,
        frame: FrameContext,
        *,
        stitched_track_points: Sequence[tuple[float, float, float]] | None = None,
    ) -> float:
        # IoU contribution is positive-only and cannot decrease score by itself.
        base_score = self._iou_cost.score(
            particle,
            detection,
            frame,
            stitched_track_points=stitched_track_points,
        )
        if base_score <= 0.0:
            return 0.0

        alignment, has_signal = self._point_cloud_cost.alignment_score(
            particle,
            frame,
            stitched_track_points=stitched_track_points,
        )
        if not has_signal:
            return base_score

        if alignment < self._zero_alignment_threshold:
            return 0.0

        delta = (alignment - self._alignment_baseline) * self._alignment_influence
        return max(0.0, base_score * (1.0 + delta))


def _normalize_weights(particles: Sequence[Particle]) -> list[Particle]:
    if not particles:
        return []

    raw = np.asarray([max(0.0, particle.weight) for particle in particles], dtype=np.float64)
    total = float(raw.sum())

    if total <= 0.0:
        normalized = np.full(len(particles), 1.0 / len(particles), dtype=np.float64)
    else:
        normalized = raw / total

    return [
        particle.evolved(weight=float(weight))
        for particle, weight in zip(particles, normalized, strict=True)
    ]


class SystematicResampler(Resampler):
    def resample(self, particles: Sequence[Particle], count: int, rng: random.Random) -> list[Particle]:
        if count <= 0 or not particles:
            return []

        normalized = _normalize_weights(particles)
        cumulative = np.cumsum([particle.weight for particle in normalized], dtype=np.float64)

        step = 1.0 / count
        start = rng.random() * step
        points = start + step * np.arange(count, dtype=np.float64)
        indices = np.searchsorted(cumulative, points, side="left")
        indices = np.clip(indices, 0, len(normalized) - 1)

        return [normalized[int(index)].evolved(weight=1.0 / count) for index in indices]


class StratifiedResampler(Resampler):
    def resample(self, particles: Sequence[Particle], count: int, rng: random.Random) -> list[Particle]:
        if count <= 0 or not particles:
            return []

        normalized = _normalize_weights(particles)
        cumulative = np.cumsum([particle.weight for particle in normalized], dtype=np.float64)

        points = (np.arange(count, dtype=np.float64) + np.asarray([rng.random() for _ in range(count)])) / count
        indices = np.searchsorted(cumulative, points, side="left")
        indices = np.clip(indices, 0, len(normalized) - 1)

        return [normalized[int(index)].evolved(weight=1.0 / count) for index in indices]


class MultinomialResampler(Resampler):
    def resample(self, particles: Sequence[Particle], count: int, rng: random.Random) -> list[Particle]:
        if count <= 0 or not particles:
            return []

        normalized = _normalize_weights(particles)
        population = list(range(len(normalized)))
        weights = [particle.weight for particle in normalized]
        indices = rng.choices(population=population, weights=weights, k=count)
        return [normalized[index].evolved(weight=1.0 / count) for index in indices]


def build_resampler(name: str) -> Resampler:
    normalized = name.strip().lower()
    if normalized == "systematic":
        return SystematicResampler()
    if normalized == "stratified":
        return StratifiedResampler()
    if normalized == "multinomial":
        return MultinomialResampler()
    raise ValueError("Unknown resampler. Expected one of: systematic, stratified, multinomial")


class SingleObjectParticleFilter:
    def __init__(
        self,
        *,
        track_id: str,
        label: str,
        particles: list[Particle],
        motion_model: MotionModel,
        cost_function: CostFunction,
        resampler: Resampler,
        rng: random.Random,
        stitched_voxel_size_m: float,
        max_stitched_points: int,
        point_cloud_ops: PointCloudOps,
        stitched_icp_max_iterations: int,
        stitched_icp_tolerance_m: float,
        stitched_icp_max_correspondence_distance_m: float,
        stitched_icp_min_correspondence_ratio: float,
        stitched_icp_max_mean_error_m: float,
    ) -> None:
        self.track_id = track_id
        self.label = label
        self.particles = particles
        self._motion_model = motion_model
        self._cost_function = cost_function
        self._assignment_iou_cost = BackProjectionIoUCost()
        self._resampler = resampler
        self._rng = rng
        self._missed_frames = 0
        self._point_cloud_ops = point_cloud_ops
        self._stitched_points = np.empty((0, 3), dtype=np.float64)
        self._stitched_voxel_size_m = max(1e-6, stitched_voxel_size_m)
        self._max_stitched_points = max(1, max_stitched_points)
        self._stitched_icp_max_iterations = max(1, int(stitched_icp_max_iterations))
        self._stitched_icp_tolerance_m = max(1e-9, float(stitched_icp_tolerance_m))
        self._stitched_icp_max_correspondence_distance_m = max(
            1e-6,
            float(stitched_icp_max_correspondence_distance_m),
        )
        self._stitched_icp_min_correspondence_ratio = float(
            np.clip(stitched_icp_min_correspondence_ratio, 0.0, 1.0)
        )
        self._stitched_icp_max_mean_error_m = max(
            1e-6,
            float(stitched_icp_max_mean_error_m),
        )

    @property
    def missed_frames(self) -> int:
        return self._missed_frames

    @property
    def stitched_points(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(tuple(float(value) for value in row) for row in self._stitched_points.tolist())

    def predict(self, frame: FrameContext) -> None:
        self.particles = [
            self._motion_model.propagate(particle, frame, self._rng)
            for particle in self.particles
        ]

    def update(
        self,
        frame: FrameContext,
        detection: Detection2D | None,
        *,
        score_executor: Executor | None = None,
    ) -> None:
        if not self.particles:
            return

        if detection is None:
            self._missed_frames += 1
            self.particles = _normalize_weights(self.particles)
            return

        self._missed_frames = 0
        if score_executor is not None and len(self.particles) > 1:
            def _score_particle(particle: Particle) -> Particle:
                return particle.evolved(
                    weight=self._cost_function.score(
                        particle,
                        detection,
                        frame,
                        stitched_track_points=self._stitched_points,
                    )
                )

            scored = list(score_executor.map(_score_particle, self.particles))
        else:
            scored = [
                particle.evolved(
                    weight=self._cost_function.score(
                        particle,
                        detection,
                        frame,
                        stitched_track_points=self._stitched_points,
                    )
                )
                for particle in self.particles
            ]
        normalized = _normalize_weights(scored)
        self._update_stitched_points(frame, normalized)
        self.particles = self._resampler.resample(normalized, len(self.particles), self._rng)

    def mean_iou_for_detection(self, frame: FrameContext, detection: Detection2D) -> float:
        mean_iou, _, _ = self.detection_match_metrics(
            frame,
            detection,
            support_iou_threshold=0.20,
        )
        return mean_iou

    def detection_match_metrics(
        self,
        frame: FrameContext,
        detection: Detection2D,
        *,
        support_iou_threshold: float,
    ) -> tuple[float, float, int]:
        if not self.particles:
            return (0.0, 0.0, 0)

        particles = _normalize_weights(self.particles)
        score_sum = 0.0
        support_count = 0
        for particle in particles:
            iou_score = self._assignment_iou_cost.score(particle, detection, frame)
            score_sum += iou_score * particle.weight
            if iou_score >= support_iou_threshold:
                support_count += 1
        support_ratio = support_count / len(particles)
        return (float(score_sum), float(support_ratio), int(support_count))

    def _update_stitched_points(
        self,
        frame: FrameContext,
        weighted_particles: Sequence[Particle],
    ) -> None:
        if not frame.point_cloud_points:
            return
        if not weighted_particles:
            return

        best_particle = max(weighted_particles, key=lambda particle: particle.weight)
        observed = self._crop_points_for_particle(best_particle, frame.point_cloud_points)
        if len(observed) == 0:
            return

        if len(self._stitched_points) == 0:
            combined = observed
        else:
            registration = self._point_cloud_ops.register_icp(
                observed,
                self._stitched_points,
                max_iterations=self._stitched_icp_max_iterations,
                tolerance_m=self._stitched_icp_tolerance_m,
                max_correspondence_distance_m=self._stitched_icp_max_correspondence_distance_m,
            )
            if (
                registration.correspondence_ratio
                < self._stitched_icp_min_correspondence_ratio
                or registration.mean_error_m > self._stitched_icp_max_mean_error_m
            ):
                return
            combined = np.vstack(
                [self._stitched_points, registration.transformed_source_points]
            )
        self._stitched_points = self._downsample_and_limit(combined)

    def _crop_points_for_particle(
        self,
        particle: Particle,
        point_cloud_points: Sequence[tuple[float, float, float]],
    ) -> np.ndarray:
        return self._point_cloud_ops.crop_aabb(
            point_cloud_points,
            center=particle.state.position,
            size_xyz=particle.state.bounding_box,
        )

    def _downsample_and_limit(self, points: np.ndarray) -> np.ndarray:
        return self._point_cloud_ops.merge_downsample(
            np.empty((0, 3), dtype=np.float64),
            points,
            voxel_size_m=self._stitched_voxel_size_m,
            max_points=self._max_stitched_points,
        )

    def estimate(self) -> TrackEstimate:
        if not self.particles:
            raise ValueError("Cannot estimate track without particles")

        particles = _normalize_weights(self.particles)
        weights = np.asarray([particle.weight for particle in particles], dtype=np.float64)

        positions = np.asarray([particle.state.position for particle in particles], dtype=np.float64)
        bboxes = np.asarray([particle.state.bounding_box for particle in particles], dtype=np.float64)
        masses = np.asarray([particle.state.mass for particle in particles], dtype=np.float64)
        velocities = np.asarray([particle.state.velocity for particle in particles], dtype=np.float64)

        mean_position = np.average(positions, axis=0, weights=weights)
        mean_bbox = np.average(bboxes, axis=0, weights=weights)
        mean_mass = float(np.average(masses, weights=weights))
        mean_velocity = np.average(velocities, axis=0, weights=weights)

        return TrackEstimate(
            track_id=self.track_id,
            label=self.label,
            position=(
                float(mean_position[0]),
                float(mean_position[1]),
                float(mean_position[2]),
            ),
            bounding_box=(
                float(mean_bbox[0]),
                float(mean_bbox[1]),
                float(mean_bbox[2]),
            ),
            mass=mean_mass,
            velocity=(
                float(mean_velocity[0]),
                float(mean_velocity[1]),
                float(mean_velocity[2]),
            ),
            particle_count=len(particles),
        )


class MultiObjectParticleFilter:
    """Track multiple object instances with one independent particle set per track."""

    def __init__(
        self,
        *,
        motion_model: MotionModel | None = None,
        cost_function: CostFunction | None = None,
        config: ParticleFilterConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._config = config or ParticleFilterConfig()
        self._rng = rng or random.Random()
        self._point_cloud_selection = create_point_cloud_ops(
            requested_backend=self._config.point_cloud_backend,
            require_backend=self._config.require_point_cloud_backend,
        )
        self._point_cloud_ops = self._point_cloud_selection.backend
        self._motion_model = motion_model or ConstantVelocityMotionModel()
        self._cost_function = cost_function or IoUPointCloudCost(
            point_cloud_ops=self._point_cloud_ops
        )
        self._resampler = build_resampler(self._config.resampler)

        self._tracks: dict[str, SingleObjectParticleFilter] = {}
        self._label_counts: dict[str, int] = {}
        self._track_label_counts: dict[str, dict[str, int]] = {}
        self._particle_counter = 0
        self._assignment_workers = _normalize_worker_count(self._config.assignment_workers)
        self._particle_score_workers = _normalize_worker_count(self._config.particle_score_workers)
        if self._point_cloud_ops.backend_name != "numpy":
            self._particle_score_workers = 1

    @property
    def tracks(self) -> Mapping[str, SingleObjectParticleFilter]:
        return self._tracks

    @property
    def point_cloud_backend_name(self) -> str:
        return self._point_cloud_ops.backend_name

    @property
    def stitched_points_by_track(self) -> Mapping[str, tuple[tuple[float, float, float], ...]]:
        return {track_id: track.stitched_points for track_id, track in self._tracks.items()}

    @property
    def track_label_counts_by_track(self) -> Mapping[str, Mapping[str, int]]:
        return {
            track_id: dict(label_counts)
            for track_id, label_counts in self._track_label_counts.items()
        }

    def step(self, frame: FrameContext) -> FilterResult:
        if len(self._tracks) > 1 and self._assignment_workers > 1:
            predict_workers = min(self._assignment_workers, len(self._tracks))
            with ThreadPoolExecutor(max_workers=predict_workers) as executor:
                list(executor.map(lambda track: track.predict(frame), self._tracks.values()))
        else:
            for track in self._tracks.values():
                track.predict(frame)

        existing_track_ids = set(self._tracks.keys())
        assigned = self._associate_detections(frame)
        assigned_existing_count = sum(
            1 for track_id in assigned.keys() if track_id in existing_track_ids
        )

        assigned_detection_indices = {
            assignment.detection_index for assignment in assigned.values()
        }
        spawn_projection_cache = self._project_point_cloud_for_spawn(frame)
        spawned_count = 0
        spawned_from_cloud_count = 0
        spawned_fallback_count = 0
        spawned_cloud_point_matches = 0
        for detection_index, detection in enumerate(frame.detections):
            if detection_index in assigned_detection_indices:
                continue
            track_id, spawned_from_cloud, spawn_match_count = self._spawn_track(
                detection,
                frame,
                spawn_projection_cache,
            )
            assigned[track_id] = _AssignmentCandidate(
                track_id=track_id,
                detection_index=detection_index,
                mean_iou=1.0,
                support_ratio=1.0,
                support_count=self._config.particles_per_track,
                label_match=True,
            )
            spawned_count += 1
            if spawned_from_cloud:
                spawned_from_cloud_count += 1
                spawned_cloud_point_matches += int(spawn_match_count)
            else:
                spawned_fallback_count += 1

        score_executor: Executor | None = None
        if self._particle_score_workers > 1:
            score_executor = ThreadPoolExecutor(max_workers=self._particle_score_workers)
        removed_count = 0
        updated_with_detection_count = 0
        updated_without_detection_count = 0
        try:
            for track_id in list(self._tracks.keys()):
                track = self._tracks[track_id]
                assignment = assigned.get(track_id)
                detection = (
                    frame.detections[assignment.detection_index]
                    if assignment is not None
                    else None
                )
                track.update(frame, detection, score_executor=score_executor)
                if detection is not None:
                    updated_with_detection_count += 1
                else:
                    updated_without_detection_count += 1
                if assignment is not None and detection is not None:
                    self._record_track_label_observation(
                        track_id,
                        detection.label,
                        support_count=max(1, assignment.support_count),
                    )
                if track.missed_frames > self._config.max_missed_frames:
                    del self._tracks[track_id]
                    self._track_label_counts.pop(track_id, None)
                    removed_count += 1
        finally:
            if score_executor is not None:
                score_executor.shutdown(wait=True)

        estimates = {track_id: track.estimate() for track_id, track in self._tracks.items()}
        particles_by_track = {
            track_id: tuple(track.particles) for track_id, track in self._tracks.items()
        }
        stitched_points_by_track = {
            track_id: track.stitched_points for track_id, track in self._tracks.items()
        }
        stitched_track_count = sum(
            1 for points in stitched_points_by_track.values() if len(points) > 0
        )
        stitched_point_count = sum(len(points) for points in stitched_points_by_track.values())

        return FilterResult(
            frame_index=frame.frame_index,
            estimates=estimates,
            particles_by_track=particles_by_track,
            stitched_points_by_track=stitched_points_by_track,
            track_label_counts_by_track=self.track_label_counts_by_track,
            diagnostics={
                "detections": int(len(frame.detections)),
                "tracks_before_update": int(len(existing_track_ids)),
                "tracks_after_update": int(len(self._tracks)),
                "matched_existing_tracks": int(assigned_existing_count),
                "spawned_tracks": int(spawned_count),
                "spawned_from_cloud_tracks": int(spawned_from_cloud_count),
                "spawned_fallback_tracks": int(spawned_fallback_count),
                "spawned_cloud_point_matches": int(spawned_cloud_point_matches),
                "removed_tracks": int(removed_count),
                "updated_with_detection": int(updated_with_detection_count),
                "updated_without_detection": int(updated_without_detection_count),
                "stitched_track_count": int(stitched_track_count),
                "stitched_point_count": int(stitched_point_count),
            },
        )

    def _associate_detections(
        self,
        frame: FrameContext,
    ) -> dict[str, "_AssignmentCandidate"]:
        candidates = self._build_assignment_candidates(frame)
        assigned: dict[str, _AssignmentCandidate] = {}
        used_tracks: set[str] = set()
        used_detection_indices: set[int] = set()

        label_consistent = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.label_match
                and candidate.mean_iou >= self._config.min_assignment_iou
            ),
            key=lambda candidate: (
                candidate.mean_iou,
                candidate.support_ratio,
                candidate.support_count,
            ),
            reverse=True,
        )
        for candidate in label_consistent:
            if candidate.track_id in used_tracks:
                continue
            if candidate.detection_index in used_detection_indices:
                continue
            assigned[candidate.track_id] = candidate
            used_tracks.add(candidate.track_id)
            used_detection_indices.add(candidate.detection_index)

        relabel_candidates: list[_AssignmentCandidate] = []
        for detection_index, detection in enumerate(frame.detections):
            if detection_index in used_detection_indices:
                continue
            unmatched = sorted(
                (
                    candidate
                    for candidate in candidates
                    if candidate.detection_index == detection_index
                    and candidate.track_id not in used_tracks
                ),
                key=lambda candidate: (
                    candidate.mean_iou,
                    candidate.support_ratio,
                    candidate.support_count,
                ),
                reverse=True,
            )
            if not unmatched:
                continue

            best = unmatched[0]
            if best.label_match:
                continue

            second_best_score = unmatched[1].mean_iou if len(unmatched) > 1 else 0.0
            if best.mean_iou < self._config.min_label_flicker_assignment_iou:
                continue
            if (
                best.support_ratio
                < self._config.min_label_flicker_particle_support_ratio
            ):
                continue
            if (
                best.mean_iou - second_best_score
                < self._config.label_flicker_iou_margin
            ):
                continue
            if detection.confidence <= 0.0:
                continue
            relabel_candidates.append(best)

        relabel_candidates.sort(
            key=lambda candidate: (
                candidate.mean_iou,
                candidate.support_ratio,
                candidate.support_count,
            ),
            reverse=True,
        )
        for candidate in relabel_candidates:
            if candidate.track_id in used_tracks:
                continue
            if candidate.detection_index in used_detection_indices:
                continue
            assigned[candidate.track_id] = candidate
            used_tracks.add(candidate.track_id)
            used_detection_indices.add(candidate.detection_index)

        return assigned

    def _build_assignment_candidates(self, frame: FrameContext) -> list["_AssignmentCandidate"]:
        track_items = list(self._tracks.items())
        detections = list(frame.detections)

        if not track_items or not detections:
            return []

        def _candidates_for_track(
            item: tuple[str, SingleObjectParticleFilter],
        ) -> list[_AssignmentCandidate]:
            track_id, track = item
            local: list[_AssignmentCandidate] = []
            for detection_index, detection in enumerate(detections):
                mean_iou, support_ratio, support_count = track.detection_match_metrics(
                    frame,
                    detection,
                    support_iou_threshold=self._config.particle_support_iou_threshold,
                )
                if mean_iou <= 0.0:
                    continue
                local.append(
                    _AssignmentCandidate(
                        track_id=track_id,
                        detection_index=detection_index,
                        mean_iou=mean_iou,
                        support_ratio=support_ratio,
                        support_count=support_count,
                        label_match=(track.label == detection.label),
                    )
                )
            return local

        if len(track_items) > 1 and self._assignment_workers > 1:
            worker_count = min(self._assignment_workers, len(track_items))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                chunks = list(executor.map(_candidates_for_track, track_items))
            candidates: list[_AssignmentCandidate] = []
            for chunk in chunks:
                candidates.extend(chunk)
            return candidates

        candidates: list[_AssignmentCandidate] = []
        for item in track_items:
            candidates.extend(_candidates_for_track(item))
        return candidates

    def _project_point_cloud_for_spawn(
        self,
        frame: FrameContext,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if not frame.point_cloud_points:
            return None
        points_world = np.asarray(frame.point_cloud_points, dtype=np.float64)
        if points_world.ndim != 2 or points_world.shape[1] != 3:
            return None
        if points_world.size == 0:
            return None

        rotation = np.asarray(frame.camera_pose.rotation, dtype=np.float64)
        translation = np.asarray(frame.camera_pose.translation, dtype=np.float64)
        points_camera = points_world @ rotation.T + translation[None, :]
        z = points_camera[:, 2]
        finite_mask = np.isfinite(points_camera).all(axis=1)
        visible_mask = finite_mask & (z > 1e-6)
        if not np.any(visible_mask):
            return None

        world_visible = points_world[visible_mask]
        camera_visible = points_camera[visible_mask]
        z_visible = camera_visible[:, 2]
        x = camera_visible[:, 0] / z_visible
        y = camera_visible[:, 1] / z_visible

        coeffs = tuple(frame.camera_intrinsics.distortion_coeffs)
        if coeffs:
            k1 = float(coeffs[0]) if len(coeffs) > 0 else 0.0
            k2 = float(coeffs[1]) if len(coeffs) > 1 else 0.0
            p1 = float(coeffs[2]) if len(coeffs) > 2 else 0.0
            p2 = float(coeffs[3]) if len(coeffs) > 3 else 0.0
            k3 = float(coeffs[4]) if len(coeffs) > 4 else 0.0

            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2
            radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
            x_tan = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            y_tan = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            x = x * radial + x_tan
            y = y * radial + y_tan

        u = frame.camera_intrinsics.fx_px * x + frame.camera_intrinsics.cx_px
        v = frame.camera_intrinsics.fy_px * y + frame.camera_intrinsics.cy_px
        uv_finite_mask = np.isfinite(u) & np.isfinite(v) & np.isfinite(z_visible)
        if not np.any(uv_finite_mask):
            return None

        return (
            world_visible[uv_finite_mask],
            u[uv_finite_mask],
            v[uv_finite_mask],
            z_visible[uv_finite_mask],
        )

    def _points_projected_inside_detection(
        self,
        detection: Detection2D,
        projected_cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if projected_cache is None:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )
        world_visible, u, v, z = projected_cache
        inside = (
            (u >= detection.aabb.x_min)
            & (u <= detection.aabb.x_max)
            & (v >= detection.aabb.y_min)
            & (v <= detection.aabb.y_max)
        )
        if not np.any(inside):
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )
        return (world_visible[inside], z[inside])

    def _spawn_state_from_point_cloud_match(
        self,
        detection: Detection2D,
        frame: FrameContext,
        matched_points_world: np.ndarray,
        matched_depths: np.ndarray,
    ) -> tuple[tuple[float, float, float], float, tuple[float, float, float]] | None:
        if matched_points_world.size == 0 or matched_depths.size == 0:
            return None

        points_world = np.asarray(matched_points_world, dtype=np.float64)
        depths = np.asarray(matched_depths, dtype=np.float64)
        if points_world.ndim != 2 or points_world.shape[1] != 3 or depths.size == 0:
            return None

        trim = float(np.clip(self._config.spawn_depth_quantile_trim, 0.0, 0.49))
        if len(depths) >= 8 and trim > 0.0:
            lower = float(np.quantile(depths, trim))
            upper = float(np.quantile(depths, 1.0 - trim))
            keep_mask = (depths >= lower) & (depths <= upper)
            if int(np.sum(keep_mask)) >= 3:
                points_world = points_world[keep_mask]
                depths = depths[keep_mask]

        depth_m = float(np.median(depths))
        if depth_m <= 1e-6:
            return None

        center = np.mean(points_world, axis=0)
        extents = np.max(points_world, axis=0) - np.min(points_world, axis=0)

        width_guess = max(
            1e-3,
            detection.aabb.width * depth_m / max(frame.camera_intrinsics.fx_px, 1e-6),
        )
        height_guess = max(
            1e-3,
            detection.aabb.height * depth_m / max(frame.camera_intrinsics.fy_px, 1e-6),
        )
        depth_guess = max(1e-3, (width_guess + height_guess) * 0.5)

        bbox = (
            max(1e-3, float(max(extents[0], width_guess * 0.35))),
            max(1e-3, float(max(extents[1], height_guess * 0.35))),
            max(1e-3, float(max(extents[2], depth_guess * 0.35))),
        )
        return (
            (float(center[0]), float(center[1]), float(center[2])),
            depth_m,
            bbox,
        )

    def _spawn_track(
        self,
        detection: Detection2D,
        frame: FrameContext,
        projected_cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    ) -> tuple[str, bool, int]:
        label_index = self._label_counts.get(detection.label, 0) + 1
        self._label_counts[detection.label] = label_index
        track_id = f"{detection.label}-{label_index}"

        matched_points_world, matched_depths = self._points_projected_inside_detection(
            detection,
            projected_cache,
        )
        spawn_from_cloud = False
        spawn_state = self._spawn_state_from_point_cloud_match(
            detection,
            frame,
            matched_points_world,
            matched_depths,
        )
        if spawn_state is not None:
            center_world, _depth_m, size_xyz = spawn_state
            width_m, height_m, depth_size_m = size_xyz
            spawn_from_cloud = True
        else:
            center_px = detection.aabb.center
            depth_m = self._config.initial_depth_m
            center_world = back_project_pixel(
                center_px,
                depth_m,
                frame.camera_intrinsics,
                frame.camera_pose,
            )
            width_m = max(1e-3, detection.aabb.width * depth_m / frame.camera_intrinsics.fx_px)
            height_m = max(1e-3, detection.aabb.height * depth_m / frame.camera_intrinsics.fy_px)
            depth_size_m = max(1e-3, (width_m + height_m) * 0.5)

        track_rng = random.Random(self._rng.getrandbits(63))
        particles: list[Particle] = []
        for _ in range(self._config.particles_per_track):
            position = (
                center_world[0] + track_rng.gauss(0.0, self._config.initial_position_std_m),
                center_world[1] + track_rng.gauss(0.0, self._config.initial_position_std_m),
                center_world[2] + track_rng.gauss(0.0, self._config.initial_position_std_m),
            )
            velocity = (
                track_rng.gauss(0.0, self._config.initial_velocity_std_mps),
                track_rng.gauss(0.0, self._config.initial_velocity_std_mps),
                track_rng.gauss(0.0, self._config.initial_velocity_std_mps),
            )
            bbox = (
                max(1e-3, width_m + track_rng.gauss(0.0, self._config.initial_bbox_std_m)),
                max(1e-3, height_m + track_rng.gauss(0.0, self._config.initial_bbox_std_m)),
                max(1e-3, depth_size_m + track_rng.gauss(0.0, self._config.initial_bbox_std_m)),
            )
            mass = max(
                1e-6,
                self._config.initial_mass_kg + track_rng.gauss(0.0, self._config.initial_mass_std_kg),
            )

            particles.append(
                Particle(
                    particle_id=self._next_particle_id(),
                    track_id=track_id,
                    label=detection.label,
                    state=ParticleState(
                        position=position,
                        bounding_box=bbox,
                        mass=mass,
                        velocity=velocity,
                    ),
                    weight=1.0 / self._config.particles_per_track,
                    age=0,
                )
            )

        self._tracks[track_id] = SingleObjectParticleFilter(
            track_id=track_id,
            label=detection.label,
            particles=particles,
            motion_model=self._motion_model,
            cost_function=self._cost_function,
            resampler=self._resampler,
            rng=track_rng,
            stitched_voxel_size_m=self._config.stitched_voxel_size_m,
            max_stitched_points=self._config.max_stitched_points_per_track,
            point_cloud_ops=self._point_cloud_ops,
            stitched_icp_max_iterations=self._config.stitched_icp_max_iterations,
            stitched_icp_tolerance_m=self._config.stitched_icp_tolerance_m,
            stitched_icp_max_correspondence_distance_m=self._config.stitched_icp_max_correspondence_distance_m,
            stitched_icp_min_correspondence_ratio=self._config.stitched_icp_min_correspondence_ratio,
            stitched_icp_max_mean_error_m=self._config.stitched_icp_max_mean_error_m,
        )
        self._track_label_counts[track_id] = {}
        return (track_id, spawn_from_cloud, int(len(matched_depths)))

    def _record_track_label_observation(
        self,
        track_id: str,
        detection_label: str,
        *,
        support_count: int,
    ) -> None:
        label_counts = self._track_label_counts.setdefault(track_id, {})
        label_counts[detection_label] = label_counts.get(detection_label, 0) + max(
            1,
            int(support_count),
        )

    def _next_particle_id(self) -> str:
        value = self._particle_counter
        self._particle_counter += 1
        return f"p-{value}"


@dataclass(frozen=True)
class _AssignmentCandidate:
    track_id: str
    detection_index: int
    mean_iou: float
    support_ratio: float
    support_count: int
    label_match: bool
