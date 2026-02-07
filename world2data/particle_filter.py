from __future__ import annotations

"""Multi-object particle filter with separated motion, cost, and resampling components."""

import random
from dataclasses import dataclass
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

    def score(self, particle: Particle, detection: Detection2D, frame: FrameContext) -> float:
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
    ) -> None:
        self.track_id = track_id
        self.label = label
        self.particles = particles
        self._motion_model = motion_model
        self._cost_function = cost_function
        self._resampler = resampler
        self._rng = rng
        self._missed_frames = 0

    @property
    def missed_frames(self) -> int:
        return self._missed_frames

    def predict(self, frame: FrameContext) -> None:
        self.particles = [
            self._motion_model.propagate(particle, frame, self._rng)
            for particle in self.particles
        ]

    def update(self, frame: FrameContext, detection: Detection2D | None) -> None:
        if not self.particles:
            return

        if detection is None:
            self._missed_frames += 1
            self.particles = _normalize_weights(self.particles)
            return

        self._missed_frames = 0
        scored = [
            particle.evolved(weight=self._cost_function.score(particle, detection, frame))
            for particle in self.particles
        ]
        normalized = _normalize_weights(scored)
        self.particles = self._resampler.resample(normalized, len(self.particles), self._rng)

    def mean_iou_for_detection(self, frame: FrameContext, detection: Detection2D) -> float:
        if not self.particles:
            return 0.0

        particles = _normalize_weights(self.particles)
        score_sum = 0.0
        for particle in particles:
            score_sum += self._cost_function.score(particle, detection, frame) * particle.weight
        return float(score_sum)

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
        self._motion_model = motion_model or ConstantVelocityMotionModel()
        self._cost_function = cost_function or BackProjectionIoUCost()
        self._resampler = build_resampler(self._config.resampler)

        self._tracks: dict[str, SingleObjectParticleFilter] = {}
        self._label_counts: dict[str, int] = {}
        self._particle_counter = 0

    @property
    def tracks(self) -> Mapping[str, SingleObjectParticleFilter]:
        return self._tracks

    def step(self, frame: FrameContext) -> FilterResult:
        for track in self._tracks.values():
            track.predict(frame)

        assigned: dict[str, Detection2D] = {}
        used_tracks: set[str] = set()

        for detection in frame.detections:
            track_id = self._assign_detection_to_track(frame, detection, used_tracks)
            if track_id is None:
                track_id = self._spawn_track(detection, frame)
            assigned[track_id] = detection
            used_tracks.add(track_id)

        for track_id in list(self._tracks.keys()):
            track = self._tracks[track_id]
            track.update(frame, assigned.get(track_id))
            if track.missed_frames > self._config.max_missed_frames:
                del self._tracks[track_id]

        estimates = {track_id: track.estimate() for track_id, track in self._tracks.items()}
        particles_by_track = {
            track_id: tuple(track.particles) for track_id, track in self._tracks.items()
        }

        return FilterResult(
            frame_index=frame.frame_index,
            estimates=estimates,
            particles_by_track=particles_by_track,
        )

    def _assign_detection_to_track(
        self,
        frame: FrameContext,
        detection: Detection2D,
        used_tracks: set[str],
    ) -> str | None:
        best_track_id: str | None = None
        best_score = -1.0

        for track_id, track in self._tracks.items():
            if track_id in used_tracks:
                continue
            if track.label != detection.label:
                continue
            score = track.mean_iou_for_detection(frame, detection)
            if score > best_score:
                best_score = score
                best_track_id = track_id

        if best_track_id is None:
            return None
        if best_score < self._config.min_assignment_iou:
            return None
        return best_track_id

    def _spawn_track(self, detection: Detection2D, frame: FrameContext) -> str:
        label_index = self._label_counts.get(detection.label, 0) + 1
        self._label_counts[detection.label] = label_index
        track_id = f"{detection.label}-{label_index}"

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

        particles: list[Particle] = []
        for _ in range(self._config.particles_per_track):
            position = (
                center_world[0] + self._rng.gauss(0.0, self._config.initial_position_std_m),
                center_world[1] + self._rng.gauss(0.0, self._config.initial_position_std_m),
                center_world[2] + self._rng.gauss(0.0, self._config.initial_position_std_m),
            )
            velocity = (
                self._rng.gauss(0.0, self._config.initial_velocity_std_mps),
                self._rng.gauss(0.0, self._config.initial_velocity_std_mps),
                self._rng.gauss(0.0, self._config.initial_velocity_std_mps),
            )
            bbox = (
                max(1e-3, width_m + self._rng.gauss(0.0, self._config.initial_bbox_std_m)),
                max(1e-3, height_m + self._rng.gauss(0.0, self._config.initial_bbox_std_m)),
                max(1e-3, depth_size_m + self._rng.gauss(0.0, self._config.initial_bbox_std_m)),
            )
            mass = max(1e-6, self._config.initial_mass_kg + self._rng.gauss(0.0, self._config.initial_mass_std_kg))

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
            rng=self._rng,
        )
        return track_id

    def _next_particle_id(self) -> str:
        value = self._particle_counter
        self._particle_counter += 1
        return f"p-{value}"
