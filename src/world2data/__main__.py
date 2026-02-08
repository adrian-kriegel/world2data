from __future__ import annotations

import random

from .model import AABB2D, CameraIntrinsics, Detection2D, FrameContext
from .particle_filter import MultiObjectParticleFilter, ParticleFilterConfig
from .vision import camera_pose_from_world_position


def main() -> None:
    intr = CameraIntrinsics(width_px=640, height_px=480, fx_px=600.0, fy_px=600.0, cx_px=320.0, cy_px=240.0)
    detector_output = (
        Detection2D(label="box", aabb=AABB2D(x_min=280.0, y_min=200.0, x_max=340.0, y_max=280.0), confidence=0.95),
    )

    tracker = MultiObjectParticleFilter(
        config=ParticleFilterConfig(particles_per_track=64),
        rng=random.Random(7),
    )
    frame = FrameContext(
        frame_index=0,
        timestamp_s=0.0,
        dt_s=1.0 / 30.0,
        camera_pose=camera_pose_from_world_position((0.0, 0.0, 0.0)),
        camera_intrinsics=intr,
        detections=detector_output,
    )
    result = tracker.step(frame)
    print(f"tracks={len(result.estimates)}")


if __name__ == "__main__":
    main()
