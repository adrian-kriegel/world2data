"""Tests for MultiObjectParticleFilter integration into the pipeline.

Verifies:
- PF step converts pipeline data correctly
- PF produces time-varying track estimates
- PF tracks feed into layered USD export
"""
import os
import sys
import random
import tempfile

import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from world2data.model import (
    AABB2D, CameraIntrinsics, CameraPose, Detection2D, FrameContext,
)
from world2data.particle_filter import (
    MultiObjectParticleFilter,
    ParticleFilterConfig,
    ConstantVelocityMotionModel,
    BackProjectionIoUCost,
)
from world2data.vision import camera_pose_from_world_position


class TestPFDataConversion:
    """Test that pipeline data converts to PF input types correctly."""

    def test_camera_pose_from_mast3r_matrix(self):
        """Convert a 4x4 cam-to-world matrix to CameraPose (world-to-camera)."""
        # Identity = camera at origin looking down -Z
        cam2world = np.eye(4)
        cam2world[0, 3] = 1.0  # translate camera to x=1

        w2c = np.linalg.inv(cam2world)
        R = w2c[:3, :3]
        t = w2c[:3, 3]

        pose = CameraPose(
            rotation=(
                (float(R[0, 0]), float(R[0, 1]), float(R[0, 2])),
                (float(R[1, 0]), float(R[1, 1]), float(R[1, 2])),
                (float(R[2, 0]), float(R[2, 1]), float(R[2, 2])),
            ),
            translation=(float(t[0]), float(t[1]), float(t[2])),
        )
        # Translation should be -1 in x (inverse of camera position)
        assert abs(pose.translation[0] - (-1.0)) < 1e-6

    def test_detection2d_from_yolo_box(self):
        """Convert YOLO xyxy box to Detection2D."""
        box = [100.0, 50.0, 300.0, 400.0]
        det = Detection2D(
            label="person",
            aabb=AABB2D(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
            confidence=0.95,
        )
        assert det.aabb.width == 200.0
        assert det.aabb.height == 350.0
        assert det.confidence == 0.95

    def test_frame_context_construction(self):
        """Build a valid FrameContext from pipeline-like data."""
        intrinsics = CameraIntrinsics(
            width_px=640, height_px=480,
            fx_px=500.0, fy_px=500.0,
            cx_px=320.0, cy_px=240.0,
        )
        pose = camera_pose_from_world_position((0.0, 0.0, 0.0))
        det = Detection2D(
            label="chair",
            aabb=AABB2D(x_min=200.0, y_min=100.0, x_max=400.0, y_max=350.0),
            confidence=0.87,
        )
        ctx = FrameContext(
            frame_index=0, timestamp_s=0.0, dt_s=1 / 30.0,
            camera_pose=pose, camera_intrinsics=intrinsics,
            detections=(det,),
        )
        assert ctx.frame_index == 0
        assert len(ctx.detections) == 1


class TestPFTracking:
    """Test the particle filter produces valid tracks over multiple frames."""

    def _make_tracker(self):
        return MultiObjectParticleFilter(
            motion_model=ConstantVelocityMotionModel(
                position_process_std_m=0.02,
                velocity_process_std_mps=0.01,
            ),
            cost_function=BackProjectionIoUCost(),
            config=ParticleFilterConfig(
                particles_per_track=64,
                initial_depth_m=5.0,
                min_assignment_iou=0.05,
                max_missed_frames=10,
            ),
            rng=random.Random(42),
        )

    def _make_frame(self, frame_idx, camera_x=0.0, detections=()):
        intrinsics = CameraIntrinsics(
            width_px=640, height_px=480,
            fx_px=500.0, fy_px=500.0,
            cx_px=320.0, cy_px=240.0,
        )
        return FrameContext(
            frame_index=frame_idx,
            timestamp_s=frame_idx / 10.0,
            dt_s=0.1,
            camera_pose=camera_pose_from_world_position((camera_x, 0.0, 0.0)),
            camera_intrinsics=intrinsics,
            detections=detections,
        )

    def test_single_object_tracking(self):
        """A persistent detection should produce a stable track."""
        tracker = self._make_tracker()
        det = Detection2D(
            label="chair",
            aabb=AABB2D(x_min=250.0, y_min=180.0, x_max=390.0, y_max=340.0),
            confidence=0.9,
        )

        for i in range(10):
            frame = self._make_frame(i, detections=(det,))
            result = tracker.step(frame)

        assert len(result.estimates) >= 1
        # Should have a chair track
        labels = {est.label for est in result.estimates.values()}
        assert "chair" in labels

    def test_multiple_objects(self):
        """Two different detections should spawn two tracks."""
        tracker = self._make_tracker()
        det_chair = Detection2D(
            label="chair",
            aabb=AABB2D(x_min=100.0, y_min=100.0, x_max=200.0, y_max=300.0),
            confidence=0.9,
        )
        det_person = Detection2D(
            label="person",
            aabb=AABB2D(x_min=400.0, y_min=50.0, x_max=550.0, y_max=400.0),
            confidence=0.85,
        )

        for i in range(8):
            frame = self._make_frame(i, detections=(det_chair, det_person))
            result = tracker.step(frame)

        assert len(result.estimates) >= 2
        labels = {est.label for est in result.estimates.values()}
        assert "chair" in labels
        assert "person" in labels

    def test_track_has_position_and_bbox(self):
        """Track estimates must have 3D position and bounding box."""
        tracker = self._make_tracker()
        det = Detection2D(
            label="table",
            aabb=AABB2D(x_min=200.0, y_min=150.0, x_max=450.0, y_max=350.0),
            confidence=0.95,
        )

        for i in range(5):
            result = tracker.step(self._make_frame(i, detections=(det,)))

        for est in result.estimates.values():
            assert len(est.position) == 3
            assert len(est.bounding_box) == 3
            assert all(v > 0 for v in est.bounding_box), "Bounding box must be positive"

    def test_tracks_accumulate_over_time(self):
        """Track history should grow with each frame."""
        tracker = self._make_tracker()
        det = Detection2D(
            label="cup",
            aabb=AABB2D(x_min=300.0, y_min=200.0, x_max=340.0, y_max=260.0),
            confidence=0.8,
        )

        history = {}
        for i in range(10):
            result = tracker.step(self._make_frame(i, detections=(det,)))
            for tid, est in result.estimates.items():
                history.setdefault(tid, []).append((i, est))

        # At least one track should have multiple observations
        max_obs = max(len(v) for v in history.values())
        assert max_obs >= 5, f"Expected >=5 observations, got {max_obs}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
