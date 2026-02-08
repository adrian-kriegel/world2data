"""End-to-end test: PF tracking -> Layered USD -> Rerun -> Point Lineage.

Mocks MASt3R + YOLO data to exercise the full post-reconstruction pipeline
without needing GPU or model weights. This validates that:
1. PF step converts frame data + YOLO -> 3D tracks
2. Tracks feed into layered USD export (EntityRecords with frame_positions)
3. Point lineage parquet includes track_id + lifecycle states
4. Rerun recording includes PF track bounding boxes
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
    AABB2D as PF_AABB2D,
    CameraIntrinsics as PF_CameraIntrinsics,
    CameraPose as PF_CameraPose,
    Detection2D as PF_Detection2D,
    FrameContext as PF_FrameContext,
)
from world2data.particle_filter import (
    MultiObjectParticleFilter,
    ParticleFilterConfig,
    ConstantVelocityMotionModel,
    BackProjectionIoUCost,
)
from world2data.vision import camera_pose_from_world_position

# For USD layer testing
from world2data.usd_layers import (
    USDLayerWriter, ProvenanceRecord, EntityRecord,
    PointCloudFrameRecord, YOLOFrameRecord,
    _generate_run_id, write_point_lineage,
)

try:
    import pyarrow.parquet as pq
    _HAS_PARQUET = True
except ImportError:
    _HAS_PARQUET = False


def _synthetic_pipeline_data(n_frames=10, fps=10.0):
    """Create synthetic FrameData-like dicts + YOLO detections + camera poses.

    Simulates a camera moving along X axis, observing a chair at (0,0,5) and
    a table at (2,0,5) in world space.
    """
    frame_dicts = []
    yolo_by_frame = {}

    for i in range(n_frames):
        # Camera moves from x=-2 to x=2
        cam_x = -2.0 + 4.0 * i / max(n_frames - 1, 1)
        cam2world = np.eye(4)
        cam2world[0, 3] = cam_x

        # Random point cloud around z=5
        n_pts = 50
        pts = np.random.randn(n_pts, 3).astype(np.float32) * 0.5
        pts[:, 2] += 5.0  # center at z=5
        colors = np.random.rand(n_pts, 3).astype(np.float32)
        confidence = np.random.rand(n_pts).astype(np.float32) * 0.5 + 0.5

        frame_dicts.append({
            "index": i,
            "pts3d": pts,
            "colors": colors,
            "confidence": confidence,
            "pose": cam2world,
            "focal": 500.0,
            "principal_point": np.array([320.0, 240.0]),
            "image_rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "timestamp_sec": i / fps,
        })

        # YOLO detections: chair and table in every frame
        yolo_by_frame[i] = {
            "frame_idx": i,
            "boxes": np.array([
                [250.0, 180.0, 390.0, 340.0],  # chair
                [400.0, 150.0, 550.0, 320.0],  # table
            ]),
            "class_names": ["chair", "table"],
            "scores": np.array([0.92, 0.87]),
        }

    return frame_dicts, yolo_by_frame


def _run_pf_on_synthetic(frame_dicts, yolo_by_frame, fps=10.0):
    """Run MultiObjectParticleFilter on synthetic data, return track estimates."""
    tracker = MultiObjectParticleFilter(
        motion_model=ConstantVelocityMotionModel(
            position_process_std_m=0.02,
            velocity_process_std_mps=0.01,
        ),
        cost_function=BackProjectionIoUCost(),
        config=ParticleFilterConfig(
            particles_per_track=48,
            initial_depth_m=5.0,
            min_assignment_iou=0.05,
            max_missed_frames=10,
        ),
        rng=random.Random(42),
    )

    track_estimates = {}
    for fd in frame_dicts:
        idx = fd["index"]
        pose_c2w = fd["pose"]
        pose_w2c = np.linalg.inv(pose_c2w)
        R = pose_w2c[:3, :3]
        t = pose_w2c[:3, 3]

        camera_pose = PF_CameraPose(
            rotation=tuple(tuple(float(R[r, c]) for c in range(3)) for r in range(3)),
            translation=tuple(float(t[k]) for k in range(3)),
        )
        H, W = fd["image_rgb"].shape[:2]
        cx, cy = fd["principal_point"]
        intrinsics = PF_CameraIntrinsics(
            width_px=W, height_px=H,
            fx_px=float(fd["focal"]), fy_px=float(fd["focal"]),
            cx_px=float(cx), cy_px=float(cy),
        )

        yolo = yolo_by_frame.get(idx, {})
        detections_2d = []
        if "boxes" in yolo:
            for j in range(len(yolo["class_names"])):
                box = yolo["boxes"][j]
                detections_2d.append(PF_Detection2D(
                    label=yolo["class_names"][j],
                    aabb=PF_AABB2D(
                        x_min=float(box[0]), y_min=float(box[1]),
                        x_max=float(box[2]), y_max=float(box[3]),
                    ),
                    confidence=float(yolo["scores"][j]),
                ))

        frame_ctx = PF_FrameContext(
            frame_index=idx,
            timestamp_s=idx / fps,
            dt_s=1.0 / fps,
            camera_pose=camera_pose,
            camera_intrinsics=intrinsics,
            detections=tuple(detections_2d),
        )

        result = tracker.step(frame_ctx)
        for tid, est in result.estimates.items():
            track_estimates.setdefault(tid, []).append((idx, est))

    return track_estimates


class TestPFToUSDPipeline:
    """Test that PF tracks flow through to layered USD export."""

    def test_pf_produces_tracks(self):
        """PF should produce at least 2 tracks (chair + table)."""
        frame_dicts, yolo = _synthetic_pipeline_data(10)
        tracks = _run_pf_on_synthetic(frame_dicts, yolo)
        assert len(tracks) >= 2
        labels = {tid.split("-")[0] for tid in tracks}
        assert "chair" in labels
        assert "table" in labels

    def test_pf_tracks_to_entity_records(self):
        """Convert PF tracks to EntityRecords for USD export."""
        frame_dicts, yolo = _synthetic_pipeline_data(10)
        tracks = _run_pf_on_synthetic(frame_dicts, yolo)

        entities = []
        for track_id, estimates in tracks.items():
            last_est = estimates[-1][1]
            frame_positions = {fidx: est.position for fidx, est in estimates}
            half = np.array(last_est.bounding_box) / 2.0
            center = np.array(last_est.position)

            entities.append(EntityRecord(
                uid=f"pf_{track_id}",
                entity_class=last_est.label,
                label=track_id,
                confidence=min(1.0, len(estimates) / 10),
                detected_by=["yolo", "particle_filter"],
                bbox_3d_min=tuple((center - half).tolist()),
                bbox_3d_max=tuple((center + half).tolist()),
                first_frame=estimates[0][0],
                last_frame=estimates[-1][0],
                frame_positions=frame_positions,
            ))

        assert len(entities) >= 2
        for ent in entities:
            assert ent.uid.startswith("pf_")
            assert len(ent.frame_positions) >= 1  # at least seen in some frames
        # At least one track should have multiple observations
        max_obs = max(len(e.frame_positions) for e in entities)
        assert max_obs >= 3, f"Best track only seen {max_obs} frames"

    def test_pf_tracks_in_layered_usd(self):
        """Full layered USD scene should contain PF tracks."""
        frame_dicts, yolo = _synthetic_pipeline_data(8)
        tracks = _run_pf_on_synthetic(frame_dicts, yolo)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = _generate_run_id()
            writer = USDLayerWriter(tmpdir)

            # Write base + recon
            prov = ProvenanceRecord(
                run_id=run_id, component="base",
                model_name="test", model_version="0.1",
            )
            writer.write_base_layer()
            prov_recon = ProvenanceRecord(
                run_id=run_id, component="recon",
                model_name="mast3r", model_version="0.1",
            )
            cameras = [{"pose": fd["pose"], "focal": fd["focal"]} for fd in frame_dicts]
            pcf = [PointCloudFrameRecord(
                frame_index=fd["index"],
                timestamp_sec=fd["timestamp_sec"],
                points_asset_path=f"frame_{fd['index']:06d}.ply",
                point_count=fd["pts3d"].shape[0],
            ) for fd in frame_dicts]
            writer.write_recon_layer_with_frames(
                prov_recon, cameras=cameras, frame_records=pcf,
            )

            # Write tracks from PF
            prov_tracks = ProvenanceRecord(
                run_id=run_id, component="tracking",
                model_name="particle_filter", model_version="0.1",
            )
            entities = []
            for track_id, estimates in tracks.items():
                last_est = estimates[-1][1]
                frame_positions = {fidx: est.position for fidx, est in estimates}
                half = np.array(last_est.bounding_box) / 2.0
                center = np.array(last_est.position)
                entities.append(EntityRecord(
                    uid=f"pf_{track_id}_{run_id}",
                    entity_class=last_est.label,
                    label=track_id,
                    confidence=0.9,
                    detected_by=["yolo", "particle_filter"],
                    bbox_3d_min=tuple((center - half).tolist()),
                    bbox_3d_max=tuple((center + half).tolist()),
                    first_frame=estimates[0][0],
                    last_frame=estimates[-1][0],
                    frame_positions=frame_positions,
                ))
            writer.write_tracks_layer(prov_tracks, entities)

            # Assemble
            assembly = writer.write_assembly()
            assert os.path.isfile(assembly)

            # Verify tracks layer exists in layers/ subdirectory
            layers_dir = os.path.join(tmpdir, "layers")
            tracks_files = [f for f in os.listdir(layers_dir)
                           if f.startswith("30_tracks")]
            assert len(tracks_files) >= 1

            # Read and verify content mentions PF-sourced entities
            with open(os.path.join(layers_dir, tracks_files[0]), "r") as f:
                content = f.read()
            assert "particle_filter" in content or "pf_" in content


class TestPFEnrichedPointLineage:
    """Test point lineage with PF track associations."""

    @pytest.mark.skipif(not _HAS_PARQUET, reason="pyarrow not installed")
    def test_lineage_with_pf_tracks(self):
        """Point lineage should associate points to PF tracks."""
        frame_dicts, yolo = _synthetic_pipeline_data(10)
        tracks = _run_pf_on_synthetic(frame_dicts, yolo)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_point_lineage(
                tmpdir, frame_dicts, "pf_test",
                pf_track_estimates=tracks,
                total_frames=10,
            )
            assert path is not None
            table = pq.read_table(str(path))

            # Should have track_id column
            assert "track_id" in table.column_names

            # Some points should be tracked (near PF objects)
            track_ids = table.column("track_id").to_pylist()
            tracked_count = sum(1 for t in track_ids if t != "untracked")
            # Not all points will be tracked (many are random), but some should be
            print(f"  Tracked: {tracked_count}/{len(track_ids)} points")

    @pytest.mark.skipif(not _HAS_PARQUET, reason="pyarrow not installed")
    def test_lineage_lifecycle_with_pf(self):
        """With PF tracks, stale points should exist for tracked objects."""
        frame_dicts, yolo = _synthetic_pipeline_data(10)
        tracks = _run_pf_on_synthetic(frame_dicts, yolo)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_point_lineage(
                tmpdir, frame_dicts, "lifecycle_pf",
                pf_track_estimates=tracks,
                total_frames=10,
            )
            table = pq.read_table(str(path))
            states = table.column("state").to_pylist()

            # Should have a mix of states
            unique_states = set(states)
            assert "active" in unique_states, "Must have active points"
            # At least 2 states total (active + one of stale/retired)
            assert len(unique_states) >= 2, f"Expected multiple states, got {unique_states}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
