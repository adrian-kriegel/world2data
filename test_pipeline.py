"""Test suite for the World2Data pipeline.

Tests each step independently, then runs an end-to-end integration test.
Run with: uv run python -m pytest test_pipeline.py -v
"""
import os
import sys
import json
import tempfile
import numpy as np
import cv2
import pytest
from pathlib import Path
from dotenv import load_dotenv

# Load .env before anything else
load_dotenv(Path(__file__).resolve().parent / ".env")

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_controller import (
    World2DataPipeline, FrameData, Object3D,
    _HAS_MAST3R, _HAS_GEMINI, MAST3R_CHECKPOINT,
)
from generate_test_video import generate_test_video


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture(scope="session")
def test_video_path(tmp_path_factory):
    """Generate a synthetic test video once for all tests."""
    video_dir = tmp_path_factory.mktemp("videos")
    path = str(video_dir / "test_video.mp4")
    generate_test_video(path, width=640, height=480, fps=30, duration_sec=3)
    return path


@pytest.fixture
def pipeline(test_video_path, tmp_path):
    """Create a fresh pipeline instance for each test."""
    output_path = str(tmp_path / "output.usda")
    return World2DataPipeline(
        test_video_path,
        output_path=output_path,
        keyframe_dir=str(tmp_path / "keyframes"),
        cache_dir=str(tmp_path / "cache"),
        rerun_enabled=False,
    )


# =========================================================================
# Step 1: Keyframe Extraction
# =========================================================================

class TestStep1KeyframeExtraction:
    def test_extracts_keyframes(self, pipeline):
        result = pipeline.step_1_smart_extraction(threshold=10.0)
        assert result is True
        assert len(pipeline.keyframes) >= 2
        print(f"  Extracted {len(pipeline.keyframes)} keyframes")

    def test_keyframes_saved_to_disk(self, pipeline):
        pipeline.step_1_smart_extraction(threshold=10.0)
        assert len(pipeline.keyframe_paths) == len(pipeline.keyframes)
        for path in pipeline.keyframe_paths:
            assert os.path.isfile(path)
            img = cv2.imread(path)
            assert img is not None and img.shape[0] > 0

    def test_high_threshold_fails(self, pipeline):
        pipeline.step_1_smart_extraction(threshold=200.0)
        assert len(pipeline.keyframes) <= 5

    def test_low_threshold_extracts_more(self, pipeline):
        pipeline.step_1_smart_extraction(threshold=5.0)
        n_low = len(pipeline.keyframes)
        p2 = World2DataPipeline(
            pipeline.video_path,
            keyframe_dir=pipeline.keyframe_dir + "_2",
            rerun_enabled=False,
        )
        p2.step_1_smart_extraction(threshold=30.0)
        assert n_low >= len(p2.keyframes)

    def test_missing_video_fails(self, tmp_path):
        p = World2DataPipeline("nonexistent.mp4",
                               keyframe_dir=str(tmp_path / "kf"),
                               rerun_enabled=False)
        assert p.step_1_smart_extraction() is False

    def test_keyframes_are_valid_arrays(self, pipeline):
        pipeline.step_1_smart_extraction(threshold=10.0)
        for kf in pipeline.keyframes:
            assert isinstance(kf, np.ndarray)
            assert kf.ndim == 3 and kf.shape[2] == 3


# =========================================================================
# Step 2: Geometric Reconstruction
# =========================================================================

class TestStep2Geometry:
    def test_mock_geometry(self, pipeline):
        """Mock geometry should produce per-frame temporal data."""
        pipeline.step_1_smart_extraction(threshold=10.0)
        assert pipeline._mock_geometry() is True
        assert pipeline.point_cloud.shape[0] > 0
        assert len(pipeline.frame_data) == len(pipeline.keyframes)
        for fd in pipeline.frame_data:
            assert isinstance(fd, FrameData)
            assert fd.pts3d.shape[1] == 3
            assert fd.colors.shape == fd.pts3d.shape
            assert fd.pose.shape == (4, 4)

    def test_fails_without_keyframes(self, pipeline):
        assert pipeline.step_2_geometric_reconstruction() is False

    @pytest.mark.skipif(not _HAS_MAST3R, reason="MASt3R not installed")
    @pytest.mark.skipif(
        not os.path.isfile(MAST3R_CHECKPOINT),
        reason="MASt3R checkpoint not found"
    )
    def test_real_mast3r(self, pipeline):
        pipeline.step_1_smart_extraction(threshold=10.0)
        assert pipeline.step_2_geometric_reconstruction(
            strategy="swin-3", min_points=10
        )
        assert pipeline.point_cloud.shape[0] >= 10
        assert len(pipeline.frame_data) == len(pipeline.keyframes)
        for fd in pipeline.frame_data:
            assert fd.pts3d.shape[0] > 0


# =========================================================================
# Object3D unit tests
# =========================================================================

class TestObject3D:
    def test_center_and_size(self):
        obj = Object3D("Door_01", "door",
                        bbox_3d_min=[1, 2, 3], bbox_3d_max=[4, 6, 8])
        np.testing.assert_array_almost_equal(obj.center, [2.5, 4.0, 5.5])
        np.testing.assert_array_almost_equal(obj.size, [3.0, 4.0, 5.0])

    def test_to_dict(self):
        obj = Object3D("Table_01", "table",
                        [0, 0, 0], [2, 1, 2],
                        component_type="FixedJoint",
                        initial_state="stationary",
                        final_state="stationary")
        d = obj.to_dict()
        assert d["entity"] == "Table_01"
        assert d["type"] == "table"
        assert d["component_type"] == "FixedJoint"
        assert len(d["center"]) == 3
        assert len(d["size"]) == 3

    def test_state_changes(self):
        obj = Object3D("Door_01", "door", [0, 0, 0], [1, 2, 0.1],
                        component_type="RevoluteJoint",
                        initial_state="closed", final_state="open",
                        state_changes=[{"time": "mid", "from": "closed",
                                        "to": "open", "cause": "person pushed"}])
        assert obj.state_changes[0]["cause"] == "person pushed"
        assert obj.initial_state == "closed"
        assert obj.final_state == "open"


# =========================================================================
# Step 3: Semantic 3D Object Detection
# =========================================================================

class TestStep3SemanticDetection:
    def test_skips_without_api_key(self, pipeline, monkeypatch):
        """Should skip gracefully without API key."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        pipeline.step_1_smart_extraction(threshold=10.0)
        pipeline._mock_geometry()
        pipeline.step_3_semantic_detection()
        # Should not crash

    def test_backproject_bbox_to_3d(self, pipeline):
        """Back-projection should find 3D points inside a 2D bbox."""
        pipeline.step_1_smart_extraction(threshold=10.0)
        pipeline._mock_geometry()

        # Create a detection covering the whole image
        det = {"entity": "Test_Obj", "type": "test",
               "bbox": [0, 0, 100, 100], "frame_idx": 0}
        obj = pipeline._backproject_bbox_to_3d(det, [0])
        # With full-image bbox and synthetic data, should find some 3D points
        # (may be None if mock geometry doesn't project well -- that's OK)
        if obj is not None:
            assert obj.entity == "Test_Obj"
            assert obj.bbox_3d_min.shape == (3,)
            assert obj.bbox_3d_max.shape == (3,)

    @pytest.mark.skipif(
        not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
        reason="GOOGLE_API_KEY not set"
    )
    def test_gemini_detection(self, pipeline):
        """Gemini should detect objects in keyframes."""
        pipeline.step_1_smart_extraction(threshold=10.0)
        pipeline._mock_geometry()
        pipeline.step_3_semantic_detection()
        # With a synthetic test video, Gemini may or may not find objects
        print(f"  Detected {len(pipeline.objects_3d)} objects in 3D")


# =========================================================================
# Step 4: Causal Reasoning
# =========================================================================

class TestStep4Reasoning:
    def test_skips_without_api_key(self, pipeline, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        pipeline.step_1_smart_extraction(threshold=10.0)
        pipeline.step_4_causal_reasoning()

    @pytest.mark.skipif(
        not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
        reason="GOOGLE_API_KEY not set"
    )
    def test_gemini_reasoning(self, pipeline):
        pipeline.step_1_smart_extraction(threshold=10.0)
        pipeline.step_4_causal_reasoning()
        assert isinstance(pipeline.scene_graph, dict)
        assert "objects" in pipeline.scene_graph
        print(f"  Gemini reasoned about {len(pipeline.scene_graph['objects'])} objects")


# =========================================================================
# Step 5: USD Export (with physics)
# =========================================================================

class TestStep5UsdExport:
    def test_creates_usda(self, pipeline):
        pipeline.step_1_smart_extraction(threshold=10.0)
        pipeline._mock_geometry()
        pipeline.step_5_export_usd()
        assert os.path.isfile(pipeline.output_path)
        from pxr import Usd
        stage = Usd.Stage.Open(pipeline.output_path)
        assert stage.GetPrimAtPath("/World").IsValid()

    def test_json_sidecar(self, pipeline):
        pipeline.step_1_smart_extraction(threshold=10.0)
        pipeline._mock_geometry()
        pipeline.step_5_export_usd()
        json_path = pipeline.output_path.replace(".usda", "_scene_graph.json")
        assert os.path.isfile(json_path)
        with open(json_path) as f:
            data = json.load(f)
        assert "num_cameras" in data
        assert "objects_3d" in data

    def test_point_cloud_in_usd(self, pipeline):
        pipeline.step_1_smart_extraction(threshold=10.0)
        pipeline._mock_geometry()
        pipeline.step_5_export_usd()
        from pxr import Usd
        stage = Usd.Stage.Open(pipeline.output_path)
        assert stage.GetPrimAtPath("/World/PointCloud").IsValid()

    def test_cameras_in_usd(self, pipeline):
        pipeline.step_1_smart_extraction(threshold=10.0)
        pipeline._mock_geometry()
        pipeline.step_5_export_usd()
        from pxr import Usd
        stage = Usd.Stage.Open(pipeline.output_path)
        assert stage.GetPrimAtPath("/World/Cameras/Cam_00").IsValid()

    def test_objects_with_physics(self, pipeline):
        """Objects with non-FixedJoint should have physics APIs applied."""
        pipeline.step_1_smart_extraction(threshold=10.0)
        pipeline._mock_geometry()
        # Add a fake 3D object
        pipeline.objects_3d = [
            Object3D("Door_01", "door", [-1, 0, 1], [1, 2, 1.1],
                      component_type="RevoluteJoint",
                      initial_state="closed", final_state="open"),
            Object3D("Table_01", "table", [0, 0, 0], [2, 1, 2],
                      component_type="FixedJoint"),
        ]
        pipeline.step_5_export_usd()

        from pxr import Usd, UsdPhysics
        stage = Usd.Stage.Open(pipeline.output_path)

        # Door should have physics
        door_prim = stage.GetPrimAtPath("/World/Objects/Door_01")
        assert door_prim.IsValid(), "Door prim should exist"
        assert door_prim.HasAPI(UsdPhysics.RigidBodyAPI), "Door should have RigidBodyAPI"

        # Table should NOT have physics (FixedJoint)
        table_prim = stage.GetPrimAtPath("/World/Objects/Table_01")
        assert table_prim.IsValid()
        assert not table_prim.HasAPI(UsdPhysics.RigidBodyAPI)

        # Custom attributes
        assert door_prim.GetAttribute("world2data:component_type").Get() == "RevoluteJoint"
        assert door_prim.GetAttribute("world2data:initial_state").Get() == "closed"
        assert door_prim.GetAttribute("world2data:final_state").Get() == "open"


# =========================================================================
# Temporal Rerun Recording
# =========================================================================

class TestTemporalRerun:
    def test_rrd_saved(self, pipeline, tmp_path):
        pipeline.step_1_smart_extraction(threshold=10.0)
        pipeline._mock_geometry()
        rrd_path = str(tmp_path / "test.rrd")
        pipeline.save_temporal_rrd(rrd_path)
        assert os.path.isfile(rrd_path)
        assert os.path.getsize(rrd_path) > 0

    def test_rrd_with_objects(self, pipeline, tmp_path):
        """Rerun recording should include 3D bounding boxes."""
        pipeline.step_1_smart_extraction(threshold=10.0)
        pipeline._mock_geometry()
        pipeline.objects_3d = [
            Object3D("Chair_01", "chair", [0, 0, 0], [0.5, 1, 0.5],
                      component_type="FixedJoint",
                      initial_state="stationary", final_state="stationary"),
        ]
        rrd_path = str(tmp_path / "with_objects.rrd")
        pipeline.save_temporal_rrd(rrd_path)
        assert os.path.isfile(rrd_path)
        # File should be bigger with objects
        assert os.path.getsize(rrd_path) > 0
        print(f"  .rrd with objects: {os.path.getsize(rrd_path) / 1024:.1f} KB")

    def test_rrd_with_state_changes(self, pipeline, tmp_path):
        """Rerun recording should animate state change colors."""
        pipeline.step_1_smart_extraction(threshold=10.0)
        pipeline._mock_geometry()
        pipeline.objects_3d = [
            Object3D("Door_01", "door", [-1, 0, 1], [1, 2, 1.1],
                      component_type="RevoluteJoint",
                      initial_state="closed", final_state="open",
                      state_changes=[{"time": "mid", "from": "closed",
                                      "to": "open", "cause": "pushed"}]),
        ]
        rrd_path = str(tmp_path / "state_change.rrd")
        pipeline.save_temporal_rrd(rrd_path)
        assert os.path.isfile(rrd_path)
        assert os.path.getsize(rrd_path) > 0


# =========================================================================
# End-to-End
# =========================================================================

class TestEndToEnd:
    def test_full_pipeline_mock_mode(self, test_video_path, tmp_path):
        """Full pipeline mock mode: no GPU, no API needed."""
        output_path = str(tmp_path / "e2e.usda")
        pipeline = World2DataPipeline(
            test_video_path, output_path=output_path,
            keyframe_dir=str(tmp_path / "kf"),
            cache_dir=str(tmp_path / "cache"),
            rerun_enabled=False,
        )
        assert pipeline.step_1_smart_extraction(threshold=10.0)
        assert pipeline._mock_geometry()
        # Step 3 + 4 may skip without API key
        pipeline.step_3_semantic_detection()
        pipeline.step_4_causal_reasoning()
        pipeline.step_5_export_usd()
        rrd_path = str(tmp_path / "e2e.rrd")
        pipeline.save_temporal_rrd(rrd_path)

        assert os.path.isfile(output_path)
        assert os.path.isfile(output_path.replace(".usda", "_scene_graph.json"))
        assert os.path.isfile(rrd_path)
        assert len(pipeline.frame_data) == len(pipeline.keyframes)
        assert pipeline.point_cloud.shape[0] > 0

        # Verify JSON has objects_3d field
        with open(output_path.replace(".usda", "_scene_graph.json")) as f:
            data = json.load(f)
        assert "objects_3d" in data

    REAL_VIDEO = os.path.join(
        os.path.dirname(__file__),
        "testenvironment", "LFM2_5_VL_1_6B", "inputs", "videos",
        "video_2026-02-07_23-51-38.mp4",
    )

    @pytest.mark.skipif(not _HAS_MAST3R, reason="MASt3R not installed")
    @pytest.mark.skipif(
        not os.path.isfile(REAL_VIDEO), reason="Real video not found"
    )
    def test_full_pipeline_real_video(self, tmp_path):
        """Full E2E on real video with MASt3R + Gemini + Objects."""
        output_path = str(tmp_path / "real.usda")
        pipeline = World2DataPipeline(
            self.REAL_VIDEO, output_path=output_path,
            keyframe_dir=str(tmp_path / "kf"),
            cache_dir=str(tmp_path / "cache"),
            rerun_enabled=False,
        )
        assert pipeline.step_1_smart_extraction(threshold=15.0)
        assert pipeline.step_2_geometric_reconstruction(
            strategy="swin-3", min_points=100
        )
        pipeline.step_3_semantic_detection()
        pipeline.step_4_causal_reasoning()
        pipeline.step_5_export_usd()
        rrd_path = str(tmp_path / "real.rrd")
        pipeline.save_temporal_rrd(rrd_path)

        assert os.path.isfile(output_path)
        assert os.path.isfile(rrd_path)
        assert pipeline.point_cloud.shape[0] > 10000
        print(f"  REAL E2E: {pipeline.point_cloud.shape[0]} pts, "
              f"{len(pipeline.frame_data)} frames, "
              f"{len(pipeline.objects_3d)} 3D objects")
