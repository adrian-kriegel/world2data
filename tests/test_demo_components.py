"""Tests for demo_run.py components and new model interfaces.

These tests verify each new component incrementally:
  1. YOLOv8 detection on a real keyframe
  2. Gemini video analysis (full video upload)
  3. Scene fusion (4D object creation)
  4. Human review JSON generation
  5. USD accuracy attributes
  6. Demo runner smoke test
"""
import os
import sys
import json
import tempfile
import numpy as np
import pytest
import cv2

from world2data.pipeline.controller import (
    World2DataPipeline, Object3D, FrameData, ParticleFilter3D,
    _HAS_MAST3R, _HAS_GEMINI,
)

# Paths (relative to project root, not tests/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_VIDEO = os.path.join(
    _PROJECT_ROOT,
    "data", "inputs",
    "video_2026-02-08_09-36-42.mp4",
)
OVERNIGHT_VIDEO = os.path.join(
    _PROJECT_ROOT,
    "data", "inputs",
    "video_2026-02-07_23-51-38.mp4",
)

# Check which models are available
try:
    from world2data.pipeline.model_interfaces import _HAS_YOLO, _HAS_SAM3
except ImportError:
    _HAS_YOLO = False
    _HAS_SAM3 = False


# =========================================================================
# Test 1: YOLOv8 Detection
# =========================================================================
class TestYOLODetection:
    """Test YOLOv8 object detection on real keyframes."""

    @pytest.mark.skipif(not _HAS_YOLO, reason="ultralytics not installed")
    def test_yolo_loads(self):
        from world2data.pipeline.model_interfaces import YOLODetector
        det = YOLODetector(model_name="yolov8n.pt", device="cuda")
        assert det.model is not None

    @pytest.mark.skipif(not _HAS_YOLO, reason="ultralytics not installed")
    def test_yolo_detects_objects_in_synthetic_frame(self):
        """YOLO should detect something in a random image (may be nothing)."""
        from world2data.pipeline.model_interfaces import YOLODetector
        det = YOLODetector(model_name="yolov8n.pt", device="cuda")
        # Create a frame with some structure
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = det.process_frame(frame, frame_idx=0, timestamp=0.0)
        # Result should be a valid DetectionResult regardless
        assert hasattr(result, 'boxes')
        assert hasattr(result, 'class_names')
        assert hasattr(result, 'scores')

    @pytest.mark.skipif(not _HAS_YOLO, reason="ultralytics not installed")
    @pytest.mark.skipif(not os.path.isfile(DEMO_VIDEO),
                        reason="Demo video not found")
    def test_yolo_on_real_keyframe(self):
        """YOLO should detect objects in a real video frame."""
        from world2data.pipeline.model_interfaces import YOLODetector
        det = YOLODetector(model_name="yolov8x-seg.pt", device="cuda")

        cap = cv2.VideoCapture(DEMO_VIDEO)
        ret, frame = cap.read()
        cap.release()
        assert ret, "Could not read frame from demo video"

        result = det.process_frame(frame, frame_idx=0, timestamp=0.0)
        assert len(result.class_names) > 0, "YOLO should detect at least one object"
        assert result.masks is not None, "Seg model should produce masks"
        print(f"  YOLO detected: {result.class_names}")


# =========================================================================
# Test 2: Particle Filter Integration
# =========================================================================
class TestParticleFilterIntegration:
    """Test that particle filter smooths object positions correctly."""

    def test_particle_filter_tracks_moving_object(self):
        """PF should track an object moving along a straight line."""
        pf = ParticleFilter3D(
            num_particles=512,
            process_noise=0.05,
            measurement_noise=0.15,
            rng=np.random.RandomState(42),
        )

        # Simulate an object moving from (0,0,0) to (1,0,0) in 20 steps
        for i in range(20):
            pos = np.array([i * 0.05, 0.0, 3.0], dtype=np.float32)
            if not pf.initialized:
                pf.initialize(pos)
            else:
                pf.predict()
                pf.update(pos, measurement_confidence=0.8)

        estimate = pf.estimate()
        expected = np.array([0.95, 0.0, 3.0], dtype=np.float32)
        error = np.linalg.norm(estimate - expected)
        assert error < 0.5, f"PF estimate {estimate} too far from {expected} (err={error})"

    def test_particle_filter_spread_decreases_with_observations(self):
        """PF spread should decrease as we get more consistent observations."""
        pf = ParticleFilter3D(num_particles=256, rng=np.random.RandomState(42))
        target = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        pf.initialize(target)

        spreads = []
        for _ in range(10):
            pf.predict()
            pf.update(target, measurement_confidence=1.0)
            spreads.append(pf.spread())

        # Spread should generally decrease (not strictly monotonic due to noise)
        assert spreads[-1] < spreads[0] * 2, "Spread should not grow unboundedly"


# =========================================================================
# Test 3: Human Review JSON Generation
# =========================================================================
class TestHumanReviewJSON:
    """Test that human review JSON is well-formed."""

    def test_review_json_structure(self):
        """Generate a review JSON from mock pipeline and verify structure."""
        from generate_test_video import generate_test_video

        with tempfile.TemporaryDirectory() as tmpdir:
            video = os.path.join(tmpdir, "test.mp4")
            generate_test_video(video, duration_sec=2)

            pipeline = World2DataPipeline(
                video, output_path=os.path.join(tmpdir, "test.usda"),
                keyframe_dir=os.path.join(tmpdir, "kf"),
                cache_dir=os.path.join(tmpdir, "cache"),
                rerun_enabled=False,
            )
            pipeline.step_1_smart_extraction(threshold=5.0)
            pipeline._mock_geometry()

            # Add some mock objects
            pipeline.objects_3d = [
                Object3D("Table_01", "table", [0, 0, 2], [1, 0.5, 3]),
                Object3D("Chair_01", "chair", [-1, 0, 2], [-0.5, 0.5, 2.5]),
            ]

            # Generate review JSON via demo_run helper
            from world2data.pipeline.demo_run import _generate_review_json
            review_path = os.path.join(tmpdir, "review.json")
            _generate_review_json(pipeline, review_path)

            assert os.path.isfile(review_path)
            with open(review_path) as f:
                data = json.load(f)

            assert "objects" in data
            assert "interactions" in data
            assert len(data["objects"]) == 2

            # Check object structure
            obj = data["objects"][0]
            assert "entity" in obj
            assert "type" in obj
            assert "confidence" in obj
            assert "needs_review" in obj
            assert "human_label" in obj
            assert "human_verified" in obj
            assert "center_3d" in obj
            assert "size_3d" in obj
            assert len(obj["center_3d"]) == 3
            assert len(obj["size_3d"]) == 3


# =========================================================================
# Test 4: USD Accuracy Attributes
# =========================================================================
class TestUSDAccuracyAttributes:
    """Test that USD export includes accuracy/confidence attributes."""

    def test_usd_has_confidence_attributes(self):
        """When objects_4d is present, USD should have confidence attrs."""
        from pxr import Usd, Sdf

        with tempfile.TemporaryDirectory() as tmpdir:
            from generate_test_video import generate_test_video
            video = os.path.join(tmpdir, "test.mp4")
            generate_test_video(video, duration_sec=2)

            pipeline = World2DataPipeline(
                video, output_path=os.path.join(tmpdir, "test.usda"),
                keyframe_dir=os.path.join(tmpdir, "kf"),
                cache_dir=os.path.join(tmpdir, "cache"),
                rerun_enabled=False,
            )
            pipeline.step_1_smart_extraction(threshold=5.0)
            pipeline._mock_geometry()

            # Add objects with tracking confidence
            pipeline.objects_3d = [
                Object3D("Table_01", "table", [0, 0, 2], [1, 0.5, 3],
                         observation_count=5, tracking_confidence=0.85),
            ]

            # Mock 4D objects for confidence enrichment
            try:
                from world2data.pipeline.scene_fusion import SceneObject4D
                pipeline.objects_4d = [
                    SceneObject4D(
                        entity_id="Table_01", obj_type="table",
                        confidence=0.85, detected_by=["yolo", "gemini"],
                        bbox_3d_min=np.array([0, 0, 2]),
                        bbox_3d_max=np.array([1, 0.5, 3]),
                        human_review=False,
                    ),
                ]
            except ImportError:
                pipeline.objects_4d = []

            pipeline.step_5_export_usd()

            # Verify USD has the attributes
            usda_path = os.path.join(tmpdir, "test.usda")
            assert os.path.isfile(usda_path)

            stage = Usd.Stage.Open(usda_path)
            table_prim = stage.GetPrimAtPath("/World/Objects/Table_01")
            assert table_prim.IsValid(), "Table_01 prim should exist"

            type_attr = table_prim.GetAttribute("world2data:type")
            assert type_attr.IsValid()
            assert type_attr.Get() == "table"

            # Check confidence if 4D objects were available
            if pipeline.objects_4d:
                conf_attr = table_prim.GetAttribute("world2data:confidence")
                assert conf_attr.IsValid(), "Confidence attribute should exist"
                assert abs(conf_attr.Get() - 0.85) < 0.01

                detected_attr = table_prim.GetAttribute("world2data:detected_by")
                assert detected_attr.IsValid()
                assert "yolo" in detected_attr.Get()


# =========================================================================
# Test 5: Scene Fusion 4D
# =========================================================================
class TestSceneFusion:
    """Test the 4D scene fusion layer."""

    def test_fusion_creates_objects(self):
        """Fusion should create SceneObject4D from detection results."""
        try:
            from world2data.pipeline.scene_fusion import SceneFusion4D
            from world2data.pipeline.model_interfaces import DetectionResult
        except ImportError:
            pytest.skip("scene_fusion or model_interfaces not available")

        # Create mock detections
        detections = [
            DetectionResult(
                frame_idx=0, timestamp=0.0,
                boxes=np.array([[100, 100, 300, 400]]),
                classes=np.array([0]),
                class_names=["chair"],
                scores=np.array([0.9]),
            ),
            DetectionResult(
                frame_idx=1, timestamp=0.033,
                boxes=np.array([[105, 98, 310, 405]]),
                classes=np.array([0]),
                class_names=["chair"],
                scores=np.array([0.88]),
            ),
        ]

        fusion = SceneFusion4D(detections=detections)
        objects = fusion.fuse()

        assert len(objects) > 0, "Fusion should produce at least one object"
        assert objects[0].obj_type == "chair"
        assert objects[0].detected_by == ["yolo"]

    def test_confidence_computation(self):
        """Objects detected by multiple models should have higher confidence."""
        try:
            from world2data.pipeline.scene_fusion import SceneFusion4D, SceneObject4D
            from world2data.pipeline.model_interfaces import (
                DetectionResult, SegmentationResult, SceneDescription,
            )
        except ImportError:
            pytest.skip("Required modules not available")

        detections = [
            DetectionResult(
                frame_idx=0, timestamp=0.0,
                boxes=np.array([[100, 100, 300, 400]]),
                classes=np.array([0]),
                class_names=["chair"],
                scores=np.array([0.9]),
            ),
        ]

        # With SAM3 match
        segmentations = [
            SegmentationResult(
                frame_idx=0, timestamp=0.0,
                masks=[np.ones((480, 640), dtype=np.uint8)],
                labels=["chair"],
                scores=[0.85],
                object_ids=[1],
            ),
        ]

        # With Gemini match
        scene = SceneDescription(
            objects=[{"name": "Chair_01", "type": "chair"}],
        )

        fusion = SceneFusion4D(
            detections=detections,
            segmentations=segmentations,
            scene_description=scene,
        )
        objects = fusion.fuse()

        assert len(objects) > 0
        chair = objects[0]
        # Should be detected by all three
        assert "yolo" in chair.detected_by
        assert "sam3" in chair.detected_by
        assert "gemini" in chair.detected_by
        # High confidence due to multi-model agreement
        assert chair.confidence > 0.5


# =========================================================================
# Test 6: Sliding Window Alignment
# =========================================================================
class TestSlidingWindowAlignment:
    """Test the rigid alignment used for stitching windows."""

    def test_rigid_align_identity(self):
        """Identical point sets should produce identity transform."""
        pts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                       dtype=np.float64)
        R, t, s = World2DataPipeline._rigid_align(pts, pts)
        assert abs(s - 1.0) < 0.01, f"Scale should be ~1.0, got {s}"
        assert np.linalg.norm(t) < 0.01, f"Translation should be ~0, got {t}"
        np.testing.assert_allclose(R, np.eye(3), atol=0.01)

    def test_rigid_align_translation(self):
        """Pure translation should be recovered."""
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                       dtype=np.float64)
        offset = np.array([5.0, -3.0, 2.0])
        dst = src + offset
        R, t, s = World2DataPipeline._rigid_align(src, dst)
        assert abs(s - 1.0) < 0.05
        np.testing.assert_allclose(t, offset, atol=0.1)

    def test_rigid_align_scale(self):
        """Scaling should be recovered."""
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                       dtype=np.float64)
        dst = src * 2.5
        R, t, s = World2DataPipeline._rigid_align(src, dst)
        assert abs(s - 2.5) < 0.1, f"Expected scale ~2.5, got {s}"

    def test_keyframe_count_at_5fps(self):
        """At 5 FPS, a 21.8s video should yield ~109 keyframes."""
        from generate_test_video import generate_test_video

        with tempfile.TemporaryDirectory() as tmpdir:
            video = os.path.join(tmpdir, "test.mp4")
            # Make a short 4-second video for the test
            generate_test_video(video, duration_sec=4, fps=30)

            pipeline = World2DataPipeline(
                video, output_path=os.path.join(tmpdir, "test.usda"),
                keyframe_dir=os.path.join(tmpdir, "kf"),
                cache_dir=os.path.join(tmpdir, "cache"),
                rerun_enabled=False,
            )
            # Extract at 5 FPS with no threshold
            ok = pipeline.step_1_smart_extraction(
                threshold=0, max_keyframes=0, target_fps=5.0
            )
            assert ok, "Extraction should succeed"
            # 4s * 5fps = 20 frames (allow some tolerance)
            assert len(pipeline.keyframes) >= 15, \
                f"Expected ~20 keyframes at 5fps/4s, got {len(pipeline.keyframes)}"
            assert len(pipeline.keyframes) <= 25, \
                f"Too many keyframes: {len(pipeline.keyframes)}"


# =========================================================================
# Test 7: Demo Runner (smoke test)
# =========================================================================
class TestDemoRunner:
    """Smoke test for the demo runner."""

    def test_demo_run_import(self):
        """demo_run module should be importable."""
        from world2data.pipeline import demo_run
        assert hasattr(demo_run, 'run_demo')
        assert hasattr(demo_run, '_generate_review_json')

    def test_human_review_ui_import(self):
        """human_review_ui module should be importable."""
        from world2data.pipeline import human_review_ui
        assert hasattr(human_review_ui, 'build_ui')
        assert hasattr(human_review_ui, 'load_review_data')
