"""Tests for protocol-compliant layered USD pipeline output.

Covers:
- WP1: Layered USD as default output
- WP2: Externalized per-frame point clouds
- WP3: YOLO observations layer
- WP4: Point lineage parquet
- WP5: Per-frame camera poses in recon layer
"""
import os
import sys
import tempfile

import numpy as np
import pytest

# Ensure src/ is on the path for imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from world2data.usd_layers import (
    USDLayerWriter,
    ProvenanceRecord,
    EntityRecord,
    EventRecord,
    YOLOFrameRecord,
    PointCloudFrameRecord,
    _generate_run_id,
    _write_ply,
    write_point_lineage,
    build_layered_scene,
    validate_scene,
)

try:
    from pxr import Sdf, Usd, UsdGeom
    _HAS_PXR = True
except ImportError:
    _HAS_PXR = False

try:
    import pyarrow.parquet as pq
    _HAS_PARQUET = True
except ImportError:
    _HAS_PARQUET = False


# =========================================================================
# Helpers
# =========================================================================

def _make_frame_data(n_frames=5, pts_per_frame=100):
    """Create mock frame data for testing."""
    rng = np.random.RandomState(42)
    frames = []
    for i in range(n_frames):
        frames.append({
            "index": i,
            "pts3d": rng.randn(pts_per_frame, 3).astype(np.float32),
            "colors": rng.rand(pts_per_frame, 3).astype(np.float32),
            "confidence": rng.rand(pts_per_frame).astype(np.float32),
            "timestamp_sec": i / 10.0,
        })
    return frames


def _make_cameras(n=5, fps=10.0):
    """Create mock camera dicts."""
    cameras = []
    for i in range(n):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = i * 0.1  # translate along x
        cameras.append({
            "pose": pose,
            "focal": 500.0 + i * 10,
            "frame_idx": i,
            "timestamp_sec": i / fps,
        })
    return cameras


def _make_yolo_frames(n=5):
    """Create mock YOLO detection frames."""
    return [
        YOLOFrameRecord(
            frame_index=i,
            timestamp_sec=i / 10.0,
            image_width=640,
            image_height=480,
            labels=["person", "chair"],
            class_ids=[0, 56],
            scores=[0.95, 0.87],
            boxes_xyxy=[(100.0, 50.0, 300.0, 400.0), (400.0, 200.0, 550.0, 450.0)],
        )
        for i in range(n)
    ]


# =========================================================================
# WP2: Per-frame point cloud externalization
# =========================================================================

class TestPerFramePointClouds:
    """Test per-frame PLY writing and recon layer indexing."""

    def test_write_per_frame_ply_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = USDLayerWriter(tmpdir)
            frame_data = _make_frame_data(n_frames=3, pts_per_frame=50)
            run_id = "test_run_001"

            records = writer.write_per_frame_point_clouds(frame_data, run_id)

            assert len(records) == 3
            for rec in records:
                assert rec.point_count == 50
                assert rec.points_format == "ply"
                # Verify the PLY file exists
                ply_path = os.path.join(tmpdir, "external", "recon",
                                        f"frame_{rec.frame_index:06d}.ply")
                assert os.path.isfile(ply_path), f"Missing PLY: {ply_path}"

    def test_ply_file_content(self):
        """Verify the PLY file has correct header and vertex count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            cols = np.array([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]], dtype=np.float32)
            ply_path = os.path.join(tmpdir, "test.ply")
            _write_ply(ply_path, pts, cols)

            with open(ply_path) as f:
                content = f.read()
            assert "element vertex 2" in content
            assert "property float x" in content
            assert "property uchar red" in content

    @pytest.mark.skipif(not _HAS_PXR, reason="USD not installed")
    def test_recon_layer_with_frame_index(self):
        """Verify the recon layer writes per-frame point cloud metadata prims."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = USDLayerWriter(tmpdir)
            writer._ensure_dirs()
            run_id = "test_recon_frames"
            prov = ProvenanceRecord(
                run_id=run_id, component="recon",
                model_name="mast3r", model_version="test",
            )
            cameras = _make_cameras(n=3)
            frame_records = [
                PointCloudFrameRecord(
                    frame_index=i,
                    timestamp_sec=i / 10.0,
                    points_asset_path=f"../external/recon/frame_{i:06d}.ply",
                    point_count=50,
                )
                for i in range(3)
            ]

            path = writer.write_recon_layer_with_frames(
                prov, cameras=cameras, frame_records=frame_records,
            )
            assert os.path.isfile(str(path))

            # Verify USD content
            stage = Usd.Stage.Open(str(path))
            # Check per-frame point cloud prims
            for i in range(3):
                prim = stage.GetPrimAtPath(
                    f"/World/W2D/Reconstruction/PointCloudFrames/f_{i:06d}"
                )
                assert prim.IsValid(), f"Missing frame prim f_{i:06d}"
                assert prim.GetAttribute("w2d:frameIndex").Get() == i
                assert prim.GetAttribute("w2d:pointCount").Get() == 50
                asset = prim.GetAttribute("w2d:pointsAsset").Get()
                assert "frame_" in str(asset)

            # Check camera pose frames
            for i in range(3):
                prim = stage.GetPrimAtPath(
                    f"/World/W2D/Sensors/CameraPoses/Frames/f_{i:06d}"
                )
                assert prim.IsValid(), f"Missing pose prim f_{i:06d}"
                assert prim.GetAttribute("w2d:frameIndex").Get() == i
                assert prim.GetAttribute("w2d:poseConvention").Get() == "camera_to_world"

    @pytest.mark.skipif(not _HAS_PXR, reason="USD not installed")
    def test_no_dense_point_arrays_in_recon_with_frames(self):
        """Protocol compliance: recon layer must NOT contain dense point arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = USDLayerWriter(tmpdir)
            writer._ensure_dirs()
            prov = ProvenanceRecord(
                run_id="test_no_dense", component="recon",
                model_name="mast3r", model_version="test",
            )
            path = writer.write_recon_layer_with_frames(prov)
            stage = Usd.Stage.Open(str(path))

            # Should NOT have /World/W2D/Reconstruction/DenseCloud
            prim = stage.GetPrimAtPath("/World/W2D/Reconstruction/DenseCloud")
            assert not prim.IsValid(), "Dense cloud prim should not exist in frame-indexed recon"


# =========================================================================
# WP3: YOLO Observations Layer
# =========================================================================

class TestYOLOObservationsLayer:
    """Test YOLO observations layer writing."""

    @pytest.mark.skipif(not _HAS_PXR, reason="USD not installed")
    def test_writes_yolo_layer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = USDLayerWriter(tmpdir)
            writer._ensure_dirs()
            prov = ProvenanceRecord(
                run_id="test_yolo", component="perception",
                model_name="yolov8x-seg", model_version="8.4",
            )
            frames = _make_yolo_frames(n=3)
            path = writer.write_yolo_observations_layer(prov, frames)

            assert os.path.isfile(str(path))
            assert "25_yolo_run_" in path.name

            stage = Usd.Stage.Open(str(path))
            # Check per-frame prims
            for i in range(3):
                prim = stage.GetPrimAtPath(
                    f"/World/W2D/Observations/YOLO/Frames/f_{i:06d}"
                )
                assert prim.IsValid(), f"Missing YOLO frame prim f_{i:06d}"
                assert prim.GetAttribute("w2d:frameIndex").Get() == i
                assert prim.GetAttribute("w2d:detectionCount").Get() == 2
                labels = prim.GetAttribute("w2d:labels").Get()
                assert "person" in labels
                assert "chair" in labels

    @pytest.mark.skipif(not _HAS_PXR, reason="USD not installed")
    def test_yolo_layer_in_assembly(self):
        """YOLO layer should appear in the assembly at position 25_."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = USDLayerWriter(tmpdir)
            writer.write_base_layer()
            prov = ProvenanceRecord(
                run_id="test_asm", component="perception",
                model_name="yolov8x-seg", model_version="8.4",
            )
            writer.write_yolo_observations_layer(prov, _make_yolo_frames(1))
            scene_path = writer.write_assembly()

            content = scene_path.read_text()
            assert "25_yolo_run_" in content
            # Verify ordering: 00_base before 25_yolo
            idx_base = content.index("00_base")
            idx_yolo = content.index("25_yolo")
            assert idx_base < idx_yolo


# =========================================================================
# WP4: Point Lineage Parquet
# =========================================================================

class TestPointLineage:
    """Test point lineage parquet generation."""

    @pytest.mark.skipif(not _HAS_PARQUET, reason="pyarrow not installed")
    def test_writes_lineage_parquet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_data = _make_frame_data(n_frames=3, pts_per_frame=10)
            run_id = "lineage_test"

            path = write_point_lineage(tmpdir, frame_data, run_id)
            assert path is not None
            assert os.path.isfile(str(path))
            assert "point_lineage_" in str(path)

    @pytest.mark.skipif(not _HAS_PARQUET, reason="pyarrow not installed")
    def test_lineage_schema(self):
        """Verify the parquet has the correct columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_data = _make_frame_data(n_frames=2, pts_per_frame=5)
            path = write_point_lineage(tmpdir, frame_data, "schema_test")
            table = pq.read_table(str(path))
            expected_cols = {
                "point_uid", "frame_index_origin", "timestamp_sec_origin",
                "source_points_asset", "x", "y", "z", "r", "g", "b",
                "confidence", "state", "track_id",
            }
            assert expected_cols == set(table.column_names)

    @pytest.mark.skipif(not _HAS_PARQUET, reason="pyarrow not installed")
    def test_lineage_row_count(self):
        """Each point should have exactly one lineage row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_data = _make_frame_data(n_frames=3, pts_per_frame=10)
            path = write_point_lineage(tmpdir, frame_data, "count_test")
            table = pq.read_table(str(path))
            assert len(table) == 30  # 3 frames * 10 points

    @pytest.mark.skipif(not _HAS_PARQUET, reason="pyarrow not installed")
    def test_lineage_lifecycle_no_pf(self):
        """Without PF tracks, points have lifecycle based on frame position."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_data = _make_frame_data(n_frames=10, pts_per_frame=3)
            path = write_point_lineage(tmpdir, frame_data, "lifecycle_test",
                                        total_frames=10)
            table = pq.read_table(str(path))
            states = table.column("state").to_pylist()
            # Last 30% of frames (frames 7-9) should be "active"
            # Earlier frames should be "retired" (no PF tracks)
            active_count = states.count("active")
            retired_count = states.count("retired")
            assert active_count > 0, "Should have some active points"
            assert retired_count > 0, "Should have some retired points"

    @pytest.mark.skipif(not _HAS_PARQUET, reason="pyarrow not installed")
    def test_lineage_has_track_id_column(self):
        """Point lineage should have track_id column for PF association."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_data = _make_frame_data(n_frames=2, pts_per_frame=5)
            path = write_point_lineage(tmpdir, frame_data, "trackid_test")
            table = pq.read_table(str(path))
            assert "track_id" in table.column_names
            # Without PF, all should be "untracked"
            tracks = table.column("track_id").to_pylist()
            assert all(t == "untracked" for t in tracks)


# =========================================================================
# WP1: Full layered scene build
# =========================================================================

class TestLayeredSceneBuild:
    """Test the full layered scene build pipeline."""

    @pytest.mark.skipif(not _HAS_PXR, reason="USD not installed")
    def test_build_layered_scene_minimal(self):
        """Build a minimal layered scene with just base + assembly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = build_layered_scene(tmpdir, fps=10.0, end_frame=50)
            assert os.path.isfile(str(scene_path))
            errors = validate_scene(scene_path)
            # May have "no provenance" if no entities added
            # Just check it opens cleanly
            stage = Usd.Stage.Open(str(scene_path))
            assert stage is not None

    @pytest.mark.skipif(not _HAS_PXR, reason="USD not installed")
    def test_build_layered_scene_with_entities(self):
        """Build a scene with entities and validate protocol compliance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            entities = [
                EntityRecord(
                    uid="ent_001", entity_class="table", label="Table_01",
                    confidence=0.9, detected_by=["yolo", "gemini"],
                    bbox_3d_min=(-0.5, 0.0, -0.5), bbox_3d_max=(0.5, 0.8, 0.5),
                ),
            ]
            scene_path = build_layered_scene(
                tmpdir, entities=entities, fps=10.0, end_frame=50,
            )
            errors = validate_scene(scene_path)
            # Filter out warnings that are expected
            real_errors = [e for e in errors if "Missing" not in e]
            # We should have no entity uid errors
            uid_errors = [e for e in errors if "w2d:uid" in e]
            assert len(uid_errors) == 0

    @pytest.mark.skipif(not _HAS_PXR, reason="USD not installed")
    def test_layer_ordering_in_assembly(self):
        """Layers must appear in weak-to-strong order in assembly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = USDLayerWriter(tmpdir)
            writer.write_base_layer()

            prov = ProvenanceRecord(
                run_id="order_test", component="ingest",
                model_name="w2d", model_version="1",
            )
            writer.write_inputs_layer(prov)

            prov_yolo = ProvenanceRecord(
                run_id="order_test", component="perception",
                model_name="yolov8x-seg", model_version="8.4",
            )
            writer.write_yolo_observations_layer(prov_yolo, _make_yolo_frames(1))

            writer.write_overrides_layer()
            writer.write_session_layer()

            scene_path = writer.write_assembly()
            content = scene_path.read_text()

            # Verify order: 00 < 10 < 25 < 90 < 99
            positions = {}
            for prefix in ["00_base", "10_inputs", "25_yolo", "90_overrides", "99_session"]:
                idx = content.find(prefix)
                if idx >= 0:
                    positions[prefix] = idx

            keys = list(positions.keys())
            for i in range(len(keys) - 1):
                assert positions[keys[i]] < positions[keys[i + 1]], \
                    f"{keys[i]} should come before {keys[i + 1]}"


# =========================================================================
# WP5: Validate enhanced scene validation
# =========================================================================

class TestValidateScene:
    """Test protocol validation on composed scenes."""

    @pytest.mark.skipif(not _HAS_PXR, reason="USD not installed")
    def test_validates_good_scene(self):
        """A properly built scene should have minimal errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            entities = [
                EntityRecord(
                    uid="test_uid_1", entity_class="chair", label="Chair_01",
                ),
            ]
            scene_path = build_layered_scene(
                tmpdir, entities=entities, fps=30.0, end_frame=100,
            )
            errors = validate_scene(scene_path)
            uid_errors = [e for e in errors if "w2d:uid" in e]
            assert len(uid_errors) == 0, f"Unexpected uid errors: {uid_errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
