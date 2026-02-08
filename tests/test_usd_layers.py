"""Tests for the OpenUSD Layering Protocol (usd_layers.py)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from world2data.usd_layers import (
    EntityRecord,
    EventRecord,
    OverrideRecord,
    ProvenanceRecord,
    USDLayerWriter,
    build_layered_scene,
    validate_scene,
)


@pytest.fixture
def run_id():
    return "test_20260208T120000Z"


@pytest.fixture
def sample_entities():
    return [
        EntityRecord(
            uid="ent_001",
            entity_class="table",
            label="Table_01",
            confidence=0.85,
            detected_by=["yolo", "gemini"],
            bbox_3d_min=(-0.5, 0.0, 2.0),
            bbox_3d_max=(0.5, 0.8, 3.0),
            first_frame=0,
            last_frame=100,
            frame_positions={0: (0.0, 0.4, 2.5), 50: (0.0, 0.4, 2.5), 100: (0.0, 0.4, 2.5)},
        ),
        EntityRecord(
            uid="ent_002",
            entity_class="chair",
            label="Chair_01",
            confidence=0.72,
            detected_by=["yolo"],
            bbox_3d_min=(1.0, 0.0, 2.0),
            bbox_3d_max=(1.6, 0.9, 2.6),
            first_frame=10,
            last_frame=90,
        ),
    ]


@pytest.fixture
def sample_events():
    return [
        EventRecord(
            uid="evt_001",
            predicate="sits_on",
            subject_uid="ent_003",
            object_uid="ent_002",
            t_start=30,
            t_end=80,
            confidence=0.65,
            description="Person sits on chair",
        ),
    ]


class TestBaseLayer:
    def test_creates_base_layer(self, tmp_path):
        pytest.importorskip("pxr")
        from pxr import Usd, UsdGeom

        writer = USDLayerWriter(tmp_path / "scene")
        path = writer.write_base_layer(fps=30.0, start_frame=0, end_frame=900)

        assert path.exists()
        stage = Usd.Stage.Open(str(path))
        assert stage is not None

        # Check default prim
        default_prim = stage.GetDefaultPrim()
        assert default_prim.GetPath() == "/World"

        # Check namespace scopes exist
        for scope in [
            "/World/W2D",
            "/World/W2D/Inputs",
            "/World/W2D/Sensors",
            "/World/W2D/Entities",
            "/World/W2D/Events",
            "/World/W2D/Provenance",
            "/World/W2D/Overrides",
        ]:
            prim = stage.GetPrimAtPath(scope)
            assert prim, f"Missing scope: {scope}"

        # Check stage metadata
        assert stage.GetMetadata("upAxis") == "Y"
        assert stage.GetMetadata("metersPerUnit") == 1.0
        assert stage.GetStartTimeCode() == 0.0
        assert stage.GetEndTimeCode() == 900.0


class TestInputsLayer:
    def test_creates_inputs_layer_without_files(self, tmp_path, run_id):
        pytest.importorskip("pxr")
        from pxr import Usd

        writer = USDLayerWriter(tmp_path / "scene")
        writer.write_base_layer()
        prov = ProvenanceRecord(
            run_id=run_id, component="ingest",
            model_name="world2data", model_version="0.3.0",
        )
        path = writer.write_inputs_layer(prov)
        assert path.exists()
        stage = Usd.Stage.Open(str(path))
        assert stage is not None

    def test_copies_video_to_external(self, tmp_path, run_id):
        pytest.importorskip("pxr")
        from pxr import Usd

        # Create a dummy video file
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"fake video data")

        writer = USDLayerWriter(tmp_path / "scene")
        writer.write_base_layer()
        prov = ProvenanceRecord(
            run_id=run_id, component="ingest",
            model_name="world2data", model_version="0.3.0",
        )
        path = writer.write_inputs_layer(prov, video_path=str(video))

        # Check external file was copied
        external_video = tmp_path / "scene" / "external" / "inputs" / "test_video.mp4"
        assert external_video.exists()

        # Check USD references it
        stage = Usd.Stage.Open(str(path))
        vid_prim = stage.GetPrimAtPath("/World/W2D/Inputs/Video_01")
        assert vid_prim
        uri_val = vid_prim.GetAttribute("w2d:uri").Get()
        # USD returns Sdf.AssetPath; compare the authored path string
        authored = getattr(uri_val, "path", str(uri_val))
        assert authored == "../external/inputs/test_video.mp4"
        assert vid_prim.GetAttribute("w2d:uid").Get() == f"video_{run_id}"


class TestReconLayer:
    def test_creates_cameras_and_points(self, tmp_path, run_id):
        pytest.importorskip("pxr")
        from pxr import Usd, Gf

        cameras = [
            {"pose": np.eye(4).tolist(), "focal": 500.0, "frame_idx": 0},
            {"pose": np.eye(4).tolist(), "focal": 510.0, "frame_idx": 30},
        ]
        cloud = np.random.randn(1000, 3).astype(np.float32)
        colors = np.random.rand(1000, 3).astype(np.float32)

        writer = USDLayerWriter(tmp_path / "scene")
        writer.write_base_layer()
        prov = ProvenanceRecord(
            run_id=run_id, component="recon",
            model_name="mast3r", model_version="ViTLarge_512",
        )
        path = writer.write_recon_layer(
            prov, cameras=cameras,
            point_cloud=cloud, point_colors=colors,
        )

        stage = Usd.Stage.Open(str(path))

        # Check cameras
        cam0 = stage.GetPrimAtPath("/World/W2D/Sensors/Rig_01/Cam_000")
        assert cam0
        assert cam0.GetAttribute("w2d:uid").Get().startswith("cam_")

        cam1 = stage.GetPrimAtPath("/World/W2D/Sensors/Rig_01/Cam_001")
        assert cam1

        # Check point cloud
        pc = stage.GetPrimAtPath("/World/W2D/Reconstruction/DenseCloud")
        assert pc
        pts = pc.GetAttribute("points").Get()
        assert len(pts) == 1000


class TestTracksLayer:
    def test_creates_entities_with_tracks(self, tmp_path, run_id, sample_entities):
        pytest.importorskip("pxr")
        from pxr import Usd

        writer = USDLayerWriter(tmp_path / "scene")
        writer.write_base_layer()
        prov = ProvenanceRecord(
            run_id=run_id, component="tracking",
            model_name="yolov8x-seg", model_version="8.4",
        )
        path = writer.write_tracks_layer(prov, sample_entities)

        stage = Usd.Stage.Open(str(path))

        # Check entity prims
        table = stage.GetPrimAtPath("/World/W2D/Entities/Objects/Table_01")
        assert table
        assert table.GetAttribute("w2d:uid").Get() == "ent_001"
        assert table.GetAttribute("w2d:class").Get() == "table"
        assert table.GetAttribute("w2d:confidence").Get() == pytest.approx(0.85, abs=0.01)
        assert table.GetAttribute("w2d:detectedBy").Get() == "yolo,gemini"

        chair = stage.GetPrimAtPath("/World/W2D/Entities/Objects/Chair_01")
        assert chair
        assert chair.GetAttribute("w2d:uid").Get() == "ent_002"

        # Check track prim (only table has frame_positions)
        track = stage.GetPrimAtPath("/World/W2D/Tracks/Table_01")
        assert track
        assert track.GetAttribute("w2d:uid").Get() == "track_ent_001"

    def test_provenance_recorded(self, tmp_path, run_id, sample_entities):
        pytest.importorskip("pxr")
        from pxr import Usd

        writer = USDLayerWriter(tmp_path / "scene")
        writer.write_base_layer()
        prov = ProvenanceRecord(
            run_id=run_id, component="tracking",
            model_name="yolov8x-seg", model_version="8.4",
        )
        path = writer.write_tracks_layer(prov, sample_entities)

        stage = Usd.Stage.Open(str(path))
        run_prim = stage.GetPrimAtPath(
            f"/World/W2D/Provenance/runs/{run_id}"
        )
        assert run_prim
        assert run_prim.GetAttribute("w2d:runId").Get() == run_id
        assert run_prim.GetAttribute("w2d:component").Get() == "tracking"
        assert run_prim.GetAttribute("w2d:modelName").Get() == "yolov8x-seg"


class TestEventsLayer:
    def test_creates_events(self, tmp_path, run_id, sample_events):
        pytest.importorskip("pxr")
        from pxr import Usd

        writer = USDLayerWriter(tmp_path / "scene")
        writer.write_base_layer()
        prov = ProvenanceRecord(
            run_id=run_id, component="reasoning",
            model_name="gemini-2.5-flash", model_version="2.5",
        )
        path = writer.write_events_layer(prov, sample_events)

        stage = Usd.Stage.Open(str(path))
        # Find the event prim
        events_scope = stage.GetPrimAtPath("/World/W2D/Events")
        assert events_scope
        children = list(events_scope.GetChildren())
        assert len(children) == 1

        evt_prim = children[0]
        assert evt_prim.GetAttribute("w2d:predicate").Get() == "sits_on"
        assert evt_prim.GetAttribute("w2d:tStart").Get() == 30
        assert evt_prim.GetAttribute("w2d:tEnd").Get() == 80
        assert evt_prim.GetAttribute("w2d:confidence").Get() == pytest.approx(0.65, abs=0.01)


class TestOverridesLayer:
    def test_creates_empty_overrides(self, tmp_path):
        pytest.importorskip("pxr")
        from pxr import Usd

        writer = USDLayerWriter(tmp_path / "scene")
        writer.write_base_layer()
        path = writer.write_overrides_layer()
        assert path.exists()

        stage = Usd.Stage.Open(str(path))
        overrides = stage.GetPrimAtPath("/World/W2D/Overrides")
        assert overrides


class TestAssembly:
    def test_assembly_composes_all_layers(self, tmp_path, run_id, sample_entities, sample_events):
        pytest.importorskip("pxr")
        from pxr import Usd

        writer = USDLayerWriter(tmp_path / "scene")
        writer.write_base_layer(fps=30.0, start_frame=0, end_frame=100)

        prov = ProvenanceRecord(
            run_id=run_id, component="tracking",
            model_name="yolov8x-seg", model_version="8.4",
        )
        writer.write_tracks_layer(prov, sample_entities)

        prov_evt = ProvenanceRecord(
            run_id=run_id, component="reasoning",
            model_name="gemini-2.5-flash", model_version="2.5",
        )
        writer.write_events_layer(prov_evt, sample_events)
        writer.write_overrides_layer()
        writer.write_session_layer()

        scene_path = writer.write_assembly(fps=30.0, start_frame=0, end_frame=100)
        assert scene_path.exists()

        # Open the composed scene
        stage = Usd.Stage.Open(str(scene_path))
        assert stage is not None

        # Check that entities from tracks layer are visible
        table = stage.GetPrimAtPath("/World/W2D/Entities/Objects/Table_01")
        assert table, "Entity from tracks layer not composed into scene"

        # Check that events from events layer are visible
        events = stage.GetPrimAtPath("/World/W2D/Events")
        assert events
        assert list(events.GetChildren()), "Events not composed into scene"


class TestBuildLayeredScene:
    def test_one_call_builds_full_scene(self, tmp_path, run_id):
        pytest.importorskip("pxr")
        from pxr import Usd

        entities = [
            EntityRecord(
                uid="e1", entity_class="cup", label="Cup_01",
                confidence=0.9, detected_by=["yolo", "gemini"],
                bbox_3d_min=(0.0, 0.5, 2.0), bbox_3d_max=(0.1, 0.6, 2.1),
            ),
        ]
        events = [
            EventRecord(
                uid="ev1", predicate="picked_up",
                subject_uid="person_01", object_uid="e1",
                t_start=10, t_end=15, confidence=0.7,
                description="Person picks up the cup",
            ),
        ]
        cloud = np.random.randn(500, 3).astype(np.float32)
        colors = np.random.rand(500, 3).astype(np.float32)

        scene_path = build_layered_scene(
            tmp_path / "scene",
            point_cloud=cloud,
            point_colors=colors,
            entities=entities,
            events=events,
            fps=30.0,
            end_frame=100,
            run_id=run_id,
        )

        assert scene_path.exists()
        stage = Usd.Stage.Open(str(scene_path))
        assert stage is not None

        # Validate
        errs = validate_scene(scene_path)
        assert len(errs) == 0, f"Validation errors: {errs}"


class TestValidation:
    def test_valid_scene_passes(self, tmp_path, run_id):
        pytest.importorskip("pxr")

        entities = [
            EntityRecord(
                uid="e1", entity_class="table", label="Table_01",
                confidence=0.8, detected_by=["yolo"],
                bbox_3d_min=(0, 0, 0), bbox_3d_max=(1, 1, 1),
            ),
        ]
        scene_path = build_layered_scene(
            tmp_path / "scene", entities=entities, run_id=run_id,
        )
        errs = validate_scene(scene_path)
        assert errs == []

    def test_detects_missing_provenance(self, tmp_path):
        pytest.importorskip("pxr")
        from pxr import Usd, UsdGeom

        # Build a minimal scene without provenance runs
        scene_dir = tmp_path / "bad_scene"
        scene_dir.mkdir()
        stage = Usd.Stage.CreateNew(str(scene_dir / "scene.usda"))
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.Scope.Define(stage, "/World/W2D")
        UsdGeom.Scope.Define(stage, "/World/W2D/Provenance")
        UsdGeom.Scope.Define(stage, "/World/W2D/Provenance/runs")
        stage.GetRootLayer().Save()

        errs = validate_scene(scene_dir / "scene.usda")
        assert any("provenance" in e.lower() for e in errs)
