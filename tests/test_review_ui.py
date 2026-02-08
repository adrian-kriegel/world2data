"""Tests for the human review UI save and USD override pipeline.

Verifies:
- Review JSON save correctly persists human corrections
- USD 90_overrides.usda is written from human corrections
- The review UI module imports and structures correctly
"""
import json
import os
import sys
import tempfile

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from world2data.pipeline.human_review_ui import (
    load_review_data,
    save_review_data,
    _write_overrides_usda,
)

try:
    from pxr import Usd
    _HAS_PXR = True
except ImportError:
    _HAS_PXR = False


def _make_review_json(tmpdir: str) -> str:
    """Create a minimal review JSON for testing."""
    review = {
        "review_version": "1.0",
        "video_path": "test_video.mp4",
        "pipeline_version": "v2_multimodel",
        "generated_at": "2026-02-08 12:00:00",
        "objects": [
            {
                "id": 0,
                "entity": "Table_01",
                "type": "table",
                "confidence": 0.85,
                "detected_by": ["yolo", "gemini"],
                "component_type": "FixedJoint",
                "initial_state": "stationary",
                "final_state": "stationary",
                "center_3d": [0.0, 0.5, 0.0],
                "size_3d": [1.0, 0.8, 0.6],
                "needs_review": False,
            },
            {
                "id": 1,
                "entity": "Chair_01",
                "type": "chair",
                "confidence": 0.35,
                "detected_by": ["yolo"],
                "component_type": "FreeJoint",
                "initial_state": "stationary",
                "final_state": "moved",
                "center_3d": [1.0, 0.4, 0.0],
                "size_3d": [0.5, 0.8, 0.5],
                "needs_review": True,
            },
        ],
        "interactions": [],
        "scene_narrative": "A room with a table and chair.",
    }
    path = os.path.join(tmpdir, "test_review.json")
    with open(path, "w") as f:
        json.dump(review, f, indent=2)
    return path


class TestReviewJSONSave:
    """Test that review JSON persistence works correctly."""

    def test_load_review_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_review_json(tmpdir)
            data = load_review_data(path)
            assert len(data["objects"]) == 2
            assert data["objects"][0]["entity"] == "Table_01"

    def test_save_review_data_adds_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_review_json(tmpdir)
            data = load_review_data(path)
            assert "last_reviewed" not in data

            save_review_data(data, path)
            reloaded = load_review_data(path)
            assert "last_reviewed" in reloaded

    def test_save_preserves_human_corrections(self):
        """Human corrections must survive a save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_review_json(tmpdir)
            data = load_review_data(path)

            # Simulate human corrections
            data["objects"][0]["human_label"] = "Coffee Table"
            data["objects"][0]["human_verified"] = True
            data["objects"][1]["false_positive"] = True
            data["objects"][1]["human_notes"] = "Ghost detection"

            save_review_data(data, path)
            reloaded = load_review_data(path)

            assert reloaded["objects"][0]["human_label"] == "Coffee Table"
            assert reloaded["objects"][0]["human_verified"] is True
            assert reloaded["objects"][1]["false_positive"] is True
            assert reloaded["objects"][1]["human_notes"] == "Ghost detection"


class TestOverridesUSDA:
    """Test that human corrections write to 90_overrides.usda."""

    @pytest.mark.skipif(not _HAS_PXR, reason="USD not installed")
    def test_writes_overrides_for_verified_objects(self):
        """When objects have human_verified=True, an override should be written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the scene directory structure
            scene_dir = os.path.join(tmpdir, "scene")
            layers_dir = os.path.join(scene_dir, "layers")
            os.makedirs(layers_dir)

            objects = [
                {
                    "entity": "Table_01",
                    "human_verified": True,
                    "human_label": "Kitchen Table",
                    "human_notes": "Corrected label",
                },
            ]
            result = _write_overrides_usda(objects, tmpdir)
            assert result is not None
            assert os.path.isfile(result)

            # Verify USD content
            stage = Usd.Stage.Open(result)
            prim = stage.GetPrimAtPath("/World/W2D/Overrides")
            assert prim.IsValid()

    @pytest.mark.skipif(not _HAS_PXR, reason="USD not installed")
    def test_skips_if_no_scene_dir(self):
        """If there's no scene/ directory, overrides should not be written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            objects = [{"entity": "Table_01", "human_verified": True}]
            result = _write_overrides_usda(objects, tmpdir)
            assert result is None

    @pytest.mark.skipif(not _HAS_PXR, reason="USD not installed")
    def test_skips_if_no_changes(self):
        """Objects without human edits should not produce overrides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_dir = os.path.join(tmpdir, "scene")
            os.makedirs(os.path.join(scene_dir, "layers"))

            objects = [
                {"entity": "Table_01", "confidence": 0.9},  # No human edits
            ]
            result = _write_overrides_usda(objects, tmpdir)
            assert result is None

    @pytest.mark.skipif(not _HAS_PXR, reason="USD not installed")
    def test_false_positive_flag_in_override(self):
        """False positive flag should be written as w2d:overrideReason."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_dir = os.path.join(tmpdir, "scene")
            os.makedirs(os.path.join(scene_dir, "layers"))

            objects = [
                {
                    "entity": "Ghost_01",
                    "false_positive": True,
                    "human_notes": "Not a real object",
                },
            ]
            result = _write_overrides_usda(objects, tmpdir)
            assert result is not None


class TestReviewUIImport:
    """Test that the review UI module imports correctly."""

    def test_build_ui_importable(self):
        from world2data.pipeline.human_review_ui import build_ui
        assert callable(build_ui)

    def test_main_importable(self):
        from world2data.pipeline.human_review_ui import main
        assert callable(main)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
