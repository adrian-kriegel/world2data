from __future__ import annotations

import pytest

from world2data.model import TrackEstimate
from world2data.openusd import track_estimates_to_stage, track_estimates_to_usda


def test_track_estimates_to_stage_creates_expected_prims() -> None:
    pytest.importorskip("pxr")

    estimates = {
        "box-1": TrackEstimate(
            track_id="box-1",
            label="box",
            position=(1.0, 2.0, 3.0),
            bounding_box=(0.9, 1.0, 0.8),
            mass=1.2,
            velocity=(0.0, 0.0, 0.0),
            particle_count=128,
        )
    }

    stage = track_estimates_to_stage(estimates, frame_index=7)
    prim = stage.GetPrimAtPath("/World/ParticleCentroids/box_1")

    assert prim
    assert prim.GetAttribute("world2data:trackId").Get() == "box-1"
    assert prim.GetAttribute("world2data:label").Get() == "box"
    assert prim.GetAttribute("world2data:particleCount").Get() == 128


def test_track_estimates_to_usda_is_parseable() -> None:
    pytest.importorskip("pxr")

    from pxr import Sdf

    estimates = {
        "crate-1": TrackEstimate(
            track_id="crate-1",
            label="crate",
            position=(0.0, 0.0, 5.0),
            bounding_box=(1.0, 1.0, 1.0),
            mass=3.0,
            velocity=(0.0, 0.0, 0.0),
            particle_count=64,
        )
    }

    usda = track_estimates_to_usda(estimates, frame_index=0)
    layer = Sdf.Layer.CreateAnonymous(".usda")
    assert layer.ImportFromString(usda)
