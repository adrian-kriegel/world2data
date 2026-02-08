from __future__ import annotations

import pytest

from world2data.calibration import CalibrationResult, infer_distortion_model, write_calibration_usda


def test_infer_distortion_model() -> None:
    assert infer_distortion_model([0.1, 0.01, 0.0, 0.0, 0.001]) == "opencv_plumb_bob"
    assert infer_distortion_model([0.1] * 8) == "opencv_rational"
    assert infer_distortion_model([0.1] * 4) == "opencv_fisheye_4"


def test_calibration_result_to_dict() -> None:
    result = CalibrationResult(
        image_size=(1920, 1080),
        camera_matrix=[[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
        distortion_coeffs=[0.01, -0.02, 0.0, 0.0, 0.001],
        distortion_model="opencv_plumb_bob",
        reprojection_error_px=0.42,
        frames_sampled=200,
        frames_used=48,
    )

    payload = result.to_dict()
    assert payload["image_size"] == [1920, 1080]
    assert payload["distortion_model"] == "opencv_plumb_bob"
    assert payload["frames_used"] == 48


def test_write_calibration_usda_creates_camera_prim(tmp_path) -> None:
    pytest.importorskip("pxr")

    result = CalibrationResult(
        image_size=(640, 480),
        camera_matrix=[[620.0, 0.0, 320.0], [0.0, 618.0, 240.0], [0.0, 0.0, 1.0]],
        distortion_coeffs=[0.1, -0.02, 0.001, 0.0001, 0.0],
        distortion_model="opencv_plumb_bob",
        reprojection_error_px=0.3,
        frames_sampled=100,
        frames_used=30,
    )

    output_path = tmp_path / "calibration.usda"
    write_calibration_usda(
        result=result,
        output_path=output_path,
        run_id="run-123",
        model_version="4.13.0",
        params={"frame_step": 2},
    )

    from pxr import Usd

    stage = Usd.Stage.Open(str(output_path))
    prim = stage.GetPrimAtPath("/World/W2D/Sensors/CalibrationCamera")
    assert prim
    assert prim.GetAttribute("w2d:distortionModel").Get() == "opencv_plumb_bob"
    assert prim.GetAttribute("w2d:producedByRunId").Get() == "run-123"
    assert not prim.HasAttribute("xformOp:transform")
    run_prim = stage.GetPrimAtPath("/World/W2D/Provenance/runs/run_123")
    assert run_prim
