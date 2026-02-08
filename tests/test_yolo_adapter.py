from __future__ import annotations

import pytest

from world2data.yolo_adapter import (
    YoloFrameDetections,
    YoloRawDetection,
    yolo_observations_to_stage,
    yolo_observations_to_usda,
)


def test_yolo_observations_to_stage_is_protocol_aligned() -> None:
    pytest.importorskip("pxr")

    detections = (
        YoloFrameDetections(
            frame_index=0,
            timestamp_s=0.0,
            image_width=640,
            image_height=480,
            detections=(
                YoloRawDetection(
                    class_id=0,
                    label="person",
                    confidence=0.92,
                    bbox_xyxy=(10.0, 20.0, 110.0, 220.0),
                ),
            ),
        ),
        YoloFrameDetections(
            frame_index=3,
            timestamp_s=0.1,
            image_width=640,
            image_height=480,
            detections=(
                YoloRawDetection(
                    class_id=41,
                    label="cup",
                    confidence=0.81,
                    bbox_xyxy=(210.0, 120.0, 260.0, 200.0),
                ),
                YoloRawDetection(
                    class_id=67,
                    label="cell phone",
                    confidence=0.77,
                    bbox_xyxy=(300.0, 140.0, 360.0, 220.0),
                ),
            ),
        ),
    )

    stage = yolo_observations_to_stage(
        detections=detections,
        run_id="RUN_001",
        model_name="yolov8n.pt",
        model_version="8.4.12",
        params={"frame_step": 1},
        git_commit="abc123",
        time_codes_per_second=30.0,
        video_uri="../external/inputs/demo.mp4",
    )

    assert stage.GetDefaultPrim().GetPath().pathString == "/World"
    assert stage.GetPrimAtPath("/World/W2D/Observations/YOLO")
    assert stage.GetPrimAtPath("/World/W2D/Observations/YOLO/Frames/f_000000")
    assert stage.GetPrimAtPath("/World/W2D/Observations/YOLO/Frames/f_000003")

    det = stage.GetPrimAtPath("/World/W2D/Observations/YOLO/Frames/f_000003/det_0001")
    assert det
    assert det.GetAttribute("w2d:class").Get() == "cell phone"
    assert det.GetAttribute("w2d:classId").Get() == 67
    assert det.GetAttribute("w2d:producedByRunId").Get() == "RUN_001"

    run = stage.GetPrimAtPath("/World/W2D/Provenance/runs/RUN_001")
    assert run
    assert run.GetAttribute("w2d:component").Get() == "perception.yolo"
    assert run.GetAttribute("w2d:modelName").Get() == "yolov8n.pt"
    assert run.GetAttribute("w2d:gitCommit").Get() == "abc123"


def test_yolo_observations_to_usda_is_parseable() -> None:
    pytest.importorskip("pxr")
    from pxr import Sdf

    detections = (
        YoloFrameDetections(
            frame_index=1,
            timestamp_s=0.05,
            image_width=320,
            image_height=240,
            detections=(),
        ),
    )

    usda = yolo_observations_to_usda(
        detections=detections,
        run_id="R1",
        model_name="yolov8n.pt",
        model_version="8.4.12",
        params={},
        git_commit="unknown",
        time_codes_per_second=20.0,
    )
    layer = Sdf.Layer.CreateAnonymous(".usda")
    assert layer.ImportFromString(usda)
