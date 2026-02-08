from __future__ import annotations

"""YOLO adapter: run detections and author protocol-aligned OpenUSD observation layers."""

import argparse
import json
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_]")


def _safe_prim_name(raw: str) -> str:
    value = _SAFE_NAME_RE.sub("_", raw).strip("_")
    if not value:
        value = "item"
    if value[0].isdigit():
        value = f"n_{value}"
    return value


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "opencv-python is required for YOLO video adaptation. "
            "Install `opencv-contrib-python`."
        ) from exc
    return cv2


def _require_ultralytics() -> Any:
    try:
        import ultralytics
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ultralytics is required for YOLO video adaptation. "
            "Install `ultralytics`."
        ) from exc
    return ultralytics


def _require_pxr() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "OpenUSD Python bindings are required. Install `usd-core`."
        ) from exc
    return Gf, Sdf, Usd, UsdGeom, Vt


def _git_commit_or_unknown() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    value = result.stdout.strip()
    return value if value else "unknown"


@dataclass(frozen=True)
class YoloRawDetection:
    class_id: int
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class YoloFrameDetections:
    frame_index: int
    timestamp_s: float
    image_width: int
    image_height: int
    detections: tuple[YoloRawDetection, ...]


@dataclass(frozen=True)
class YoloVideoDetections:
    video_path: str
    model_name: str
    model_version: str
    video_fps: float
    processed_frames: int
    frames: tuple[YoloFrameDetections, ...]


def run_yolo_on_video(
    *,
    video_path: Path,
    model_name: str = "yolov8n.pt",
    device: str = "",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    frame_step: int = 1,
    max_frames: int = 0,
) -> YoloVideoDetections:
    cv2 = _require_cv2()
    ultralytics = _require_ultralytics()
    YOLO = ultralytics.YOLO

    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")
    if max_frames < 0:
        raise ValueError("max_frames must be >= 0")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    model = YOLO(model_name)
    model_version = getattr(ultralytics, "__version__", "unknown")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    processed_count = 0
    frame_index = 0
    frames: list[YoloFrameDetections] = []
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % frame_step != 0:
                frame_index += 1
                continue
            if max_frames > 0 and processed_count >= max_frames:
                break

            height, width = frame.shape[:2]
            kwargs: dict[str, Any] = {
                "source": frame,
                "conf": conf_threshold,
                "iou": iou_threshold,
                "verbose": False,
            }
            if device.strip():
                kwargs["device"] = device.strip()
            results = model.predict(**kwargs)
            detection_list = _extract_detections_from_yolo_result(
                result=results[0],
                image_width=int(width),
                image_height=int(height),
            )

            frames.append(
                YoloFrameDetections(
                    frame_index=frame_index,
                    timestamp_s=float(frame_index / video_fps),
                    image_width=int(width),
                    image_height=int(height),
                    detections=tuple(detection_list),
                )
            )

            processed_count += 1
            frame_index += 1
    finally:
        cap.release()

    return YoloVideoDetections(
        video_path=str(video_path),
        model_name=model_name,
        model_version=model_version,
        video_fps=video_fps,
        processed_frames=processed_count,
        frames=tuple(frames),
    )


def _extract_detections_from_yolo_result(
    *,
    result: Any,
    image_width: int,
    image_height: int,
) -> list[YoloRawDetection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    scores = boxes.conf.cpu().numpy()
    names = getattr(result, "names", {})

    detections: list[YoloRawDetection] = []
    for idx in range(len(class_ids)):
        class_id = int(class_ids[idx])
        label = _lookup_class_name(names, class_id)
        x_min, y_min, x_max, y_max = [float(value) for value in xyxy[idx]]

        x_min = max(0.0, min(float(image_width), x_min))
        x_max = max(0.0, min(float(image_width), x_max))
        y_min = max(0.0, min(float(image_height), y_min))
        y_max = max(0.0, min(float(image_height), y_max))
        if x_max <= x_min or y_max <= y_min:
            continue

        detections.append(
            YoloRawDetection(
                class_id=class_id,
                label=label,
                confidence=float(scores[idx]),
                bbox_xyxy=(x_min, y_min, x_max, y_max),
            )
        )
    return detections


def _lookup_class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        if class_id in names:
            return str(names[class_id])
        key = str(class_id)
        if key in names:
            return str(names[key])
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        if 0 <= class_id < len(names):
            return str(names[class_id])
    return f"class_{class_id}"


def yolo_observations_to_stage(
    *,
    detections: Sequence[YoloFrameDetections],
    run_id: str,
    model_name: str,
    model_version: str,
    params: dict[str, Any] | None = None,
    git_commit: str | None = None,
    time_codes_per_second: float = 30.0,
    video_uri: str = "",
) -> Any:
    Gf, Sdf, Usd, UsdGeom, Vt = _require_pxr()

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(float(time_codes_per_second))

    frame_indices = [frame.frame_index for frame in detections]
    if frame_indices:
        stage.SetStartTimeCode(float(min(frame_indices)))
        stage.SetEndTimeCode(float(max(frame_indices)))
    else:
        stage.SetStartTimeCode(0.0)
        stage.SetEndTimeCode(0.0)

    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    UsdGeom.Scope.Define(stage, "/World/W2D")
    UsdGeom.Scope.Define(stage, "/World/W2D/Observations")
    yolo_scope = UsdGeom.Scope.Define(stage, "/World/W2D/Observations/YOLO").GetPrim()
    frames_scope = UsdGeom.Scope.Define(stage, "/World/W2D/Observations/YOLO/Frames").GetPrim()
    UsdGeom.Scope.Define(stage, "/World/W2D/Provenance")
    UsdGeom.Scope.Define(stage, "/World/W2D/Provenance/runs")

    yolo_scope.CreateAttribute("w2d:component", Sdf.ValueTypeNames.String, custom=True).Set(
        "perception.yolo"
    )
    yolo_scope.CreateAttribute("w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True).Set(
        run_id
    )
    yolo_scope.CreateAttribute("w2d:modelName", Sdf.ValueTypeNames.String, custom=True).Set(
        model_name
    )
    yolo_scope.CreateAttribute("w2d:modelVersion", Sdf.ValueTypeNames.String, custom=True).Set(
        model_version
    )
    if video_uri.strip():
        yolo_scope.CreateAttribute("w2d:videoUri", Sdf.ValueTypeNames.Asset, custom=True).Set(
            Sdf.AssetPath(video_uri)
        )

    frame_rel = frames_scope.CreateRelationship("w2d:frames", custom=True)
    total_detections = 0
    for frame in detections:
        frame_path = f"/World/W2D/Observations/YOLO/Frames/f_{frame.frame_index:06d}"
        frame_prim = UsdGeom.Scope.Define(stage, frame_path).GetPrim()
        frame_prim.CreateAttribute("w2d:frameIndex", Sdf.ValueTypeNames.Int, custom=True).Set(
            int(frame.frame_index)
        )
        frame_prim.CreateAttribute(
            "w2d:timestampSec", Sdf.ValueTypeNames.Double, custom=True
        ).Set(float(frame.timestamp_s))
        frame_prim.CreateAttribute("w2d:imageWidth", Sdf.ValueTypeNames.Int, custom=True).Set(
            int(frame.image_width)
        )
        frame_prim.CreateAttribute("w2d:imageHeight", Sdf.ValueTypeNames.Int, custom=True).Set(
            int(frame.image_height)
        )
        frame_prim.CreateAttribute(
            "w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True
        ).Set(run_id)

        labels: list[str] = []
        class_ids: list[int] = []
        scores: list[float] = []
        boxes_xyxy: list[Any] = []
        det_rel = frame_prim.CreateRelationship("w2d:detections", custom=True)

        for det_index, det in enumerate(frame.detections):
            det_path = f"{frame_path}/det_{det_index:04d}"
            det_prim = UsdGeom.Scope.Define(stage, det_path).GetPrim()
            det_prim.CreateAttribute("w2d:classId", Sdf.ValueTypeNames.Int, custom=True).Set(
                int(det.class_id)
            )
            det_prim.CreateAttribute("w2d:class", Sdf.ValueTypeNames.String, custom=True).Set(
                det.label
            )
            det_prim.CreateAttribute(
                "w2d:confidence", Sdf.ValueTypeNames.Float, custom=True
            ).Set(float(det.confidence))
            det_prim.CreateAttribute(
                "w2d:bboxXYXY", Sdf.ValueTypeNames.Float4, custom=True
            ).Set(
                Gf.Vec4f(
                    float(det.bbox_xyxy[0]),
                    float(det.bbox_xyxy[1]),
                    float(det.bbox_xyxy[2]),
                    float(det.bbox_xyxy[3]),
                )
            )
            det_prim.CreateAttribute(
                "w2d:producedByRunId", Sdf.ValueTypeNames.String, custom=True
            ).Set(run_id)
            det_rel.AddTarget(det_prim.GetPath())

            labels.append(det.label)
            class_ids.append(int(det.class_id))
            scores.append(float(det.confidence))
            boxes_xyxy.append(
                Gf.Vec4f(
                    float(det.bbox_xyxy[0]),
                    float(det.bbox_xyxy[1]),
                    float(det.bbox_xyxy[2]),
                    float(det.bbox_xyxy[3]),
                )
            )
            total_detections += 1

        frame_prim.CreateAttribute(
            "w2d:detectionCount", Sdf.ValueTypeNames.Int, custom=True
        ).Set(len(frame.detections))
        frame_prim.CreateAttribute(
            "w2d:labels", Sdf.ValueTypeNames.StringArray, custom=True
        ).Set(Vt.StringArray(labels))
        frame_prim.CreateAttribute(
            "w2d:classIds", Sdf.ValueTypeNames.IntArray, custom=True
        ).Set(Vt.IntArray(class_ids))
        frame_prim.CreateAttribute(
            "w2d:scores", Sdf.ValueTypeNames.FloatArray, custom=True
        ).Set(Vt.FloatArray(scores))
        frame_prim.CreateAttribute(
            "w2d:boxesXYXY", Sdf.ValueTypeNames.Float4Array, custom=True
        ).Set(Vt.Vec4fArray(boxes_xyxy))
        frame_rel.AddTarget(frame_prim.GetPath())

    yolo_scope.CreateAttribute("w2d:processedFrameCount", Sdf.ValueTypeNames.Int, custom=True).Set(
        len(detections)
    )
    yolo_scope.CreateAttribute("w2d:totalDetectionCount", Sdf.ValueTypeNames.Int, custom=True).Set(
        int(total_detections)
    )

    run_prim = UsdGeom.Scope.Define(
        stage, f"/World/W2D/Provenance/runs/{_safe_prim_name(run_id)}"
    ).GetPrim()
    run_prim.CreateAttribute("w2d:runId", Sdf.ValueTypeNames.String, custom=True).Set(run_id)
    run_prim.CreateAttribute("w2d:component", Sdf.ValueTypeNames.String, custom=True).Set(
        "perception.yolo"
    )
    run_prim.CreateAttribute("w2d:modelName", Sdf.ValueTypeNames.String, custom=True).Set(
        model_name
    )
    run_prim.CreateAttribute("w2d:modelVersion", Sdf.ValueTypeNames.String, custom=True).Set(
        model_version
    )
    run_prim.CreateAttribute("w2d:gitCommit", Sdf.ValueTypeNames.String, custom=True).Set(
        git_commit if git_commit is not None else _git_commit_or_unknown()
    )
    run_prim.CreateAttribute(
        "w2d:timestampIso8601", Sdf.ValueTypeNames.String, custom=True
    ).Set(datetime.now(timezone.utc).isoformat())
    run_prim.CreateAttribute("w2d:params", Sdf.ValueTypeNames.String, custom=True).Set(
        json.dumps(params or {}, sort_keys=True)
    )

    return stage


def yolo_observations_to_usda(
    *,
    detections: Sequence[YoloFrameDetections],
    run_id: str,
    model_name: str,
    model_version: str,
    params: dict[str, Any] | None = None,
    git_commit: str | None = None,
    time_codes_per_second: float = 30.0,
    video_uri: str = "",
) -> str:
    stage = yolo_observations_to_stage(
        detections=detections,
        run_id=run_id,
        model_name=model_name,
        model_version=model_version,
        params=params,
        git_commit=git_commit,
        time_codes_per_second=time_codes_per_second,
        video_uri=video_uri,
    )
    return stage.GetRootLayer().ExportToString()


def write_yolo_observations_usd(
    *,
    detections: Sequence[YoloFrameDetections],
    output_path: Path,
    run_id: str,
    model_name: str,
    model_version: str,
    params: dict[str, Any] | None = None,
    git_commit: str | None = None,
    time_codes_per_second: float = 30.0,
    video_uri: str = "",
) -> None:
    stage = yolo_observations_to_stage(
        detections=detections,
        run_id=run_id,
        model_name=model_name,
        model_version=model_version,
        params=params,
        git_commit=git_commit,
        time_codes_per_second=time_codes_per_second,
        video_uri=video_uri,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(output_path))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO on a video and export a protocol-aligned OpenUSD layer "
            "with raw detections."
        )
    )
    parser.add_argument("--video", type=Path, required=True, help="input video path")
    parser.add_argument(
        "--output-usd",
        type=Path,
        default=Path("outputs/yolo_observations.usda"),
        help="output OpenUSD layer path (.usda/.usdc)",
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path/name")
    parser.add_argument("--device", type=str, default="", help="YOLO device (e.g. cpu, cuda:0)")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--frame-step", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process all frames")
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="run id for provenance; default is UTC timestamp",
    )
    parser.add_argument(
        "--video-uri",
        type=str,
        default="",
        help="optional relative asset URI stored in layer (w2d:videoUri)",
    )
    parser.add_argument(
        "--timecodes-per-second",
        type=float,
        default=0.0,
        help="stage timeCodesPerSecond; 0 means use source video FPS",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    params = {
        "video": str(args.video),
        "model": str(args.model),
        "device": str(args.device),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "frame_step": int(args.frame_step),
        "max_frames": int(args.max_frames),
    }

    yolo_result = run_yolo_on_video(
        video_path=args.video,
        model_name=args.model,
        device=args.device,
        conf_threshold=float(args.conf),
        iou_threshold=float(args.iou),
        frame_step=int(args.frame_step),
        max_frames=int(args.max_frames),
    )
    tps = float(args.timecodes_per_second)
    if tps <= 0.0:
        tps = float(yolo_result.video_fps)

    write_yolo_observations_usd(
        detections=yolo_result.frames,
        output_path=args.output_usd,
        run_id=run_id,
        model_name=yolo_result.model_name,
        model_version=yolo_result.model_version,
        params=params,
        time_codes_per_second=tps,
        video_uri=args.video_uri,
    )

    print(f"wrote yolo observations layer: {args.output_usd}")
    print(f"processed_frames={yolo_result.processed_frames}")
    print(
        "total_detections="
        f"{sum(len(frame.detections) for frame in yolo_result.frames)}"
    )
    print(f"run_id={run_id}")


if __name__ == "__main__":
    main()
