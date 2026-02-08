from __future__ import annotations

"""Live YOLO label viewer.

Shows real-time detections with a side panel listing:
- labels visible in the current frame (with counts)
- labels seen across the whole session

Usage examples:
  python demo/yolo_live_labels/live_labels.py --source 0
  python demo/yolo_live_labels/live_labels.py --source res/video_360p_30fps.mp4 --model yolov8n.pt
"""

import argparse
import time
from collections import Counter

import cv2
from ultralytics import YOLO


def _parse_source(source: str) -> int | str:
    text = source.strip()
    if text.isdigit():
        return int(text)
    return text


def _class_color(class_id: int) -> tuple[int, int, int]:
    # deterministic pseudo-palette in BGR
    return (
        (37 * class_id + 90) % 255,
        (17 * class_id + 140) % 255,
        (29 * class_id + 200) % 255,
    )


def _draw_box(frame, xyxy, label_text: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (label_w, label_h), baseline = cv2.getTextSize(
        label_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1,
    )
    y_label = max(0, y1 - label_h - baseline - 4)
    cv2.rectangle(
        frame,
        (x1, y_label),
        (x1 + label_w + 6, y_label + label_h + baseline + 4),
        color,
        thickness=-1,
    )
    cv2.putText(
        frame,
        label_text,
        (x1 + 3, y_label + label_h + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def _render_panel(
    frame,
    current_counts: Counter[str],
    seen_labels: set[str],
    fps: float,
    panel_width: int,
):
    h, w = frame.shape[:2]
    panel = 255 * (frame[:, :panel_width].copy() * 0 + 1)

    y = 24
    cv2.putText(panel, f"FPS: {fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    y += 26
    cv2.putText(panel, "[q] quit", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

    y += 30
    cv2.putText(panel, "Current frame", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
    y += 20

    if current_counts:
        for label, count in sorted(current_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            cv2.putText(
                panel,
                f"- {label}: {count}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (10, 10, 10),
                1,
                cv2.LINE_AA,
            )
            y += 18
            if y > h - 100:
                break
    else:
        cv2.putText(panel, "- (none)", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
        y += 18

    y += 16
    cv2.putText(panel, "Seen this session", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
    y += 20

    if seen_labels:
        for label in sorted(seen_labels):
            cv2.putText(
                panel,
                f"- {label}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (10, 10, 10),
                1,
                cv2.LINE_AA,
            )
            y += 18
            if y > h - 10:
                break
    else:
        cv2.putText(panel, "- (none)", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)

    canvas = cv2.hconcat([frame, panel])
    return canvas


def run_demo(
    *,
    source: int | str,
    model_path: str,
    conf: float,
    imgsz: int,
    device: str | None,
    window_title: str,
) -> None:
    model = YOLO(model_path)
    seen_labels: set[str] = set()

    last_t = time.perf_counter()

    for result in model(
        source=source,
        stream=True,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    ):
        frame = result.orig_img.copy()

        now = time.perf_counter()
        dt = max(1e-6, now - last_t)
        fps = 1.0 / dt
        last_t = now

        current_counts: Counter[str] = Counter()

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            class_ids = boxes.cls.int().tolist()
            confs = boxes.conf.tolist()
            xyxy_values = boxes.xyxy.tolist()

            for cls_id, conf_score, xyxy in zip(class_ids, confs, xyxy_values, strict=True):
                label = result.names[int(cls_id)]
                seen_labels.add(label)
                current_counts[label] += 1
                color = _class_color(int(cls_id))
                _draw_box(frame, xyxy, f"{label} {conf_score:.2f}", color)

        panel_width = 340
        canvas = _render_panel(frame, current_counts, seen_labels, fps, panel_width)
        cv2.imshow(window_title, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live YOLO label viewer")
    parser.add_argument(
        "--source",
        default="0",
        help="video source: webcam index (e.g. 0) or file path",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path/name")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument(
        "--device",
        default=None,
        help="inference device (e.g. cpu, cuda:0). Default lets Ultralytics choose.",
    )
    parser.add_argument("--window-title", default="YOLO Live Labels", help="OpenCV window title")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_demo(
        source=_parse_source(args.source),
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        window_title=args.window_title,
    )


if __name__ == "__main__":
    main()
