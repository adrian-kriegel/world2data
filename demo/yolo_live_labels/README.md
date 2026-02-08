# YOLO Live Label Demo

Minimal live detector demo that shows:
- bounding boxes + class labels in the frame
- a live side panel with labels in the current frame
- a running list of all labels seen in this session

## Run

From repo root:

```bash
python demo/yolo_live_labels/live_labels.py --source 0
```

Video file example:

```bash
python demo/yolo_live_labels/live_labels.py --source res/video_360p_30fps.mp4
```

Use a larger model if needed:

```bash
python demo/yolo_live_labels/live_labels.py --model yolov8x.pt --source 0
```

## Controls

- `q` or `Esc`: quit

## Notes

- Requires `ultralytics` and OpenCV in your environment.
- First run may download YOLO weights automatically.
