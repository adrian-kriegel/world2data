"""Generate a synthetic test video with camera-like motion for pipeline testing.

Creates a scene with textured surfaces and simulates a panning camera,
producing enough visual change to trigger keyframe extraction and provide
MASt3R with features to match across views.
"""
import cv2
import numpy as np
import os


def generate_test_video(output_path="test_video.mp4", width=640, height=480,
                        fps=30, duration_sec=4, seed=42):
    """Generate a synthetic video simulating a camera panning across a textured room.

    The scene contains:
    - A richly textured background (random noise pattern = wall texture)
    - Colored rectangles at different depths (furniture/objects)
    - Simulated lateral camera pan (translating the viewport across a larger canvas)

    This creates real parallax: near objects shift more than far objects.
    """
    rng = np.random.RandomState(seed)
    total_frames = fps * duration_sec

    # Create a large canvas (the "world") that we'll crop from to simulate camera motion
    canvas_w = width * 3
    canvas_h = height * 2

    # Layer 0: Textured background (far wall) - rich random pattern
    bg = rng.randint(60, 180, (canvas_h, canvas_w, 3), dtype=np.uint8)
    # Add some structure: horizontal and vertical lines (like a tiled wall)
    for y in range(0, canvas_h, 40):
        cv2.line(bg, (0, y), (canvas_w, y), (100, 100, 120), 1)
    for x in range(0, canvas_w, 40):
        cv2.line(bg, (x, 0), (x, canvas_h), (100, 100, 120), 1)

    # Layer 1: "Near" objects (colored rectangles with text = more parallax)
    objects = [
        {"pos": (200, 150), "size": (180, 120), "color": (0, 0, 200), "label": "Door", "depth": 0.7},
        {"pos": (700, 200), "size": (140, 200), "color": (0, 180, 0), "label": "Cabinet", "depth": 0.5},
        {"pos": (1200, 100), "size": (200, 160), "color": (200, 100, 0), "label": "Window", "depth": 0.8},
        {"pos": (400, 400), "size": (100, 80), "color": (0, 150, 200), "label": "Cup", "depth": 0.3},
        {"pos": (1000, 350), "size": (160, 100), "color": (180, 0, 180), "label": "Table", "depth": 0.4},
    ]

    # Camera path: smooth lateral pan from left to right
    # Start at x_offset=0, end at x_offset = canvas_w - width
    max_x_offset = canvas_w - width
    max_y_offset = canvas_h - height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer at {output_path}")

    for frame_idx in range(total_frames):
        t = frame_idx / max(total_frames - 1, 1)  # 0 to 1

        # Camera position: smooth pan with slight vertical wobble
        cam_x = int(t * max_x_offset * 0.6)  # don't use full range
        cam_y = int(max_y_offset * 0.25 + 20 * np.sin(t * 2 * np.pi))

        # Start with background crop (far layer, moves with camera)
        frame = bg[cam_y:cam_y + height, cam_x:cam_x + width].copy()

        # Draw objects with parallax (near objects shift more relative to camera)
        for obj in objects:
            depth = obj["depth"]
            # Parallax: near objects (low depth) shift MORE opposite to camera
            parallax_factor = 1.0 / (depth + 0.1)
            ox = int(obj["pos"][0] - cam_x * parallax_factor * 0.3)
            oy = int(obj["pos"][1] - cam_y * parallax_factor * 0.2)
            w, h = obj["size"]

            # Only draw if within frame
            if -w < ox < width and -h < oy < height:
                x1 = max(0, ox)
                y1 = max(0, oy)
                x2 = min(width, ox + w)
                y2 = min(height, oy + h)
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), obj["color"], -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    # Add label text
                    tx = x1 + 5
                    ty = y1 + 25
                    if tx < x2 - 10 and ty < y2:
                        cv2.putText(frame, obj["label"], (tx, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add frame counter (helps verify extraction)
        cv2.putText(frame, f"Frame {frame_idx}/{total_frames}",
                    (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        writer.write(frame)

    writer.release()
    print(f"Generated test video: {output_path} ({total_frames} frames, {width}x{height}, {fps}fps)")
    return output_path


if __name__ == "__main__":
    generate_test_video()
