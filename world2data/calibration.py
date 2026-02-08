from __future__ import annotations

"""Minimal ChArUco video calibration tool.

Reads a calibration video, estimates camera intrinsics/distortion, and writes:
- JSON calibration data
- optional USDA stage with a camera prim carrying intrinsic attributes
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

import numpy as np

_SAFE_PRIM_RE = re.compile(r"[^A-Za-z0-9_]")


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "OpenCV is required. Install with `pip install opencv-contrib-python`."
        ) from exc

    if not hasattr(cv2, "aruco"):
        raise ImportError(
            "cv2.aruco is not available. Install with `pip install opencv-contrib-python`."
        )
    return cv2


def _require_pxr() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "OpenUSD Python bindings are required for USDA export. Add `openusd` extras."
        ) from exc
    return Gf, Sdf, Usd, UsdGeom, Vt


def _safe_prim_name(raw: str) -> str:
    value = _SAFE_PRIM_RE.sub("_", raw).strip("_")
    if not value:
        value = "run"
    if value[0].isdigit():
        value = f"n_{value}"
    return value


@dataclass(frozen=True)
class CalibrationResult:
    image_size: tuple[int, int]
    camera_matrix: list[list[float]]
    distortion_coeffs: list[float]
    distortion_model: str
    reprojection_error_px: float
    frames_sampled: int
    frames_used: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": [self.image_size[0], self.image_size[1]],
            "camera_matrix": self.camera_matrix,
            "distortion_coeffs": self.distortion_coeffs,
            "distortion_model": self.distortion_model,
            "reprojection_error_px": self.reprojection_error_px,
            "frames_sampled": self.frames_sampled,
            "frames_used": self.frames_used,
        }


def _resolve_aruco_dictionary(aruco: Any, dictionary_name: str) -> Any:
    normalized = dictionary_name.strip()
    if normalized == "DICT_4X4":
        normalized = "DICT_4X4_50"

    if not normalized.startswith("DICT_"):
        normalized = f"DICT_{normalized}"

    if not hasattr(aruco, normalized):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")

    return aruco.getPredefinedDictionary(getattr(aruco, normalized))


def _create_charuco_board(
    aruco: Any,
    *,
    squares_x: int,
    squares_y: int,
    square_length_m: float,
    marker_length_m: float,
    dictionary: Any,
) -> Any:
    if hasattr(aruco, "CharucoBoard"):
        # OpenCV >= 4.7 API
        return aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length_m,
            marker_length_m,
            dictionary,
        )

    if hasattr(aruco, "CharucoBoard_create"):
        # Older OpenCV API
        return aruco.CharucoBoard_create(
            squares_x,
            squares_y,
            square_length_m,
            marker_length_m,
            dictionary,
        )

    raise RuntimeError("cv2.aruco CharucoBoard API not available")


def _create_detector_parameters(aruco: Any) -> Any:
    if hasattr(aruco, "DetectorParameters"):
        return aruco.DetectorParameters()
    return aruco.DetectorParameters_create()


def _detect_markers(
    aruco: Any,
    gray: np.ndarray,
    dictionary: Any,
    detector_params: Any,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, detector_params)
        corners, ids, _rejected = detector.detectMarkers(gray)
        return corners, ids

    corners, ids, _rejected = aruco.detectMarkers(
        gray,
        dictionary,
        parameters=detector_params,
    )
    return corners, ids


def _detect_charuco(
    *,
    aruco: Any,
    gray: np.ndarray,
    board: Any,
    dictionary: Any,
    detector_params: Any,
) -> tuple[np.ndarray | None, np.ndarray | None, list[np.ndarray], np.ndarray | None]:
    """Return charuco_corners, charuco_ids, marker_corners, marker_ids."""
    marker_corners, marker_ids = _detect_markers(
        aruco,
        gray,
        dictionary,
        detector_params,
    )
    if marker_ids is None or len(marker_ids) == 0:
        return None, None, marker_corners, marker_ids

    if hasattr(aruco, "interpolateCornersCharuco"):
        valid, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board,
        )
        if valid is None or valid <= 0:
            return None, None, marker_corners, marker_ids
        return charuco_corners, charuco_ids, marker_corners, marker_ids

    if hasattr(aruco, "CharucoDetector"):
        detector = aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners_2, marker_ids_2 = detector.detectBoard(
            gray
        )
        if marker_ids_2 is not None and len(marker_ids_2) > 0:
            marker_corners = marker_corners_2
            marker_ids = marker_ids_2
        return charuco_corners, charuco_ids, marker_corners, marker_ids

    raise RuntimeError(
        "No supported ChArUco corner interpolation API found in cv2.aruco"
    )


def infer_distortion_model(distortion_coeffs: list[float]) -> str:
    n = len(distortion_coeffs)
    if n >= 8:
        return "opencv_rational"
    if n >= 5:
        return "opencv_plumb_bob"
    if n == 4:
        return "opencv_fisheye_4"
    return "opencv_unknown"


def calibrate_charuco_video(
    *,
    video_path: Path,
    board_squares_x: int,
    board_squares_y: int,
    square_size_mm: float,
    marker_size_mm: float,
    dictionary_name: str,
    frame_step: int,
    max_frames: int,
    min_charuco_corners: int,
    show: bool,
) -> CalibrationResult:
    cv2 = _require_cv2()
    aruco = cv2.aruco

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")
    if board_squares_x < 2 or board_squares_y < 2:
        raise ValueError("board square counts must be >= 2")
    if marker_size_mm <= 0.0 or square_size_mm <= 0.0:
        raise ValueError("square_size_mm and marker_size_mm must be > 0")
    if marker_size_mm >= square_size_mm:
        raise ValueError("marker_size_mm must be smaller than square_size_mm")

    dictionary = _resolve_aruco_dictionary(aruco, dictionary_name)
    board = _create_charuco_board(
        aruco,
        squares_x=board_squares_x,
        squares_y=board_squares_y,
        square_length_m=square_size_mm / 1000.0,
        marker_length_m=marker_size_mm / 1000.0,
        dictionary=dictionary,
    )
    detector_params = _create_detector_parameters(aruco)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    all_charuco_corners: list[np.ndarray] = []
    all_charuco_ids: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    frames_total = 0
    frames_sampled = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frames_total += 1
            if (frames_total - 1) % frame_step != 0:
                continue
            if max_frames > 0 and frames_sampled >= max_frames:
                break

            frames_sampled += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if image_size is None:
                image_size = (gray.shape[1], gray.shape[0])

            charuco_corners, charuco_ids, marker_corners, marker_ids = _detect_charuco(
                aruco=aruco,
                gray=gray,
                board=board,
                dictionary=dictionary,
                detector_params=detector_params,
            )
            if charuco_corners is None or charuco_ids is None:
                continue
            if len(charuco_ids) < min_charuco_corners:
                continue

            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)

            if show:
                display = frame.copy()
                aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
                aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
                cv2.imshow("Charuco Calibration", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        capture.release()
        if show:
            cv2.destroyAllWindows()

    if image_size is None:
        raise RuntimeError("No readable frames in video")
    if len(all_charuco_corners) < 6:
        raise RuntimeError(
            f"Need at least 6 usable frames, found {len(all_charuco_corners)}"
        )

    if hasattr(aruco, "calibrateCameraCharucoExtended"):
        (
            reprojection_error,
            camera_matrix,
            distortion,
            _rvecs,
            _tvecs,
            _std_intr,
            _std_ext,
            _per_view,
        ) = aruco.calibrateCameraCharucoExtended(
            all_charuco_corners,
            all_charuco_ids,
            board,
            image_size,
            None,
            None,
        )
    elif hasattr(aruco, "calibrateCameraCharuco"):
        (
            reprojection_error,
            camera_matrix,
            distortion,
            _rvecs,
            _tvecs,
        ) = aruco.calibrateCameraCharuco(
            all_charuco_corners,
            all_charuco_ids,
            board,
            image_size,
            None,
            None,
        )
    else:
        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []

        for charuco_corners, charuco_ids in zip(all_charuco_corners, all_charuco_ids, strict=True):
            obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
            if obj_pts is None or img_pts is None:
                continue
            if len(obj_pts) < min_charuco_corners:
                continue
            object_points.append(np.asarray(obj_pts, dtype=np.float32))
            image_points.append(np.asarray(img_pts, dtype=np.float32))

        if len(object_points) < 3:
            raise RuntimeError(
                "Could not build enough ChArUco correspondences for cv2.calibrateCamera"
            )

        reprojection_error, camera_matrix, distortion, _rvecs, _tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )

    distortion_coeffs = np.asarray(distortion, dtype=np.float64).reshape(-1).tolist()

    return CalibrationResult(
        image_size=image_size,
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64).tolist(),
        distortion_coeffs=distortion_coeffs,
        distortion_model=infer_distortion_model(distortion_coeffs),
        reprojection_error_px=float(reprojection_error),
        frames_sampled=frames_sampled,
        frames_used=len(all_charuco_corners),
    )


def write_calibration_usda(
    *,
    result: CalibrationResult,
    output_path: Path,
    camera_prim_path: str = "/World/W2D/Sensors/CalibrationCamera",
    run_id: str,
    model_version: str,
    params: dict[str, Any],
) -> None:
    Gf, Sdf, Usd, UsdGeom, Vt = _require_pxr()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/W2D")
    UsdGeom.Scope.Define(stage, "/World/W2D/Sensors")
    UsdGeom.Scope.Define(stage, "/World/W2D/Provenance")
    UsdGeom.Scope.Define(stage, "/World/W2D/Provenance/runs")

    camera = UsdGeom.Camera.Define(stage, camera_prim_path)
    prim = camera.GetPrim()

    matrix = result.camera_matrix
    prim.CreateAttribute(
        "w2d:intrinsicMatrix",
        Sdf.ValueTypeNames.Matrix3d,
        custom=True,
    ).Set(
        Gf.Matrix3d(
            matrix[0][0], matrix[0][1], matrix[0][2],
            matrix[1][0], matrix[1][1], matrix[1][2],
            matrix[2][0], matrix[2][1], matrix[2][2],
        )
    )
    prim.CreateAttribute("w2d:imageWidth", Sdf.ValueTypeNames.Int, custom=True).Set(
        int(result.image_size[0])
    )
    prim.CreateAttribute("w2d:imageHeight", Sdf.ValueTypeNames.Int, custom=True).Set(
        int(result.image_size[1])
    )
    prim.CreateAttribute("w2d:distortionModel", Sdf.ValueTypeNames.String, custom=True).Set(
        result.distortion_model
    )
    prim.CreateAttribute("w2d:distortionCoeffs", Sdf.ValueTypeNames.FloatArray, custom=True).Set(
        Vt.FloatArray([float(value) for value in result.distortion_coeffs])
    )
    prim.CreateAttribute(
        "w2d:producedByRunId",
        Sdf.ValueTypeNames.String,
        custom=True,
    ).Set(run_id)

    run_prim = UsdGeom.Scope.Define(
        stage, f"/World/W2D/Provenance/runs/{_safe_prim_name(run_id)}"
    ).GetPrim()
    run_prim.CreateAttribute("w2d:runId", Sdf.ValueTypeNames.String, custom=True).Set(run_id)
    run_prim.CreateAttribute("w2d:component", Sdf.ValueTypeNames.String, custom=True).Set(
        "recon.calibration"
    )
    run_prim.CreateAttribute("w2d:modelName", Sdf.ValueTypeNames.String, custom=True).Set(
        "opencv_charuco"
    )
    run_prim.CreateAttribute("w2d:modelVersion", Sdf.ValueTypeNames.String, custom=True).Set(
        model_version
    )
    run_prim.CreateAttribute(
        "w2d:timestampIso8601",
        Sdf.ValueTypeNames.String,
        custom=True,
    ).Set(datetime.now(timezone.utc).isoformat())
    run_prim.CreateAttribute("w2d:params", Sdf.ValueTypeNames.String, custom=True).Set(
        json.dumps(params, sort_keys=True)
    )

    stage.GetRootLayer().Save()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate camera intrinsics/distortion from a ChArUco video"
    )
    parser.add_argument("--video", type=Path, required=True, help="input ChArUco video path")
    parser.add_argument("--board-squares-x", type=int, default=11, help="board squares in X")
    parser.add_argument("--board-squares-y", type=int, default=8, help="board squares in Y")
    parser.add_argument("--square-mm", type=float, default=15.0, help="checker square size in mm")
    parser.add_argument("--marker-mm", type=float, default=11.0, help="aruco marker size in mm")
    parser.add_argument(
        "--dictionary",
        type=str,
        default="DICT_4X4_50",
        help="ArUco dictionary (e.g. DICT_4X4_50). DICT_4X4 alias is accepted.",
    )
    parser.add_argument("--frame-step", type=int, default=2, help="process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no limit")
    parser.add_argument(
        "--min-charuco-corners",
        type=int,
        default=12,
        help="minimum interpolated corners per frame",
    )
    parser.add_argument("--show", action="store_true", help="show detection preview while processing")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/camera_calibration.json"),
        help="output JSON path",
    )
    parser.add_argument(
        "--output-usda",
        type=str,
        default="outputs/camera_calibration.usda",
        help="output USDA path (set empty to disable)",
    )
    parser.add_argument(
        "--camera-prim",
        type=str,
        default="/World/W2D/Sensors/CalibrationCamera",
        help="USD camera prim path",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="provenance run id (defaults to UTC timestamp)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    result = calibrate_charuco_video(
        video_path=args.video,
        board_squares_x=args.board_squares_x,
        board_squares_y=args.board_squares_y,
        square_size_mm=args.square_mm,
        marker_size_mm=args.marker_mm,
        dictionary_name=args.dictionary,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        min_charuco_corners=args.min_charuco_corners,
        show=args.show,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result.to_dict(), indent=2))
    print(f"wrote calibration json: {args.output_json}")
    print(f"frames_used={result.frames_used}/{result.frames_sampled}")
    print(f"reprojection_error_px={result.reprojection_error_px:.4f}")

    if args.output_usda.strip():
        cv2 = _require_cv2()
        run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        params = {
            "video": str(args.video),
            "board_squares_x": int(args.board_squares_x),
            "board_squares_y": int(args.board_squares_y),
            "square_mm": float(args.square_mm),
            "marker_mm": float(args.marker_mm),
            "dictionary": str(args.dictionary),
            "frame_step": int(args.frame_step),
            "max_frames": int(args.max_frames),
            "min_charuco_corners": int(args.min_charuco_corners),
        }
        write_calibration_usda(
            result=result,
            output_path=Path(args.output_usda),
            camera_prim_path=args.camera_prim,
            run_id=run_id,
            model_version=cv2.__version__,
            params=params,
        )
        print(f"wrote calibration usda: {args.output_usda}")


if __name__ == "__main__":
    main()
