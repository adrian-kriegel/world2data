"""World2Data 4D Scene Fusion Layer.

Connects all model outputs (MASt3R depth, YOLOv8 detection, SAM3 segmentation,
Gemini scene analysis, and reasoning validation) into a unified 4D scene graph
where every object is tracked through both space (3D) and time (video frames).

The fusion layer:
1. Groups YOLO detections by class across frames
2. Matches with SAM3 tracked objects (by 2D IoU)
3. Projects 2D regions to 3D using MASt3R depth + poses
4. Matches with Gemini scene objects (by name/type)
5. Computes confidence from cross-model agreement
6. Flags uncertain items for human-in-the-loop review
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("world2data.fusion")


# =========================================================================
# Core 4D Data Structure
# =========================================================================

@dataclass
class SceneObject4D:
    """A single object tracked through space and time (4D)."""

    # Identity
    entity_id: str              # "Table_01"
    obj_type: str               # "table"
    canonical_name: str = ""    # best name from cross-model consensus

    # Confidence & provenance
    confidence: float = 0.0     # 0.0-1.0 (cross-model agreement)
    detected_by: list = field(default_factory=list)  # ["yolo", "sam3", "gemini"]

    # Spatial (3D bounding box in world coordinates)
    bbox_3d_min: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    bbox_3d_max: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    mask_3d_points: Optional[np.ndarray] = None  # (M, 3) points belonging to this object

    # Temporal (4D: tracked across frames)
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    frame_detections: dict = field(default_factory=dict)  # {frame_idx: DetectionResult}
    frame_masks: dict = field(default_factory=dict)        # {frame_idx: mask_2d}

    # Physics
    component_type: str = "FixedJoint"
    initial_state: str = "unknown"
    final_state: str = "unknown"
    state_changes: list = field(default_factory=list)

    # Human-in-the-loop
    human_review: bool = False
    review_reason: str = ""
    reasoning_trace: str = ""

    @property
    def center(self) -> np.ndarray:
        return (self.bbox_3d_min + self.bbox_3d_max) / 2

    @property
    def size(self) -> np.ndarray:
        return self.bbox_3d_max - self.bbox_3d_min

    @property
    def frame_count(self) -> int:
        """Number of frames this object was detected in."""
        return len(self.frame_detections) + len(self.frame_masks)

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "obj_type": self.obj_type,
            "canonical_name": self.canonical_name or self.entity_id,
            "confidence": round(self.confidence, 3),
            "detected_by": self.detected_by,
            "bbox_3d_min": self.bbox_3d_min.tolist(),
            "bbox_3d_max": self.bbox_3d_max.tolist(),
            "center": self.center.tolist(),
            "size": self.size.tolist(),
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "frame_count": self.frame_count,
            "component_type": self.component_type,
            "initial_state": self.initial_state,
            "final_state": self.final_state,
            "state_changes": self.state_changes,
            "human_review": self.human_review,
            "review_reason": self.review_reason,
            "reasoning_trace": self.reasoning_trace,
        }


# =========================================================================
# Scene Fusion Engine
# =========================================================================

class SceneFusion4D:
    """Fuses all model outputs into a unified 4D scene graph."""

    # Confidence weights
    YOLO_WEIGHT = 0.40
    SAM3_WEIGHT = 0.30
    GEMINI_WEIGHT = 0.30

    # Thresholds
    HUMAN_REVIEW_CONFIDENCE = 0.5
    MIN_FRAME_RATIO = 0.20  # object must appear in 20% of frames

    def __init__(self, frame_data: list = None,
                 detections: list = None,
                 segmentations: list = None,
                 scene_description=None,
                 reasoning=None):
        """
        Args:
            frame_data: list of FrameData from MASt3R (pipeline_controller)
            detections: list of DetectionResult from YOLOv8
            segmentations: list of SegmentationResult from SAM3
            scene_description: SceneDescription from Gemini
            reasoning: ReasoningResult from reasoning engine
        """
        self.frame_data = frame_data or []
        self.detections = detections or []
        self.segmentations = segmentations or []
        self.scene_description = scene_description
        self.reasoning = reasoning

        self.objects_4d = []  # output: list[SceneObject4D]

    def fuse(self) -> list:
        """Main fusion algorithm. Returns list of SceneObject4D."""
        print("--- 4D Scene Fusion ---")

        # Step 1: Group YOLO detections by class across frames
        yolo_groups = self._group_yolo_detections()
        print(f"  YOLO: {len(yolo_groups)} unique classes detected")

        # Step 2: Match YOLO groups with SAM3 tracked objects
        sam3_matches = self._match_sam3_to_yolo(yolo_groups)
        print(f"  SAM3: {len(sam3_matches)} matched segments")

        # Step 3: Create initial SceneObject4D from YOLO groups
        self.objects_4d = []
        for class_name, group in yolo_groups.items():
            # May have multiple instances of same class
            instances = self._split_into_instances(class_name, group)
            for idx, instance in enumerate(instances):
                entity_id = f"{class_name.replace(' ', '_')}_{idx + 1:02d}"
                obj = SceneObject4D(
                    entity_id=entity_id,
                    obj_type=class_name,
                    detected_by=["yolo"],
                    first_seen_frame=instance["first_frame"],
                    last_seen_frame=instance["last_frame"],
                    frame_detections=instance["detections"],
                )

                # Add SAM3 info if matched
                if class_name in sam3_matches:
                    obj.detected_by.append("sam3")
                    obj.frame_masks = sam3_matches[class_name].get("masks", {})

                self.objects_4d.append(obj)

        # Step 4: Project 2D detections to 3D using MASt3R depth
        self._project_all_to_3d()
        print(f"  3D projection: {sum(1 for o in self.objects_4d if np.any(o.bbox_3d_max != 0))} objects positioned")

        # Step 5: Match with Gemini scene objects
        self._match_gemini_objects()

        # Step 6: Apply reasoning results
        self._apply_reasoning()

        # Step 7: Compute confidence scores
        self._compute_confidences()
        print(f"  Confidence: avg={np.mean([o.confidence for o in self.objects_4d]):.2f}")

        # Step 8: Flag for human review
        self._flag_for_review()
        n_flagged = sum(1 for o in self.objects_4d if o.human_review)
        print(f"  Human review: {n_flagged}/{len(self.objects_4d)} objects flagged")

        print(f"  FUSION COMPLETE: {len(self.objects_4d)} objects in 4D scene graph")
        return self.objects_4d

    # -----------------------------------------------------------------
    # Step 1: Group YOLO detections
    # -----------------------------------------------------------------
    def _group_yolo_detections(self) -> dict:
        """Group YOLO detections by class name across all frames."""
        groups = {}
        for det in self.detections:
            for i, name in enumerate(det.class_names):
                if name not in groups:
                    groups[name] = []
                box = det.boxes[i] if i < len(det.boxes) else None
                score = float(det.scores[i]) if i < len(det.scores) else 0.0
                groups[name].append({
                    "frame_idx": det.frame_idx,
                    "timestamp": det.timestamp,
                    "box": box,
                    "score": score,
                    "mask_idx": i,
                })
        return groups

    # -----------------------------------------------------------------
    # Step 2: Match SAM3 to YOLO
    # -----------------------------------------------------------------
    def _match_sam3_to_yolo(self, yolo_groups: dict) -> dict:
        """Match SAM3 segmentation results to YOLO detection groups."""
        matches = {}
        for seg in self.segmentations:
            for i, label in enumerate(seg.labels):
                # Direct label match
                if label in yolo_groups:
                    if label not in matches:
                        matches[label] = {"masks": {}, "object_ids": set()}
                    if i < len(seg.masks):
                        matches[label]["masks"][seg.frame_idx] = seg.masks[i]
                    if i < len(seg.object_ids):
                        matches[label]["object_ids"].add(seg.object_ids[i])
        return matches

    # -----------------------------------------------------------------
    # Step 3: Split detections into instances
    # -----------------------------------------------------------------
    def _split_into_instances(self, class_name: str, group: list) -> list:
        """Split a group of same-class detections into separate object instances.

        Uses spatial clustering: if two detections in the same frame are far apart,
        they're different instances.
        """
        if not group:
            return []

        # Simple approach: for now, treat as one instance per class
        # TODO: implement spatial clustering for multi-instance detection
        detections = {item["frame_idx"]: item for item in group}
        frames = sorted(detections.keys())

        return [{
            "first_frame": frames[0],
            "last_frame": frames[-1],
            "detections": detections,
        }]

    # -----------------------------------------------------------------
    # Step 4: Project to 3D
    # -----------------------------------------------------------------
    def _project_all_to_3d(self):
        """Project 2D detections to 3D for all objects using MASt3R depth."""
        if not self.frame_data:
            return

        # Build lookup: frame_idx -> FrameData
        fd_map = {}
        for fd in self.frame_data:
            fd_map[fd.index] = fd

        for obj in self.objects_4d:
            all_3d_points = []

            for frame_idx, det_info in obj.frame_detections.items():
                if frame_idx not in fd_map:
                    # Try to find closest keyframe
                    closest = min(fd_map.keys(),
                                  key=lambda k: abs(k - frame_idx),
                                  default=None)
                    if closest is None:
                        continue
                    fd = fd_map[closest]
                else:
                    fd = fd_map[frame_idx]

                box = det_info.get("box")
                if box is None:
                    continue

                # Check if we have a SAM3 mask for this frame
                mask_2d = obj.frame_masks.get(frame_idx)

                pts_3d = self._project_region_to_3d(fd, box, mask_2d)
                if pts_3d is not None and len(pts_3d) > 0:
                    all_3d_points.append(pts_3d)

            if all_3d_points:
                combined = np.concatenate(all_3d_points, axis=0)
                obj.mask_3d_points = combined
                # Robust bounding box (5-95 percentile)
                obj.bbox_3d_min = np.percentile(combined, 5, axis=0).astype(np.float32)
                obj.bbox_3d_max = np.percentile(combined, 95, axis=0).astype(np.float32)

    def _project_region_to_3d(self, frame_data, box_xyxy, mask_2d=None):
        """Project a 2D region (box or mask) to 3D points using MASt3R depth.

        Args:
            frame_data: FrameData with pts3d, pose, focal, principal_point
            box_xyxy: [x1, y1, x2, y2] pixel coordinates
            mask_2d: optional (H, W) binary mask for precise segmentation

        Returns:
            (N, 3) array of 3D points within the region
        """
        if frame_data.pts3d.shape[0] == 0:
            return None

        h, w = frame_data.image_rgb.shape[:2]
        pts_world = frame_data.pts3d  # (N, 3)

        # Project 3D points to 2D
        pose_inv = np.linalg.inv(frame_data.pose)
        pts_cam = (pose_inv[:3, :3] @ pts_world.T).T + pose_inv[:3, 3]

        valid_z = pts_cam[:, 2] > 0.01
        if not np.any(valid_z):
            return None

        pts_cam_valid = pts_cam[valid_z]
        pts_world_valid = pts_world[valid_z]

        fx = frame_data.focal
        fy = frame_data.focal
        cx, cy = frame_data.principal_point

        u = fx * pts_cam_valid[:, 0] / pts_cam_valid[:, 2] + cx
        v = fy * pts_cam_valid[:, 1] / pts_cam_valid[:, 2] + cy

        if box_xyxy is not None:
            x1, y1, x2, y2 = box_xyxy
            in_box = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        else:
            in_box = np.ones(len(u), dtype=bool)

        # If we have a mask, use it for more precise selection
        if mask_2d is not None:
            u_int = np.clip(u.astype(int), 0, w - 1)
            v_int = np.clip(v.astype(int), 0, h - 1)
            in_mask = mask_2d[v_int, u_int] > 0
            selection = in_box & in_mask
        else:
            selection = in_box

        if np.sum(selection) < 3:
            return None

        return pts_world_valid[selection]

    # -----------------------------------------------------------------
    # Step 5: Match Gemini scene objects
    # -----------------------------------------------------------------
    def _match_gemini_objects(self):
        """Match Gemini scene description objects to our 4D objects."""
        if not self.scene_description or not self.scene_description.objects:
            return

        gemini_objects = self.scene_description.objects
        matched = 0

        for obj in self.objects_4d:
            for g_obj in gemini_objects:
                g_name = g_obj.get("name", "").lower().replace("_", " ")
                g_type = g_obj.get("type", "").lower()
                obj_type = obj.obj_type.lower()

                # Match by type
                if obj_type in g_name or obj_type == g_type or g_type in obj_type:
                    if "gemini" not in obj.detected_by:
                        obj.detected_by.append("gemini")
                    if g_obj.get("description"):
                        obj.canonical_name = g_obj.get("name", obj.entity_id)
                    matched += 1
                    break

        logger.info(f"Gemini matched {matched}/{len(self.objects_4d)} objects")

    # -----------------------------------------------------------------
    # Step 6: Apply reasoning results
    # -----------------------------------------------------------------
    def _apply_reasoning(self):
        """Apply reasoning results to 4D objects."""
        if not self.reasoning:
            return

        # Build lookup by entity
        reasoning_map = {}
        for v_obj in self.reasoning.verified_objects:
            key = v_obj.get("entity", "").lower().replace("_", " ")
            reasoning_map[key] = v_obj
            # Also by type
            otype = v_obj.get("type", "")
            if otype:
                reasoning_map[otype.lower()] = v_obj

        for obj in self.objects_4d:
            r = reasoning_map.get(obj.entity_id.lower().replace("_", " "))
            if not r:
                r = reasoning_map.get(obj.obj_type.lower())
            if r:
                obj.component_type = r.get("component_type", obj.component_type)
                obj.reasoning_trace = r.get("reasoning_trace", "")
                if r.get("human_review"):
                    obj.human_review = True
                    obj.review_reason = r.get("review_reason", "Reasoning model flagged")

        # Apply state changes
        for sc in self.reasoning.state_changes:
            entity = sc.get("entity", "").lower().replace("_", " ")
            for obj in self.objects_4d:
                if obj.entity_id.lower().replace("_", " ") == entity or \
                   obj.obj_type.lower() == sc.get("type", "").lower():
                    obj.state_changes.append(sc)
                    obj.initial_state = sc.get("from_state", obj.initial_state)
                    obj.final_state = sc.get("to_state", obj.final_state)
                    break

    # -----------------------------------------------------------------
    # Step 7: Compute confidence
    # -----------------------------------------------------------------
    def _compute_confidences(self):
        """Compute confidence score based on cross-model agreement."""
        total_frames = len(self.detections) if self.detections else 1

        for obj in self.objects_4d:
            # YOLO score: average detection confidence
            yolo_score = 0.0
            if obj.frame_detections:
                scores = [d.get("score", 0) for d in obj.frame_detections.values()]
                yolo_score = np.mean(scores) if scores else 0.0

            # SAM3 score: has tracked masks?
            sam3_score = 1.0 if "sam3" in obj.detected_by else 0.0

            # Gemini score: mentioned in scene description?
            gemini_score = 1.0 if "gemini" in obj.detected_by else 0.0

            # Weighted confidence
            obj.confidence = (
                self.YOLO_WEIGHT * yolo_score
                + self.SAM3_WEIGHT * sam3_score
                + self.GEMINI_WEIGHT * gemini_score
            )

            # Bonus for temporal consistency
            frame_ratio = obj.frame_count / max(total_frames, 1)
            if frame_ratio > 0.5:
                obj.confidence = min(1.0, obj.confidence * 1.1)

    # -----------------------------------------------------------------
    # Step 8: Flag for human review
    # -----------------------------------------------------------------
    def _flag_for_review(self):
        """Flag objects needing human review."""
        total_frames = len(self.detections) if self.detections else 1

        for obj in self.objects_4d:
            reasons = []

            # Low confidence
            if obj.confidence < self.HUMAN_REVIEW_CONFIDENCE:
                reasons.append(f"Low confidence ({obj.confidence:.2f})")

            # Models disagree on type
            if len(obj.detected_by) == 1:
                reasons.append(f"Only detected by {obj.detected_by[0]}")

            # State change on fixed joint (physics inconsistency)
            if obj.state_changes and obj.component_type == "FixedJoint":
                reasons.append("State change on FixedJoint (physics inconsistency)")

            # Low frame coverage
            frame_ratio = obj.frame_count / max(total_frames, 1)
            if frame_ratio < self.MIN_FRAME_RATIO and total_frames > 5:
                reasons.append(f"Low frame coverage ({frame_ratio:.0%})")

            if reasons:
                obj.human_review = True
                obj.review_reason = "; ".join(reasons)

    # -----------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------
    def to_dict(self) -> dict:
        """Export the full 4D scene graph as a dictionary."""
        return {
            "objects_4d": [obj.to_dict() for obj in self.objects_4d],
            "num_objects": len(self.objects_4d),
            "num_verified": sum(
                1 for o in self.objects_4d if o.confidence >= 0.5
            ),
            "num_flagged": sum(1 for o in self.objects_4d if o.human_review),
            "avg_confidence": float(np.mean(
                [o.confidence for o in self.objects_4d]
            )) if self.objects_4d else 0.0,
            "scene_narrative": (
                self.scene_description.narrative
                if self.scene_description else ""
            ),
        }

    def to_json(self, path: str):
        """Export to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  Saved 4D scene graph -> {path}")


# =========================================================================
# Ground Truth Evaluator (product foundation)
# =========================================================================

class GroundTruthEvaluator:
    """Evaluates scene graph quality and improves over time.

    This is the product foundation -- the cost/evaluator function that makes
    the pipeline a sellable framework. It learns from human feedback.
    """

    def __init__(self, feedback_db_path: str = "ground_truth_feedback.json"):
        self.feedback_db_path = feedback_db_path
        self.feedback_db = self._load_feedback()
        self.known_objects = self.feedback_db.get("known_objects", {})
        self.correction_history = self.feedback_db.get("corrections", [])

    def _load_feedback(self) -> dict:
        """Load accumulated human feedback from disk."""
        try:
            with open(self.feedback_db_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"known_objects": {}, "corrections": [], "runs": 0}

    def _save_feedback(self):
        """Persist feedback to disk."""
        with open(self.feedback_db_path, "w") as f:
            json.dump(self.feedback_db, f, indent=2)

    def evaluate(self, objects_4d: list) -> dict:
        """Score the scene graph quality.

        Returns evaluation metrics that can be tracked over time.
        """
        if not objects_4d:
            return {"overall_score": 0.0, "metrics": {}, "suggestions": []}

        confidences = [o.confidence for o in objects_4d]
        n_review = sum(1 for o in objects_4d if o.human_review)
        n_multi_model = sum(1 for o in objects_4d if len(o.detected_by) >= 2)
        n_with_state = sum(1 for o in objects_4d if o.state_changes)
        n_with_3d = sum(1 for o in objects_4d if np.any(o.bbox_3d_max != 0))

        metrics = {
            "cross_model_agreement": n_multi_model / max(len(objects_4d), 1),
            "temporal_consistency": 1.0 - (n_review / max(len(objects_4d), 1)),
            "physics_plausibility": 1.0 - (
                sum(1 for o in objects_4d
                    if o.state_changes and o.component_type == "FixedJoint")
                / max(len(objects_4d), 1)
            ),
            "spatial_coverage": n_with_3d / max(len(objects_4d), 1),
            "avg_confidence": float(np.mean(confidences)),
        }

        overall = float(np.mean(list(metrics.values())))

        suggestions = []
        if metrics["cross_model_agreement"] < 0.5:
            suggestions.append(
                "Low cross-model agreement. Consider running more detection models."
            )
        if metrics["spatial_coverage"] < 0.5:
            suggestions.append(
                "Low 3D coverage. MASt3R may need more keyframes or better parallax."
            )
        if metrics["avg_confidence"] < 0.5:
            suggestions.append(
                "Low average confidence. Human review recommended for flagged objects."
            )

        return {
            "overall_score": round(overall, 3),
            "metrics": {k: round(v, 3) for k, v in metrics.items()},
            "improvement_suggestions": suggestions,
            "num_objects": len(objects_4d),
            "num_flagged": n_review,
            "num_with_state_changes": n_with_state,
        }

    def flag_for_review(self, objects_4d: list) -> list:
        """Return items that need human correction."""
        return [
            {
                "entity_id": o.entity_id,
                "obj_type": o.obj_type,
                "confidence": o.confidence,
                "review_reason": o.review_reason,
                "detected_by": o.detected_by,
            }
            for o in objects_4d if o.human_review
        ]

    def incorporate_feedback(self, entity_id: str, correction: dict):
        """Store human correction for future improvement.

        Args:
            entity_id: the object being corrected
            correction: dict with keys like "correct_type", "correct_state", etc.
        """
        self.correction_history.append({
            "entity_id": entity_id,
            "correction": correction,
        })

        # Update known objects vocabulary
        if "correct_type" in correction:
            obj_type = correction["correct_type"]
            if obj_type not in self.known_objects:
                self.known_objects[obj_type] = {"count": 0, "corrections": 0}
            self.known_objects[obj_type]["count"] += 1
            self.known_objects[obj_type]["corrections"] += 1

        self.feedback_db["known_objects"] = self.known_objects
        self.feedback_db["corrections"] = self.correction_history
        self.feedback_db["runs"] = self.feedback_db.get("runs", 0) + 1
        self._save_feedback()
        logger.info(f"Incorporated feedback for {entity_id}")

    def get_known_objects(self) -> dict:
        """Return vocabulary of objects learned from past corrections."""
        return dict(self.known_objects)


# =========================================================================
# Video Annotator (overlay model outputs on video frames)
# =========================================================================

class VideoAnnotator:
    """Render model outputs as visual overlays on video frames.

    Creates annotated video for investor presentations and demos.
    """

    # Color palette for object types
    TYPE_COLORS = {
        "person": (255, 140, 0),
        "chair": (0, 200, 200),
        "table": (40, 80, 220),
        "door": (220, 40, 40),
        "cup": (200, 200, 40),
        "bottle": (200, 100, 255),
        "window": (200, 100, 40),
        "cabinet": (40, 200, 40),
        "tv": (180, 0, 180),
        "laptop": (0, 180, 180),
        "phone": (255, 80, 80),
        "book": (100, 200, 100),
    }
    DEFAULT_COLOR = (180, 180, 180)

    @classmethod
    def annotate_frame(cls, frame_bgr: np.ndarray,
                       detections=None,
                       segmentations=None,
                       objects_4d=None,
                       show_confidence: bool = True) -> np.ndarray:
        """Draw YOLO boxes, SAM3 masks, labels, and confidence on a frame."""
        import cv2
        annotated = frame_bgr.copy()

        # Draw SAM3 masks (semi-transparent overlay)
        if segmentations:
            for i, mask in enumerate(segmentations.masks if hasattr(segmentations, 'masks') else []):
                if mask is None:
                    continue
                label = segmentations.labels[i] if i < len(segmentations.labels) else "object"
                color = cls.TYPE_COLORS.get(label.lower(), cls.DEFAULT_COLOR)

                # Resize mask to frame size if needed
                h, w = frame_bgr.shape[:2]
                if mask.shape[:2] != (h, w):
                    mask_resized = cv2.resize(mask.astype(np.uint8), (w, h))
                else:
                    mask_resized = mask

                # Semi-transparent overlay
                overlay = annotated.copy()
                overlay[mask_resized > 0] = color
                annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)

        # Draw YOLO bounding boxes
        if detections:
            boxes = detections.boxes if hasattr(detections, 'boxes') else []
            for i in range(len(boxes)):
                if len(boxes) == 0:
                    break
                box = boxes[i]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                name = detections.class_names[i] if i < len(detections.class_names) else "?"
                score = float(detections.scores[i]) if i < len(detections.scores) else 0

                color = cls.TYPE_COLORS.get(name.lower(), cls.DEFAULT_COLOR)

                # Box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Label background
                label = f"{name}"
                if show_confidence:
                    label += f" {score:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated

    @classmethod
    def create_annotated_video(cls, video_path: str,
                               all_detections: list,
                               all_segmentations: list = None,
                               objects_4d: list = None,
                               output_path: str = "annotated_output.mp4",
                               sample_fps: float = 5.0) -> str:
        """Write an annotated video file with all overlays."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_step = max(1, int(video_fps / sample_fps))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, sample_fps, (w, h))

        # Build detection lookup by frame_idx
        det_map = {d.frame_idx: d for d in all_detections}
        seg_map = {}
        if all_segmentations:
            seg_map = {s.frame_idx: s for s in all_segmentations}

        frame_idx = 0
        written = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_step == 0:
                det = det_map.get(frame_idx)
                seg = seg_map.get(frame_idx)
                annotated = cls.annotate_frame(frame, det, seg, objects_4d)
                out.write(annotated)
                written += 1
            frame_idx += 1

        cap.release()
        out.release()
        print(f"  Annotated video: {output_path} ({written} frames)")
        return output_path

    @classmethod
    def log_to_rerun(cls, frame_rgb: np.ndarray, detections=None,
                     segmentations=None, frame_idx: int = 0):
        """Log annotated frame to Rerun as input/annotated panel."""
        import rerun as rr

        # Annotate
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        annotated_bgr = cls.annotate_frame(frame_bgr, detections, segmentations)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        rr.log("input/annotated", rr.Image(annotated_rgb))
