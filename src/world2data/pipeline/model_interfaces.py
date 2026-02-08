"""World2Data Model Interfaces -- Clean 4D interfaces for all AI models.

Each interface produces timestamped per-frame results that can be fused
together in the 4D SceneFusion layer.

Interfaces:
  1. DepthGeometryInterface  -- MASt3R (already in pipeline_controller.py)
  2. YOLODetector            -- YOLOv8 object detection + instance segmentation
  3. SAM3Segmenter           -- SAM3 pixel-perfect segmentation + video tracking
  4. GeminiVideoAnalyzer     -- Gemini full-video scene understanding
  5. ReasoningEngine         -- Cross-model validation + confidence scoring
"""
import os
import sys
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path

import numpy as np
import cv2

# Load .env from project root (src/world2data/pipeline/ -> three levels up)
from dotenv import load_dotenv
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger("world2data.interfaces")

# ---------------------------------------------------------------------------
# Optional imports with graceful fallback
# ---------------------------------------------------------------------------
_HAS_YOLO = False
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    logger.warning("ultralytics not installed. YOLODetector unavailable.")

_HAS_SAM3 = False
try:
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    import torch
    _HAS_SAM3 = True
except ImportError:
    logger.warning("transformers SAM3 not available. SAM3Segmenter unavailable.")

_HAS_GEMINI = False
try:
    from google import genai
    _HAS_GEMINI = True
except ImportError:
    logger.warning("google-genai not installed. GeminiVideoAnalyzer unavailable.")


# =========================================================================
# Data Structures (4D: space + time)
# =========================================================================

@dataclass
class FrameResult:
    """Base result for any per-frame model output."""
    frame_idx: int
    timestamp: float  # seconds into video

    def to_dict(self) -> dict:
        return {"frame_idx": self.frame_idx, "timestamp": self.timestamp}


@dataclass
class DetectionResult(FrameResult):
    """Per-frame object detection output (YOLOv8)."""
    boxes: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))  # (N, 4) xyxy
    classes: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))
    class_names: list = field(default_factory=list)
    scores: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    masks: Optional[np.ndarray] = None  # (N, H, W) instance masks if seg model

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "num_detections": len(self.class_names),
            "class_names": self.class_names,
            "scores": self.scores.tolist() if len(self.scores) > 0 else [],
            "boxes": self.boxes.tolist() if len(self.boxes) > 0 else [],
        })
        return d


@dataclass
class SegmentationResult(FrameResult):
    """Per-frame segmentation output (SAM3)."""
    masks: list = field(default_factory=list)       # list of (H, W) binary masks
    labels: list = field(default_factory=list)       # "table", "chair", etc.
    scores: list = field(default_factory=list)       # confidence per mask
    object_ids: list = field(default_factory=list)   # persistent IDs across frames

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "num_segments": len(self.labels),
            "labels": self.labels,
            "scores": self.scores,
            "object_ids": self.object_ids,
        })
        return d


@dataclass
class SceneDescription:
    """Holistic scene understanding from full video analysis (Gemini)."""
    objects: list = field(default_factory=list)              # [{name, type, ...}]
    events: list = field(default_factory=list)               # [{time, description, ...}]
    narrative: str = ""                                       # free-text scene description
    spatial_relations: list = field(default_factory=list)     # [{obj_a, relation, obj_b}]
    raw_response: str = ""                                    # raw model output

    def to_dict(self) -> dict:
        return {
            "objects": self.objects,
            "events": self.events,
            "narrative": self.narrative,
            "spatial_relations": self.spatial_relations,
        }


@dataclass
class ReasoningResult:
    """Cross-model validation and confidence scoring output."""
    verified_objects: list = field(default_factory=list)      # confirmed objects
    confidence_scores: dict = field(default_factory=dict)     # {entity: float}
    state_changes: list = field(default_factory=list)         # validated state changes
    human_review_flags: list = field(default_factory=list)    # items needing review
    reasoning_trace: str = ""                                 # step-by-step chain

    def to_dict(self) -> dict:
        return {
            "verified_objects": self.verified_objects,
            "confidence_scores": self.confidence_scores,
            "state_changes": self.state_changes,
            "human_review_flags": self.human_review_flags,
            "reasoning_trace": self.reasoning_trace,
        }


# =========================================================================
# Interface 1: YOLOv8 Object Detection
# =========================================================================

class YOLODetector:
    """Fast object detection + instance segmentation via YOLOv8.

    Usage:
        detector = YOLODetector(model_name="yolov8x-seg.pt")
        result = detector.process_frame(frame_bgr, frame_idx=0, timestamp=0.0)
        results = detector.process_video("video.mp4", sample_fps=5)
    """

    def __init__(self, model_name: str = "yolov8x-seg.pt", device: str = "cuda",
                 conf_threshold: float = 0.25):
        if not _HAS_YOLO:
            raise ImportError("ultralytics not installed. Run: uv add ultralytics")
        # Resolve model path: look in data/models/ first, then fall back to name
        # (ultralytics auto-downloads if a bare name like "yolov8x-seg.pt" is given)
        _local = _PROJECT_ROOT / "data" / "models" / model_name
        model_path = str(_local) if _local.is_file() else model_name
        self.model = YOLO(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.model_name = model_name
        logger.info(f"YOLODetector loaded: {model_name} on {device}")

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int = 0,
                      timestamp: float = 0.0) -> DetectionResult:
        """Run YOLOv8 on a single BGR frame."""
        results = self.model(frame_bgr, device=self.device, verbose=False,
                             conf=self.conf_threshold)
        r = results[0]

        boxes = np.empty((0, 4))
        classes = np.empty((0,), dtype=int)
        class_names = []
        scores = np.empty((0,))
        masks = None

        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()          # (N, 4)
            classes = r.boxes.cls.cpu().numpy().astype(int)  # (N,)
            scores = r.boxes.conf.cpu().numpy()         # (N,)
            class_names = [r.names[int(c)] for c in classes]

        if r.masks is not None and len(r.masks) > 0:
            masks = r.masks.data.cpu().numpy()  # (N, H, W)

        return DetectionResult(
            frame_idx=frame_idx,
            timestamp=timestamp,
            boxes=boxes,
            classes=classes,
            class_names=class_names,
            scores=scores,
            masks=masks,
        )

    def process_video(self, video_path: str, sample_fps: float = 5.0,
                      max_frames: int = 500) -> list:
        """Run YOLOv8 on video, sampling at given FPS."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_step = max(1, int(video_fps / sample_fps))
        results = []
        frame_idx = 0

        while cap.isOpened() and len(results) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_step == 0:
                timestamp = frame_idx / video_fps
                det = self.process_frame(frame, frame_idx, timestamp)
                results.append(det)
            frame_idx += 1

        cap.release()
        logger.info(f"YOLOv8 processed {len(results)} frames from {video_path}")
        return results

    def get_unique_classes(self, results: list) -> list:
        """Get all unique class names detected across all frames."""
        classes = set()
        for r in results:
            classes.update(r.class_names)
        return sorted(classes)

    def get_class_frequencies(self, results: list) -> dict:
        """Get frequency of each class across all frames."""
        freq = {}
        for r in results:
            for name in r.class_names:
                freq[name] = freq.get(name, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: -x[1]))


# =========================================================================
# Interface 2: SAM3 Segmentation + Video Tracking
# =========================================================================

class SAM3Segmenter:
    """Pixel-perfect segmentation + cross-frame object tracking via SAM3.

    Uses SAM3's video PCS (Promptable Concept Segmentation) mode:
    - Text prompts to find all instances of a concept
    - Tracks objects across frames with persistent IDs

    Usage:
        segmenter = SAM3Segmenter()
        results = segmenter.segment_video_with_text(frames, ["table", "chair"])
    """

    def __init__(self, model_name: str = "facebook/sam3", device: str = "cuda",
                 dtype=None):
        if not _HAS_SAM3:
            raise ImportError(
                "transformers SAM3 not available. "
                "Run: uv add transformers accelerate\n"
                "Then accept license at https://huggingface.co/facebook/sam3"
            )
        self.device = device
        default_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        self.dtype = dtype or default_dtype
        self.model_name = model_name
        self.hf_token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        )

        logger.info(f"Loading SAM3 model: {model_name}...")
        self.model = Sam3VideoModel.from_pretrained(
            model_name,
            token=self.hf_token,
        ).to(
            device, dtype=self.dtype
        )
        self.processor = Sam3VideoProcessor.from_pretrained(
            model_name,
            token=self.hf_token,
        )
        logger.info("SAM3 model loaded.")

    def segment_video_with_text(self, video_frames: list,
                                 text_prompts: list,
                                 video_fps: float = 30.0,
                                 max_frames: Optional[int] = None
                                 ) -> list:
        """Segment + track objects across video frames using text prompts.

        Args:
            video_frames: list of RGB numpy arrays (H, W, 3) uint8
            text_prompts: list of text concepts to segment, e.g. ["table", "chair"]
            video_fps: original video FPS for timestamps
            max_frames: max frames to process. None means all provided frames.

        Returns:
            list of SegmentationResult, one per frame
        """
        if max_frames is None or max_frames <= 0:
            frames_to_process = video_frames
        else:
            frames_to_process = video_frames[:max_frames]
        all_results = []

        # Process each text prompt separately and merge
        for prompt_text in text_prompts:
            try:
                prompt_results = self._segment_single_prompt(
                    frames_to_process, prompt_text, video_fps
                )
                # Merge into all_results
                for i, pr in enumerate(prompt_results):
                    if i >= len(all_results):
                        all_results.append(pr)
                    else:
                        # Merge masks from this prompt into existing result
                        all_results[i].masks.extend(pr.masks)
                        all_results[i].labels.extend(pr.labels)
                        all_results[i].scores.extend(pr.scores)
                        all_results[i].object_ids.extend(pr.object_ids)
            except Exception as e:
                logger.warning(f"SAM3 failed for prompt '{prompt_text}': {e}")

        return all_results

    def _segment_single_prompt(self, frames: list, text_prompt: str,
                                video_fps: float) -> list:
        """Run SAM3 video segmentation for a single text prompt."""
        from transformers.video_utils import load_video  # noqa: F401
        from PIL import Image

        # Convert numpy frames to PIL Images for the processor
        pil_frames = [Image.fromarray(f) for f in frames]

        # Initialize video inference session
        inference_session = self.processor.init_video_session(
            video=pil_frames,
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self.dtype,
        )

        # Add text prompt
        inference_session = self.processor.add_text_prompt(
            inference_session=inference_session,
            text=text_prompt,
        )

        # Propagate through video
        results_per_frame = []
        for model_outputs in self.model.propagate_in_video_iterator(
            inference_session=inference_session, max_frame_num_to_track=len(frames)
        ):
            processed = self.processor.postprocess_outputs(
                inference_session, model_outputs
            )
            frame_idx = model_outputs.frame_idx
            timestamp = frame_idx / video_fps

            # Extract masks and metadata
            masks_list = []
            scores_list = []
            object_ids_list = []
            labels_list = []

            if "masks" in processed and processed["masks"] is not None:
                masks_tensor = processed["masks"]
                if hasattr(masks_tensor, 'cpu'):
                    masks_np = masks_tensor.cpu().numpy()
                else:
                    masks_np = np.array(masks_tensor)

                n_objects = masks_np.shape[0] if masks_np.ndim >= 3 else 0
                for obj_i in range(n_objects):
                    mask = masks_np[obj_i]
                    if mask.ndim == 3:
                        mask = mask[0]  # take first channel
                    masks_list.append((mask > 0.5).astype(np.uint8))
                    labels_list.append(text_prompt)
                    object_ids_list.append(
                        int(processed["object_ids"][obj_i])
                        if "object_ids" in processed and obj_i < len(processed["object_ids"])
                        else obj_i
                    )
                    scores_list.append(
                        float(processed["scores"][obj_i])
                        if "scores" in processed and obj_i < len(processed["scores"])
                        else 1.0
                    )

            results_per_frame.append(SegmentationResult(
                frame_idx=frame_idx,
                timestamp=timestamp,
                masks=masks_list,
                labels=labels_list,
                scores=scores_list,
                object_ids=object_ids_list,
            ))

        return results_per_frame

    def segment_frame_with_boxes(self, frame_rgb: np.ndarray,
                                  boxes: np.ndarray,
                                  frame_idx: int = 0,
                                  timestamp: float = 0.0
                                  ) -> SegmentationResult:
        """Segment objects in a single frame using bounding box prompts.

        Fallback mode when video tracking isn't needed.
        Uses SAM3's image PCS with box prompts.
        """
        from transformers import Sam3Processor, Sam3Model
        from PIL import Image

        # Load image model (lighter than video model)
        if not hasattr(self, '_image_model'):
            self._image_model = Sam3Model.from_pretrained(self.model_name).to(
                self.device
            )
            self._image_processor = Sam3Processor.from_pretrained(self.model_name)

        pil_image = Image.fromarray(frame_rgb)

        # Format boxes as expected by processor
        input_boxes = [boxes.tolist()]
        input_boxes_labels = [[1] * len(boxes)]

        inputs = self._image_processor(
            images=pil_image,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._image_model(**inputs)

        results = self._image_processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        masks_list = []
        scores_list = []
        for i, mask in enumerate(results.get("masks", [])):
            if hasattr(mask, 'cpu'):
                masks_list.append(mask.cpu().numpy().astype(np.uint8))
            else:
                masks_list.append(np.array(mask).astype(np.uint8))
            scores_list.append(
                float(results["scores"][i]) if "scores" in results else 1.0
            )

        return SegmentationResult(
            frame_idx=frame_idx,
            timestamp=timestamp,
            masks=masks_list,
            labels=["object"] * len(masks_list),
            scores=scores_list,
            object_ids=list(range(len(masks_list))),
        )


# =========================================================================
# Interface 3: Gemini Video Analyzer (full video upload)
# =========================================================================

class GeminiVideoAnalyzer:
    """Holistic scene understanding by uploading full video to Gemini.

    Gemini can process up to 100MB of video (~11 min at standard quality).
    Returns structured scene description with objects, events, and relations.

    Usage:
        analyzer = GeminiVideoAnalyzer()
        scene = analyzer.analyze_video("video.mp4")
    """

    def __init__(self, api_key: str = None,
                 model_name: str = "gemini-2.5-flash"):
        if not _HAS_GEMINI:
            raise ImportError("google-genai not installed. Run: uv add google-genai")
        self.api_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not self.api_key:
            raise ValueError("No Gemini API key. Set GOOGLE_API_KEY in .env")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        logger.info(f"GeminiVideoAnalyzer initialized with {model_name}")

    def analyze_video(self, video_path: str,
                      custom_prompt: str = None) -> SceneDescription:
        """Upload full video to Gemini and get holistic scene analysis.

        For videos < 20MB: inline bytes
        For videos >= 20MB: use File API with polling
        """
        file_size = os.path.getsize(video_path)
        print(f"  Gemini Video: uploading {video_path} ({file_size / 1e6:.1f} MB)...")

        prompt = custom_prompt or self._default_scene_prompt()

        if file_size < 20 * 1024 * 1024:
            # Inline upload for small videos
            return self._analyze_inline(video_path, prompt)
        else:
            # File API for larger videos
            return self._analyze_file_api(video_path, prompt)

    def _default_scene_prompt(self) -> str:
        return """Analyze this video comprehensively. Provide a structured analysis as JSON:

{
  "narrative": "A free-text description of what happens in the video from start to end.",
  "objects": [
    {
      "name": "unique name like Table_01",
      "type": "category (table, chair, door, cup, person, etc.)",
      "first_appearance": "timestamp or description (e.g., '0:02' or 'beginning')",
      "last_appearance": "timestamp or description",
      "description": "brief visual description"
    }
  ],
  "events": [
    {
      "time": "timestamp or early/mid/late",
      "description": "what happened",
      "objects_involved": ["object names involved"]
    }
  ],
  "spatial_relations": [
    {
      "obj_a": "object name",
      "relation": "on_top_of / next_to / behind / in_front_of / inside / etc.",
      "obj_b": "object name"
    }
  ]
}

Be thorough: identify ALL visible objects, ALL events, and ALL spatial relationships.
Return ONLY valid JSON, no markdown fences."""

    def _analyze_inline(self, video_path: str, prompt: str) -> SceneDescription:
        """Upload video inline (< 20MB)."""
        mime = "video/mp4"
        if video_path.endswith(".webm"):
            mime = "video/webm"
        elif video_path.endswith(".avi"):
            mime = "video/x-msvideo"

        with open(video_path, "rb") as f:
            video_bytes = f.read()

        video_part = genai.types.Part.from_bytes(data=video_bytes, mime_type=mime)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[video_part, prompt],
        )
        return self._parse_response(response.text)

    def _analyze_file_api(self, video_path: str, prompt: str) -> SceneDescription:
        """Upload video via File API (>= 20MB), poll until processed."""
        print("  Using Gemini File API for large video...")
        video_file = self.client.files.upload(file=video_path)
        print(f"  Uploaded as: {video_file.name}, waiting for processing...")

        # Poll until processing is complete
        max_wait = 300  # 5 minutes
        waited = 0
        while hasattr(video_file, 'state') and \
              hasattr(video_file.state, 'name') and \
              video_file.state.name == "PROCESSING":
            time.sleep(5)
            waited += 5
            video_file = self.client.files.get(name=video_file.name)
            if waited > max_wait:
                raise TimeoutError("Gemini video processing timed out after 5 min")
            if waited % 30 == 0:
                print(f"  Still processing... ({waited}s)")

        print(f"  Video processed. Generating content...")

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[video_file, prompt],
        )
        return self._parse_response(response.text)

    def _parse_response(self, text: str) -> SceneDescription:
        """Parse Gemini's text response into SceneDescription."""
        json_str = text.strip()

        # Strip markdown fences
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            start = 1 if lines[0].startswith("```") else 0
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            json_str = "\n".join(lines[start:end])

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Gemini response as JSON")
            return SceneDescription(narrative=text, raw_response=text)

        if isinstance(parsed, list):
            return SceneDescription(objects=parsed, raw_response=text)

        return SceneDescription(
            objects=parsed.get("objects", []),
            events=parsed.get("events", []),
            narrative=parsed.get("narrative", ""),
            spatial_relations=parsed.get("spatial_relations", []),
            raw_response=text,
        )

    def analyze_scene_with_context(self, video_path: str,
                                    yolo_detections: list,
                                    mast3r_summary: dict) -> SceneDescription:
        """Analyze video with additional context from other models."""
        yolo_classes = set()
        for det in yolo_detections:
            yolo_classes.update(det.class_names)

        context = f"""Additional context from other AI models:
- YOLOv8 detected these object classes: {', '.join(sorted(yolo_classes))}
- 3D reconstruction found {mast3r_summary.get('num_points', 0)} 3D points across {mast3r_summary.get('num_frames', 0)} keyframes
- Camera trajectory covers approximately {mast3r_summary.get('trajectory_length_m', 'unknown')} meters

Please incorporate this information into your analysis.
"""
        prompt = context + "\n" + self._default_scene_prompt()
        return self.analyze_video(video_path, custom_prompt=prompt)


# =========================================================================
# Interface 4: Reasoning Engine (cross-model validation)
# =========================================================================

class ReasoningEngine:
    """Cross-validates all model outputs and produces verified scene graph.

    Supports two providers:
    - "gemini": Google Gemini Pro for reasoning
    - "openai": OpenAI GPT-5.2 for advanced reasoning

    Usage:
        engine = ReasoningEngine(provider="gemini")
        result = engine.synthesize(detections, segmentations, scene, geometry)
    """

    def __init__(self, provider: str = "gemini", api_key: str = None,
                 model_name: str = None):
        self.provider = provider

        if provider == "gemini":
            if not _HAS_GEMINI:
                raise ImportError("google-genai not installed")
            self.api_key = (
                api_key
                or os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("GEMINI_API_KEY")
            )
            self.model_name = model_name or "gemini-2.5-flash"
            self.client = genai.Client(api_key=self.api_key)

        elif provider == "openai":
            try:
                import openai
                self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
                self.client = openai.OpenAI(api_key=self.api_key)
                self.model_name = model_name or "gpt-4o"
            except ImportError:
                raise ImportError("openai package not installed. Run: uv add openai")

        logger.info(f"ReasoningEngine initialized with {provider}/{self.model_name}")

    def synthesize(self, detections: list = None,
                   segmentations: list = None,
                   scene: SceneDescription = None,
                   geometry_summary: dict = None,
                   objects_3d: list = None) -> ReasoningResult:
        """Cross-validate all model outputs and produce verified scene graph.

        Steps:
        1. Collect evidence from each model
        2. Match detections across models by name/type
        3. Compute confidence scores
        4. Identify disagreements
        5. Build reasoning trace
        6. Flag items for human review
        """
        # Build evidence summary
        evidence = self._collect_evidence(
            detections, segmentations, scene, geometry_summary, objects_3d
        )

        # Build the reasoning prompt
        prompt = self._build_reasoning_prompt(evidence)

        # Call the reasoning model
        reasoning_text = self._call_model(prompt)

        # Parse the response
        return self._parse_reasoning(reasoning_text, evidence)

    def _collect_evidence(self, detections, segmentations, scene,
                          geometry_summary, objects_3d) -> dict:
        """Collect evidence from all model outputs."""
        evidence = {
            "yolo_objects": {},
            "sam3_objects": {},
            "gemini_objects": {},
            "geometry_objects": {},
        }

        # YOLO evidence
        if detections:
            for det in detections:
                for name in det.class_names:
                    if name not in evidence["yolo_objects"]:
                        evidence["yolo_objects"][name] = {
                            "count": 0, "avg_confidence": 0, "frames": []
                        }
                    evidence["yolo_objects"][name]["count"] += 1
                    evidence["yolo_objects"][name]["frames"].append(det.frame_idx)
            # Compute avg confidence
            for det in detections:
                for i, name in enumerate(det.class_names):
                    conf = float(det.scores[i]) if i < len(det.scores) else 0
                    evidence["yolo_objects"][name]["avg_confidence"] = (
                        evidence["yolo_objects"][name]["avg_confidence"]
                        * (evidence["yolo_objects"][name]["count"] - 1)
                        + conf
                    ) / evidence["yolo_objects"][name]["count"]

        # SAM3 evidence
        if segmentations:
            for seg in segmentations:
                for i, label in enumerate(seg.labels):
                    if label not in evidence["sam3_objects"]:
                        evidence["sam3_objects"][label] = {
                            "count": 0, "avg_score": 0, "tracked_ids": set()
                        }
                    evidence["sam3_objects"][label]["count"] += 1
                    if i < len(seg.object_ids):
                        evidence["sam3_objects"][label]["tracked_ids"].add(
                            seg.object_ids[i]
                        )
            # Convert sets to lists for JSON
            for k in evidence["sam3_objects"]:
                evidence["sam3_objects"][k]["tracked_ids"] = list(
                    evidence["sam3_objects"][k]["tracked_ids"]
                )

        # Gemini scene evidence
        if scene and scene.objects:
            for obj in scene.objects:
                name = obj.get("name", obj.get("type", "unknown"))
                evidence["gemini_objects"][name] = obj

        # Geometry evidence
        if objects_3d:
            for obj in objects_3d:
                key = getattr(obj, 'entity', str(obj))
                evidence["geometry_objects"][key] = {
                    "type": getattr(obj, 'obj_type', 'unknown'),
                    "has_3d_position": True,
                }

        return evidence

    def _build_reasoning_prompt(self, evidence: dict) -> str:
        """Build a prompt for the reasoning model."""
        return f"""You are a scene graph validator. Cross-validate these detections from multiple AI models.

## YOLO Detections (fast object detector)
{json.dumps(evidence["yolo_objects"], indent=2, default=str)}

## SAM3 Segmentations (pixel-perfect masks)
{json.dumps(evidence["sam3_objects"], indent=2, default=str)}

## Gemini Scene Analysis (holistic understanding)
{json.dumps(evidence["gemini_objects"], indent=2, default=str)}

## 3D Geometry (MASt3R reconstruction)
{json.dumps(evidence["geometry_objects"], indent=2, default=str)}

For each unique object entity, determine:
1. verified: true/false (is this a real object, confirmed by 2+ models?)
2. confidence: 0.0-1.0 based on cross-model agreement
3. canonical_name: the best name for this object
4. canonical_type: the best type category
5. state_changes: any changes observed (from Gemini events)
6. human_review: true if uncertain (confidence < 0.5 or models disagree)
7. review_reason: why human review is needed (if applicable)

Return ONLY valid JSON:
{{
  "verified_objects": [
    {{
      "entity": "Table_01",
      "type": "table",
      "verified": true,
      "confidence": 0.92,
      "detected_by": ["yolo", "sam3", "gemini"],
      "human_review": false,
      "review_reason": ""
    }}
  ],
  "state_changes": [
    {{
      "entity": "Door_01",
      "time": "mid",
      "from_state": "closed",
      "to_state": "open",
      "cause": "person interaction"
    }}
  ],
  "reasoning_trace": "Step-by-step explanation of the validation process..."
}}"""

    def _call_model(self, prompt: str) -> str:
        """Call the reasoning model (Gemini or OpenAI)."""
        try:
            if self.provider == "gemini":
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt],
                )
                return response.text
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Reasoning model call failed: {e}")
            return json.dumps({
                "verified_objects": [],
                "state_changes": [],
                "reasoning_trace": f"Model call failed: {e}",
            })

    def _parse_reasoning(self, text: str, evidence: dict) -> ReasoningResult:
        """Parse the reasoning model's response."""
        json_str = text.strip()
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            start = 1 if lines[0].startswith("```") else 0
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            json_str = "\n".join(lines[start:end])

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return ReasoningResult(reasoning_trace=f"Parse failed. Raw: {text[:500]}")

        verified = parsed.get("verified_objects", [])
        confidence_scores = {
            obj.get("entity", f"obj_{i}"): obj.get("confidence", 0.5)
            for i, obj in enumerate(verified)
        }
        human_flags = [
            obj for obj in verified if obj.get("human_review", False)
        ]

        return ReasoningResult(
            verified_objects=verified,
            confidence_scores=confidence_scores,
            state_changes=parsed.get("state_changes", []),
            human_review_flags=human_flags,
            reasoning_trace=parsed.get("reasoning_trace", ""),
        )


# =========================================================================
# Utility: Check availability
# =========================================================================

def check_available_interfaces() -> dict:
    """Check which model interfaces are available."""
    available = {
        "yolo": _HAS_YOLO,
        "sam3": _HAS_SAM3,
        "gemini": _HAS_GEMINI and bool(
            os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        ),
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
    }

    print("Model Interface Availability:")
    for name, ok in available.items():
        status = "AVAILABLE" if ok else "NOT AVAILABLE"
        print(f"  {name:10s}: {status}")

    return available


if __name__ == "__main__":
    check_available_interfaces()
