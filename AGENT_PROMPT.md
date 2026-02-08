# Agent Implementation Prompt -- World2Data v2

## Context

You are implementing the World2Data v2 pipeline. The project converts 2D video
into a 4D Dynamic Scene Graph (OpenUSD) using multiple AI models.

**CRITICAL FILES TO READ FIRST (in this order):**
1. `IMPLEMENTATION_PLAN.md` -- Full architecture and connection plan
2. `World2Data.txt` -- Original project vision and goals
3. `pipeline_controller.py` -- Current working pipeline (v1)
4. `test_pipeline.py` -- Current test suite (27 passing + 1 overnight)
5. `pyproject.toml` -- Current dependencies
6. `.env` -- API keys (GOOGLE_API_KEY, OPENAI_API_KEY)

**WHAT ALREADY WORKS (DO NOT BREAK):**
- Step 1: Keyframe extraction from video (L1 diff)
- Step 2: MASt3R 3D reconstruction (per-frame temporal point clouds + depth)
- Step 3: Gemini-based 2D object detection + 3D back-projection
- Step 4: Gemini causal reasoning (physics joints, state changes)
- Step 5: OpenUSD export (colored point cloud, cameras, physics objects)
- Step 6: Rerun .rrd temporal recording (point clouds, objects, depth, trajectory)
- Step 7: PLY colored point cloud export
- Full test suite: 27 fast tests + 1 overnight test
- All imports resolve, all tests pass

**HARDWARE AVAILABLE:**
- NVIDIA RTX 4090 (24GB VRAM)
- CUDA installed and working
- Python 3.11, uv package manager
- Windows 10

---

## Task 1: Install Dependencies

Add to `pyproject.toml` dependencies:
```toml
"ultralytics>=8.3.0",
"transformers>=4.47.0",
"accelerate>=1.2.0",
```

Then run: `uv sync`

Verify installations:
```python
from ultralytics import YOLO
from transformers import Sam3VideoModel, Sam3VideoProcessor
```

**IMPORTANT:** SAM3 requires accepting the license on HuggingFace.
The user has a HuggingFace account. If the model download fails due to
license gate, instruct the user to visit https://huggingface.co/facebook/sam3
and accept the terms, then run `huggingface-cli login`.

---

## Task 2: Create `model_interfaces.py`

This file contains 5 clean interface classes. Each MUST follow this pattern:

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class FrameResult:
    """Base result for any per-frame model output."""
    frame_idx: int
    timestamp: float  # seconds into video

@dataclass
class SegmentationResult(FrameResult):
    masks: list[np.ndarray]      # list of (H, W) binary masks
    labels: list[str]            # "table", "chair", etc.
    scores: list[float]          # confidence per mask
    object_ids: list[int]        # persistent IDs across frames (SAM3 tracking)

@dataclass
class DetectionResult(FrameResult):
    boxes: np.ndarray            # (N, 4) xyxy pixel coordinates
    classes: np.ndarray          # (N,) integer class IDs
    class_names: list[str]       # "table", "chair", etc.
    scores: np.ndarray           # (N,) confidence scores
    masks: Optional[np.ndarray] = None  # (N, H, W) if using seg model

@dataclass
class SceneDescription:
    objects: list[dict]          # [{name, type, spatial_relation, ...}]
    events: list[dict]           # [{time, description, objects_involved}]
    narrative: str               # free-text scene description
    spatial_relations: list[dict]  # [{obj_a, relation, obj_b}]

@dataclass
class ReasoningResult:
    verified_objects: list[dict]   # objects confirmed by multiple models
    confidence_scores: dict        # {entity: float}
    state_changes: list[dict]      # validated state changes
    human_review_flags: list[dict] # items needing human review
    reasoning_trace: str           # step-by-step reasoning chain
```

### Interface Classes to Implement:

#### `class YOLODetector`
```python
class YOLODetector:
    def __init__(self, model_name="yolov8x-seg.pt", device="cuda"):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.device = device

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int,
                      timestamp: float) -> DetectionResult:
        """Run YOLOv8 on a single frame. Returns boxes + classes + masks."""
        results = self.model(frame_bgr, device=self.device, verbose=False)
        # Extract boxes, classes, scores, masks from results[0]
        ...

    def process_video(self, video_path: str) -> list[DetectionResult]:
        """Run YOLOv8 on entire video frame by frame."""
        ...

    def get_unique_classes(self, results: list[DetectionResult]) -> list[str]:
        """Get all unique class names detected across all frames."""
        ...
```

#### `class SAM3Segmenter`
```python
class SAM3Segmenter:
    def __init__(self, model_name="facebook/sam3", device="cuda"):
        from transformers import Sam3VideoModel, Sam3VideoProcessor
        import torch
        self.model = Sam3VideoModel.from_pretrained(model_name).to(
            device, dtype=torch.bfloat16
        )
        self.processor = Sam3VideoProcessor.from_pretrained(model_name)
        self.device = device

    def segment_video_with_text(self, video_frames: list[np.ndarray],
                                 text_prompts: list[str]
                                 ) -> list[SegmentationResult]:
        """Segment + track objects across video frames using text prompts.

        Uses SAM3's video PCS mode:
        1. Initialize video session with all frames
        2. Add text prompts (e.g., "table", "chair" from YOLO)
        3. Propagate through video to get masks + tracking IDs
        """
        # Use Sam3VideoProcessor.init_video_session(video=video_frames, ...)
        # Use processor.add_text_prompt(text=prompt)
        # Use model.propagate_in_video_iterator(...)
        ...

    def segment_frame_with_boxes(self, frame: np.ndarray,
                                  boxes: np.ndarray,
                                  frame_idx: int) -> SegmentationResult:
        """Segment specific objects given bounding boxes from YOLO."""
        # Fallback: use Sam3TrackerModel with box prompts
        ...
```

#### `class GeminiVideoAnalyzer`
```python
class GeminiVideoAnalyzer:
    def __init__(self, api_key: str = None):
        from google import genai
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

    def analyze_video(self, video_path: str) -> SceneDescription:
        """Upload full video to Gemini and get holistic scene analysis.

        Gemini supports up to 100MB video (~11 min).
        Returns structured scene description with objects, events, relations.
        """
        # Upload video file using genai.types.Part.from_bytes or file API
        # Use model="gemini-2.5-flash" or "gemini-2.5-pro"
        # Prompt for structured JSON output
        ...
```

**IMPORTANT for Gemini Video Upload:**
Use the File API for videos > 20MB:
```python
video_file = client.files.upload(file=video_path)
# Wait for processing
while video_file.state.name == "PROCESSING":
    time.sleep(2)
    video_file = client.files.get(name=video_file.name)
# Then use in generate_content
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[video_file, prompt],
)
```

#### `class ReasoningEngine`
```python
class ReasoningEngine:
    def __init__(self, provider="gemini", api_key=None):
        # provider can be "gemini" or "openai"
        self.provider = provider
        ...

    def synthesize(self, detections: list[DetectionResult],
                   segmentations: list[SegmentationResult],
                   scene: SceneDescription,
                   geometry: list  # FrameData from MASt3R
                   ) -> ReasoningResult:
        """Cross-validate all model outputs and produce verified scene graph.

        Steps:
        1. Match YOLO detections to SAM3 segments (IoU overlap)
        2. Match both to Gemini scene objects (name similarity)
        3. Compute confidence: how many models agree?
        4. Flag disagreements for human review
        5. Build reasoning trace explaining each decision
        """
        ...
```

---

## Task 3: Create `scene_fusion.py`

This module fuses all interface outputs into a unified 4D scene graph.

```python
@dataclass
class SceneObject4D:
    """A single object tracked through space and time."""
    entity_id: str             # "Table_01"
    obj_type: str              # "table"
    confidence: float          # 0.0-1.0 (cross-model agreement)
    detected_by: list[str]     # ["yolo", "sam3", "gemini"]

    # Spatial (3D)
    bbox_3d_min: np.ndarray    # (3,)
    bbox_3d_max: np.ndarray    # (3,)
    mask_3d_points: Optional[np.ndarray]  # points belonging to this object

    # Temporal (4D)
    first_seen_frame: int
    last_seen_frame: int
    frame_detections: dict     # {frame_idx: Detection/Segmentation}

    # Physics
    component_type: str        # "RevoluteJoint", "FixedJoint", etc.
    initial_state: str
    final_state: str
    state_changes: list[dict]

    # Human-in-the-loop
    human_review: bool         # needs human verification?
    review_reason: str
    reasoning_trace: str       # step-by-step explanation


class SceneFusion4D:
    """Fuses all model outputs into unified 4D scene graph."""

    def __init__(self, frame_data: list,  # FrameData from MASt3R
                 detections: list[DetectionResult],
                 segmentations: list[SegmentationResult],
                 scene_description: SceneDescription,
                 reasoning: ReasoningResult):
        ...

    def fuse(self) -> list[SceneObject4D]:
        """Main fusion algorithm:
        1. Group YOLO detections by class across frames
        2. Match with SAM3 tracked objects (by IoU)
        3. Project 2D regions to 3D using MASt3R depth
        4. Match with Gemini scene objects (by name/type)
        5. Compute confidence from cross-model agreement
        6. Flag uncertain items for human review
        """
        ...

    def project_mask_to_3d(self, mask_2d: np.ndarray,
                           frame_data: FrameData) -> np.ndarray:
        """Project a 2D binary mask to 3D points using MASt3R depth.
        Returns the 3D points that fall within the mask."""
        ...

    def compute_confidence(self, yolo_score: float,
                           sam3_score: float,
                           gemini_mentioned: bool) -> float:
        """Confidence = weighted average of model agreements."""
        ...
```

---

## Task 4: Update `pipeline_controller.py`

### Changes needed:

1. **Replace Step 3** with multi-model detection:
   ```python
   def step_3_multi_model_detection(self):
       # a) Run YOLOv8 on all keyframes
       # b) Run SAM3 video segmentation with YOLO class names as prompts
       # c) Run Gemini video analysis (full video upload)
       # d) Fuse all outputs
   ```

2. **Replace Step 4** with reasoning engine:
   ```python
   def step_4_reasoning_synthesis(self):
       # Use ReasoningEngine to cross-validate
       # Assign confidence scores
       # Flag items for human review
   ```

3. **Update Step 5 USD export** to include:
   - `world2data:confidence` attribute per object
   - `world2data:detected_by` attribute per object
   - `world2data:human_review` boolean attribute
   - `world2data:reasoning_trace` string attribute
   - `/World/Uncertain/` group for low-confidence objects

4. **Update Step 6 Rerun recording** to include:
   - Original video frames with YOLO overlay as `input/annotated`
   - SAM3 mask overlays per frame
   - Confidence color coding (green=high, yellow=medium, red=low)
   - Text log panel with reasoning trace
   - Scene narrative from Gemini

5. **Keep backward compatibility:**
   - `_mock_geometry()` still works
   - All existing tests still pass
   - Pipeline gracefully degrades if SAM3/YOLO not available

---

## Task 5: Create `ground_truth_evaluator.py`

```python
class GroundTruthEvaluator:
    """Evaluates scene graph quality and improves over time.

    This is the product foundation -- the cost/evaluator that makes
    the pipeline a sellable framework.
    """

    def __init__(self, feedback_db_path="ground_truth_feedback.json"):
        self.feedback_db = self._load_feedback(feedback_db_path)

    def evaluate(self, objects_4d: list[SceneObject4D]) -> dict:
        """Score the scene graph.
        Returns:
        {
            "overall_score": 0.78,
            "metrics": {
                "cross_model_agreement": 0.85,
                "temporal_consistency": 0.72,
                "physics_plausibility": 0.90,
                "coverage": 0.65,
            },
            "improvement_suggestions": [...]
        }
        """

    def flag_for_review(self, objects_4d) -> list[dict]:
        """Items that need human correction."""

    def incorporate_feedback(self, object_id: str, correction: dict):
        """Store human correction for future improvement."""

    def get_known_objects(self) -> dict:
        """Return vocabulary of objects learned from past corrections."""
```

---

## Task 6: Create `video_annotator.py`

```python
class VideoAnnotator:
    """Render model outputs as overlays on video frames."""

    def annotate_frame(self, frame_bgr, detections, segmentations,
                       objects_4d) -> np.ndarray:
        """Draw YOLO boxes, SAM3 masks, and labels on a frame."""
        # Draw YOLO bounding boxes with class names
        # Overlay semi-transparent SAM3 masks
        # Add confidence labels
        # Color code: green=confirmed, yellow=uncertain, red=flagged

    def create_annotated_video(self, video_path, all_detections,
                                all_segmentations, objects_4d,
                                output_path) -> str:
        """Write an annotated video file with all overlays."""

    def create_rerun_video_panel(self, video_path, all_detections,
                                  all_segmentations):
        """Log annotated frames to Rerun as input/annotated panel."""
```

---

## Task 7: Update Tests

Add to `test_pipeline.py` or create `test_interfaces.py`:

```python
class TestYOLODetector:
    def test_loads_model(self):
        from model_interfaces import YOLODetector
        det = YOLODetector(model_name="yolov8n.pt")  # nano for fast tests
        assert det.model is not None

    def test_detects_objects_in_frame(self, pipeline):
        # Use a keyframe from the test video
        ...

class TestSAM3Segmenter:
    @pytest.mark.skipif(not _can_load_sam3(), reason="SAM3 not available")
    def test_loads_model(self):
        from model_interfaces import SAM3Segmenter
        seg = SAM3Segmenter()
        assert seg.model is not None

    def test_segments_with_text(self, pipeline):
        ...

class TestGeminiVideoAnalyzer:
    @pytest.mark.skipif(not api_key_available(), reason="No API key")
    def test_analyzes_video(self):
        from model_interfaces import GeminiVideoAnalyzer
        analyzer = GeminiVideoAnalyzer()
        result = analyzer.analyze_video(REAL_VIDEO)
        assert isinstance(result.narrative, str)
        assert len(result.objects) > 0

class TestSceneFusion:
    def test_fuses_mock_data(self):
        # Create mock outputs from each interface
        # Verify fusion produces SceneObject4D list
        ...

    def test_confidence_scoring(self):
        # Object detected by 3 models -> high confidence
        # Object detected by 1 model -> low confidence
        ...
```

---

## STRICT REQUIREMENTS

1. **DO NOT break existing tests.** Run `uv run python -m pytest test_pipeline.py -v`
   after every change. All 27 tests must still pass.

2. **Graceful degradation.** If SAM3 or YOLO can't load, the pipeline
   must still work (skip those steps, log a warning).

3. **All data structures must be JSON-serializable** via `.to_dict()` methods.

4. **All per-frame data must include `frame_idx` and `timestamp`.**

5. **Use `torch.bfloat16` for SAM3** to fit in 24GB VRAM alongside MASt3R.

6. **YOLOv8 should use `yolov8x-seg.pt`** for best accuracy with segmentation.
   For tests, use `yolov8n.pt` (fast, small).

7. **Gemini video upload uses the File API** for videos > 20MB.
   The File API requires `client.files.upload()` and polling for processing.

8. **Confidence formula:**
   ```
   confidence = 0.4 * yolo_confidence + 0.3 * sam3_score + 0.3 * gemini_mentioned
   ```
   Where `gemini_mentioned` is 1.0 if Gemini's scene description mentions
   the object type, 0.0 otherwise.

9. **Human review flag** when:
   - confidence < 0.5
   - Models disagree on object type
   - State change detected but physics type is "FixedJoint"
   - Object appears in < 20% of frames

10. **The pipeline must complete in < 10 minutes** for a 30-second video
    on an RTX 4090. Profile and optimize if needed.

11. **Environment:** Windows 10, PowerShell. Do NOT use `&&` to chain commands.
    Use `;` or separate commands. Do NOT use unicode arrows in print statements.

12. **Package manager:** `uv` (not pip, not conda). Use `uv add` for new deps,
    `uv sync` to install, `uv run` to execute.
