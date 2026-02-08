import cv2
import os
import torch
import math
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return {"fps": fps, "frame_count": frame_count, "duration": duration}

def extract_frames_for_segment(video_path, start_sec, end_sec, target_fps=3):
    """
    Extracts frames for a specific time segment at a target framerate.
    target_fps=3 ensures we get ~15 frames for a 5s chunk. 
    15 frames * ~256 tokens = 3840 tokens. Very healthy.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Safety check for fps
    if fps <= 0: fps = 24.0

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    
    # Calculate step to match target_fps
    step = max(1, int(fps / target_fps))
    
    frames = []
    
    current_frame = start_frame
    while current_frame < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break
        try:
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        except Exception:
            pass # Skip bad frames
        current_frame += step
        
    cap.release()
    return frames

def run_high_res_analysis():
    # 1. Setup Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(base_dir, "inputs", "videos")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Find Video
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print(f"No video files found in {video_dir}. Please place an .mp4 file there.")
        return

    video_filename = video_files[0]
    video_path = os.path.join(video_dir, video_filename)
    video_name_no_ext = os.path.splitext(video_filename)[0]
    output_path = os.path.join(output_dir, f"{video_name_no_ext}.txt")
    
    info = get_video_info(video_path)
    if not info:
        print("Error reading video metadata.")
        return

    print(f"Processing: {video_filename}")
    print(f"Duration: {info['duration']:.2f}s")
    
    # 3. Load Model
    print("Loading model...")
    
    # User Requirement: Enforce CUDA execution
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please install PyTorch with CUDA support to use the GPU.")
    
    print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    model_id = "LiquidAI/LFM2.5-VL-1.6B"
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            dtype="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
        )
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Sliding Window Config
    SEGMENT_DURATION = 5.0 # Seconds per chunk
    num_segments = math.ceil(info['duration'] / SEGMENT_DURATION)
    
    # Initialize Output File
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"VIDEO ANALYSIS REPORT\n")
        f.write(f"File: {video_filename}\n")
        f.write(f"Duration: {info['duration']:.2f}s\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Strategy: Sliding Window ({SEGMENT_DURATION}s chunks @ 3fps)\n")
        f.write("=" * 60 + "\n\n")

    print(f"Starting analysis in {num_segments} chunks...")
    
    # 5. Execution Loop
    for i in range(num_segments):
        start_t = i * SEGMENT_DURATION
        end_t = min((i + 1) * SEGMENT_DURATION, info['duration'])
        
        print(f"Processing chunk {i+1}/{num_segments} ({start_t:.1f}s - {end_t:.1f}s)...")
        
        frames = extract_frames_for_segment(video_path, start_t, end_t, target_fps=3)
        
        if not frames:
            continue
            
        # Prepare Conversation
        content_list = []
        for frame in frames:
            content_list.append({"type": "image", "image": frame})
        content_list.append({"type": "text", "text": "Describe meticulously what is happening in this short video segment."})

        conversation = [{"role": "user", "content": content_list}]
        
        # Inference
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=300)
        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Format output
        # Clean up the response to just get the assistant part if possible
        if "assistant\n" in response:
            clean_response = response.split("assistant\n")[-1].strip()
        else:
            clean_response = response

        log_entry = f"TIMECODE [{start_t:.1f}s - {end_t:.1f}s]\n{clean_response}\n"
        print(log_entry)
        
        # Append to File immediately (so we don't lose progress if it crashes)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n" + "-" * 20 + "\n\n")

    print(f"Analysis complete. Full report saved to: {output_path}")

if __name__ == "__main__":
    run_high_res_analysis()
