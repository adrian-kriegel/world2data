import cv2
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

def extract_frames(video_path, num_frames=5):
    """Extracts 'num_frames' frames from the video evenly spaced."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * total_frames / (num_frames - 1)) for i in range(num_frames)]
    # Handle edge case where total_frames is small or num_frames is 1
    if num_frames == 1:
        frame_indices = [total_frames // 2]
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames

def run_video_inference():
    # 1. basic setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(base_dir, "inputs", "videos")
    
    # 2. Find a video
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print(f"No video files found in {video_dir}. Please place an .mp4 file there.")
        return
    
    selected_video = os.path.join(video_dir, video_files[0])
    print(f"Processing video: {selected_video}")

    # 3. Load Model
    print("Loading model... (this may take a while first time)")
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

    # 4. Process Video (Extract Frames)
    print("Extracting frames...")
    try:
        frames = extract_frames(selected_video, num_frames=5)
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return

    print(f"Extracted {len(frames)} frames.")

    # 5. Prepare Conversation
    # Multi-image input format: [{"type": "image", "image": img1}, {"type": "image", "image": img2}, ..., {"type": "text", "text": "..."}]
    content_list = []
    for frame in frames:
        content_list.append({"type": "image", "image": frame})
    
    content_list.append({"type": "text", "text": "Describe what is happening in this video."})

    conversation = [
        {
            "role": "user",
            "content": content_list,
        },
    ]

    # 6. Generate
    print("Generating response...")
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=256)
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    print("-" * 50)
    print("Model Analysis of Video:")
    print(response)
    print("-" * 50)

if __name__ == "__main__":
    run_video_inference()
