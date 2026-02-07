import json
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image
from PIL import Image

def run():
    # 1. Load configuration
    config_path = os.path.join("inputs", "config.json")
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    image_url = config.get("image_url")
    local_image_path = config.get("local_image_path")
    prompt_text = config.get("prompt", "Describe this image.")

    # 2. Load Image
    image = None
    if local_image_path and os.path.exists(local_image_path):
        print(f"Loading local image from: {local_image_path}")
        image = Image.open(local_image_path)
    elif image_url:
        print(f"Loading image from URL: {image_url}")
        try:
            image = load_image(image_url)
        except Exception as e:
            print(f"Error loading image from URL: {e}")
            return
    else:
        print("Error: No valid 'image_url' or 'local_image_path' provided in config.json")
        return

    # 3. Load Model and Processor
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

    # 4. Prepare Conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

    # 5. Generate Answer
    print("Generating response...")
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=256) # Increased tokens slightly
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    print("-" * 50)
    print("Model Output:")
    print(response)
    print("-" * 50)

if __name__ == "__main__":
    run()
