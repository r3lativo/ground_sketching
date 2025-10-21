# qwen_vl_utils.py
# This code is adapted from the official Qwen-VL repository.
# It is required for preprocessing vision inputs.
import base64
from io import BytesIO
from typing import List, Dict, Any

import requests
from PIL import Image

def process_vision_info(messages: List[Dict[str, Any]]):
    image_inputs, video_inputs = [], []
    for message in messages:
        if not isinstance(message['content'], list):
            continue
        for item in message['content']:
            if item['type'] == 'image':
                image_inputs.append(item['image'])
            elif item['type'] == 'video':
                video_inputs.append(item['video'])
    return image_inputs, video_inputs

def decode_base64_to_pil(base64_string: str):
    """Decodes a Base64 string to a PIL Image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")

def load_image(image_path: str):
    """Loads an image from a URL or local path."""
    if image_path.startswith("http://") or image_path.startswith("https://"):
        return Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
    else:
        return Image.open(image_path).convert("RGB")
