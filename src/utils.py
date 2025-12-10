# src/utils.py

import base64
import io
import logging
from PIL import Image
import yaml
import csv
import os
import re
import sys
from datetime import datetime
from PIL import Image, ImageFilter, ImageDraw
import socket
import json
import numpy as np
from typing import Optional, Any, Union, Dict, List
from pathlib import Path
import pandas as pd
import random

logger = logging.getLogger(__name__)

# --- Constants & Magic Strings ---
class ProjectSymbols:
    NEW = "[NEW]"
    CONTINUE = "[CONTINUE]"
    SKIP = "[SKIP]"
    ZOOM_OUT = "[ZOOM_OUT]"
    SEPARATOR = "$$$"
    MOVED = "<moved>"

LOG_HEADER = [
    "timestamp", "status", "initial_prompt", "final_prompt",
    "negative_prompt", "VLseed", "IMGseed", "true_cfg_scale",
    "num_inference_steps", "top_p", "temperature", "output_file",
    "comment",
]

# --- Logging ---
def setup_logging(log_file='logs/app.log', level=logging.INFO, log_to_console=True):
    """
    Sets up logging. Safe version that respects existing config if needed.
    """
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    handlers = [logging.FileHandler(log_file)]
    if log_to_console:
        handlers.append(logging.StreamHandler(sys.stdout))

    # Configure root logger without forcefully removing external handlers
    # unless strictly necessary.
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True 
    )

# --- Config & IO ---
def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"[UTILS] Config load error: {e}")
        raise

def pil_to_base64(pil_image: Image.Image, format="PNG") -> str:
    try:
        with io.BytesIO() as buffer:
            pil_image.save(buffer, format=format)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"[UTILS] Encoding error: {e}")
        raise

def base64_to_pil(base64_string: str) -> Image.Image:
    try:
        img_bytes = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not decode base64: {e}")

# --- Logging Helpers ---

def log_experiment_step(log_filepath, data):
    """Appends a row to a CSV log file."""
    try:
        os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
        file_exists = os.path.isfile(log_filepath)
        
        with open(log_filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=LOG_HEADER)
            if not file_exists or os.path.getsize(log_filepath) == 0:
                writer.writeheader()
            
            row = {k: data.get(k, "") for k in LOG_HEADER}
            row["timestamp"] = datetime.now().isoformat()
            writer.writerow(row)
    except Exception as e:
        logger.error(f"[UTILS] Log write failed: {e}")

def update_last_log_comment(log_filepath: str, comment: str):
    """Updates the 'comment' field of the last row in the CSV."""
    if not comment or not os.path.isfile(log_filepath):
        return
    
    try:
        rows = []
        fieldnames = LOG_HEADER
        
        with open(log_filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = reader.fieldnames
            rows = list(reader)
        
        if rows:
            rows[-1]['comment'] = comment
            
            with open(log_filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                
        logger.info(f"[UTILS] Updated comment in {log_filepath}")
    except Exception as e:
        logger.error(f"[UTILS] Failed to update log comment: {e}")

# --- Optimized Image Processing (NumPy) ---

def clean_image_artifacts(input_image, white_threshold=180, black_threshold=50):
    if input_image is None: return None
    
    # Vectorized NumPy operation (Step 1 Optimization)
    img_array = np.array(input_image.convert("RGB"))
    is_white = np.all(img_array > white_threshold, axis=2)
    is_black = np.all(img_array < black_threshold, axis=2)
    img_array[is_white] = [255, 255, 255]
    img_array[is_black] = [0, 0, 0]
    
    return Image.fromarray(img_array)

def add_padding_to_image(img_pil, scale_factor=0.8, fill_color="white"):
    if not (0 < scale_factor <= 1.0): return img_pil
    try:
        w, h = img_pil.size
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        resized = img_pil.resize((new_w, new_h), Image.LANCZOS)
        final = Image.new("RGB", (w, h), fill_color)
        final.paste(resized, ((w - new_w) // 2, (h - new_h) // 2))
        return final
    except Exception as e:
        logger.error(f"[UTILS] Padding error: {e}")
        return img_pil

# --- Network Checks ---

def check_server(host: str, port: int, timeout: int = 60) -> bool:
    """Checks if a server is reachable."""
    check_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"Checking {check_host}:{port}...", end="", flush=True)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((check_host, port))
        print(" [OK]")
        return True
    except Exception as e:
        print(f" [FAILED] ({e})")
        return False

# --- Robust Parsers ---

def json_parser(input_text: str, key_term: Optional[str] = None) -> Union[Dict, List, str, Any]:
    """
    Robustly extracts JSON from Markdown blocks or raw text.
    
    Improvements:
    1. key_term is now Optional. If None, returns the whole JSON object.
    2. Uses Regex to extract content *inside* ```json ... ``` blocks, ignoring surrounding text.
    3. Fallback to finding the first '{' and last '}' if markdown tags are missing.
    """
    if not input_text:
        return input_text

    json_str = input_text

    # 1. Regex Extraction (Best for "Text + JSON block" scenarios)
    # Looks for ```json ... ``` or just ``` ... ```
    # re.DOTALL allows the dot (.) to match newlines
    match = re.search(r"```(?:json)?\s*(.*?)```", input_text, re.DOTALL)
    
    if match:
        json_str = match.group(1).strip()
    else:
        # 2. Fallback Heuristic: Find the first '{' and last '}'
        # Useful if the model forgot markdown tags but outputted JSON mixed with text
        start_idx = input_text.find('{')
        end_idx = input_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = input_text[start_idx : end_idx + 1]

    try:
        data = json.loads(json_str)
        
        # If a specific key is requested, try to return it
        if key_term:
            if isinstance(data, dict):
                return data.get(key_term, input_text) # Return raw text if key missing
            return input_text # Cannot get key from a list/string
            
        # If no key requested, return the full parsed data
        return data

    except json.JSONDecodeError:
        # logger.debug(f"[UTILS] JSON decode failed for {key_term}, returning raw.")
        return input_text

def meta_parser(input_text: str):
    """
    Parses custom XML-style tags: <meta>, <action>, <imagery>.
    Robust against missing tags.
    """
    if not input_text:
        return {"meta": None, "action": None, "imagery_utterance": None}

    def extract(tag, text):
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    return {
        "meta": extract("meta", input_text),
        "action": extract("action", input_text),
        "imagery_utterance": extract("imagery", input_text) 
    }

def thinking_parser(text: str, delimiter: str = "</think>") -> dict:
    """Separates <think> blocks from the final answer."""
    if not text:
        return {"thinking": "", "answer": ""}
        
    parts = text.split(delimiter, 1)
    if len(parts) == 2:
        return {
            "thinking": parts[0].strip().replace("<think>", "").strip(),
            "answer": parts[1].strip()
        }
    return {"thinking": "", "answer": text.strip()}

# --- Helper Functions for ThreadPoolExecutor ---

def load_existing_image(path: str) -> Optional[str]:
    """Opens image from disk and converts to base64."""
    if os.path.exists(path):
        try:
            with Image.open(path) as img:
                return pil_to_base64(img)
        except Exception as e:
            logger.warning(f"[UTILS] Could not load existing image at {path}: {e}")
    return None

def process_zoom(current_b64: str, save_path: Path) -> Optional[str]:
    """Decodes, Zooms (Resizes), and Saves."""
    try:
        pil_img = base64_to_pil(current_b64)
        zoomed = add_padding_to_image(pil_img, scale_factor=0.8)
        zoomed.save(save_path)
        return pil_to_base64(zoomed)
    except Exception as e:
        logger.error(f"[UTILS] Zoom error: {e}")
        return None

def process_generated_image(new_b64: str, save_path: Path) -> Optional[str]:
    """Decodes, Cleans Artifacts, and Saves."""
    try:
        img = clean_image_artifacts(base64_to_pil(new_b64))
        img.save(save_path)
        return pil_to_base64(img)
    except Exception as e:
        logger.error(f"[UTILS] Save error: {e}")
        return None

def is_not_empty_val(val):
    if pd.isna(val): return False
    return str(val).strip() != ""

def mock_creation_logic(utterance):
    val = random.random()
    if val < 0.2: return f"[ZOOM_OUT] $$$ Wide of {utterance}"
    elif val < 0.4: return f"First angle {utterance} $$$ Second angle {utterance}"
    return f"[Mock Prompt] {utterance}"

def mock_gen_logic(prompt):
    color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    img = Image.new('RGB', (128, 128), color=color)
    d = ImageDraw.Draw(img)
    d.rectangle([10,10,40,40], fill="white")
    if "[ZOOM_OUT]" in str(prompt): d.text((10,50), "ZOOM", fill="white")
    return pil_to_base64(img)

# -----------------------------------------------
