# src/helpers/utils.py

import base64
import io
import logging
from PIL import Image
import yaml
import pdb
import csv
import os
import re
import sys
from datetime import datetime
from PIL import Image, ImageFilter
import socket
import json

logger = logging.getLogger(__name__)

LOG_HEADER = [
    "timestamp", "status", "initial_prompt", "final_prompt",
    "negative_prompt", "VLseed", "IMGseed", "true_cfg_scale",
    "num_inference_steps", "top_p", "temperature", "output_file",
    "comment",
]


def setup_logging(log_file='logs/app.log', level=logging.INFO, log_to_console=True):
    """
    Sets up basic logging to a file and optionally to the console.
    """
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Define the logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create a list of handlers
    handlers = [logging.FileHandler(log_file)]

    if log_to_console:
        handlers.append(logging.StreamHandler(sys.stdout))

    # Remove all existing handlers from the root logger.
    # Without this, you might get duplicate log messages.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure the root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )

def pil_to_base64(pil_image: Image.Image, format="PNG") -> str:
    """Converts a PIL Image to a Base64 encoded string."""
    try:
        with io.BytesIO() as buffer:
            pil_image.save(buffer, format=format)
            img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        logging.error(f"Error encoding PIL image to Base64: {e}", exc_info=True)
        raise

def base64_to_pil(base64_string: str) -> Image.Image:
    """Converts a Base64 encoded string to a PIL Image."""
    try:
        img_bytes = base64.b64decode(base64_string)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return pil_image
    except Exception as e:
        logging.error(f"Error decoding Base64 string to PIL image: {e}", exc_info=True)
        # Don't include raw base64 string in log for security/length reasons
        raise ValueError(f"Could not decode base64 string to image: {e}") from e

# --- Configuration Loading ---
def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise


def log_experiment_step(log_filepath: str, data: dict):
    """Appends a row to a CSV log file."""
    try:
        file_exists = os.path.isfile(log_filepath)
        # Ensure logs directory exists
        os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

        with open(log_filepath, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=LOG_HEADER)

            if not file_exists or os.path.getsize(log_filepath) == 0:
                writer.writeheader() # Write header only if file is new/empty

            # Ensure all keys exist, default to empty string if missing
            row_data = {key: data.get(key, "") for key in LOG_HEADER}
            # Format timestamp
            row_data["timestamp"] = datetime.now().isoformat()

            writer.writerow(row_data)

    except Exception as e:
        logger.error(f"Failed to write to log file {log_filepath}: {e}", exc_info=True)


def update_last_log_comment(log_filepath: str, comment: str):
    """
    Updates the 'comment' field of the last row in the CSV log file.
    
    This is done by reading the whole file, modifying the last entry in memory,
    and rewriting the entire file.
    """
    if not comment: # Do nothing if comment is empty
        return
    
    try:
        if not os.path.isfile(log_filepath):
            logger.warning(f"Log file {log_filepath} not found. Cannot update comment.")
            return
        
        rows = []
        fieldnames = LOG_HEADER # Default to our known header
        
        # Read all rows into memory
        with open(log_filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if reader.fieldnames: # Get fieldnames from file if it exists
                fieldnames = reader.fieldnames
                # Ensure 'comment' is a known fieldname
                if 'comment' not in fieldnames:
                    logger.warning("'comment' field not in log header. File may be from old version.")
                    # We will proceed, but DictWriter will add it as a new column
                    # which might be messy. It's better that LOG_HEADER is correct.
                    fieldnames.append('comment')
                    
            for row in reader:
                rows.append(row)
        
        if not rows:
            logger.warning(f"Log file {log_filepath} is empty. Cannot update comment.")
            return
        
        # Modify the last row
        rows[-1]['comment'] = comment
        
        # Rewrite the entire file
        with open(log_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Successfully updated comment for last entry in {log_filepath}")

    except Exception as e:
        logger.error(f"Failed to update log file {log_filepath} with comment: {e}", exc_info=True)


def clean_image_artifacts(input_image, white_threshold=180, black_threshold=50):
    """
    Cleans up artifacts in an image by clamping near-white and near-black 
    pixels to pure white or pure black.

    This is useful for cleaning up images (like masks) after a diffusion 
    pass, which might introduce "almost" white or "almost" black artifacts.

    Args:
        input_image (PIL.Image.Image): The input image to clean.
        white_threshold (int): Any RGB channel value *above* this (and on 
                               all channels) will be clamped to pure white.
        black_threshold (int): Any RGB channel value *below* this (and on 
                               all channels) will be clamped to pure black.

    Returns:
        PIL.Image.Image: A new image object with artifacts cleaned.
    """
    
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    
    # Ensure the image is in RGB mode for consistent pixel data
    img_rgb = input_image.convert("RGB")
    pixels = img_rgb.load() 

    # Iterate over every pixel
    for i in range(img_rgb.width):
        for j in range(img_rgb.height):
            
            current_color = pixels[i, j]

            # --- Optimization: Skip pixels that are already pure ---
            if current_color == WHITE or current_color == BLACK:
                continue

            # --- Clamp "almost white" pixels ---
            # If all 3 color channels are above the white threshold
            r, g, b = current_color
            if r > white_threshold and g > white_threshold and b > white_threshold:
                pixels[i, j] = WHITE
                
            # --- Clamp "almost black" pixels ---
            # If all 3 color channels are below the black threshold
            elif r < black_threshold and g < black_threshold and b < black_threshold:
                pixels[i, j] = BLACK

            # --- Note on logic ---
            # Any pixel that is not pure B/W and not in the ranges above
            # (e.g., a mid-grey (128, 128, 128) or a color (200, 50, 50))
            # will be left *unchanged* by this logic.
            
    return img_rgb

def check_server(host: str, port: int, timeout: int = 3) -> bool:
    """
    Checks if a server is reachable at a given host and port.
    """
    check_host = "127.0.0.1" if host == "0.0.0.0" else host
    
    print(f"{check_host}:{port}...", end="", flush=True)
    
    try:
        # Create a socket, set timeout, and try to connect
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((check_host, port))
        
        # 'with' statement auto-closes the socket.
        print(" [OK]")
        return True
    except Exception as e:
        # Catches timeout, connection refused, etc.
        print(f" [FAILED] ({e})")
        return False

def json_parser(input_text, key_term):
    try:
        cleaned_text = input_text.strip().replace('```json','').replace('```','')
        result_json = json.loads(cleaned_text)
        if isinstance(result_json, dict) and key_term in result_json:
            return result_json[key_term]
        else:
            logger.warning(f"Text parsed as JSON but missing '{key_term}' key. Using raw text.")
            return input_text
    except json.JSONDecodeError:
        logger.debug("Text is not JSON. Using raw text.")
        return input_text

def thinking_parser(text: str, delimiter: str = "</think>") -> dict:
    """
    Separates text into thinking and answer parts using </think> as the delimiter.

    Args:
        text: The input string containing thinking and/or an answer.

    Returns:
        A dictionary {thinking, answer}.
    """
    parts = text.split(delimiter, 1)  # Split only at the first occurrence
    output = {}

    if len(parts) == 2:
        output["thinking"] = parts[0].strip().replace("\n", " ")
        output["answer"] = parts[1].strip().replace("\n", " ")
    else:
        # Delimiter not found, assume the entire text is the answer
        output["thinking"] = ""
        output["answer"] = parts[0].strip().replace("\n", " ")

    return output

def add_padding_to_image(img_pil, scale_factor=0.8, fill_color="white"):
    """
    Scales down an image and adds padding to maintain the original size.
    """
    if not (0 < scale_factor <= 1.0):
        raise ValueError("Scale factor must be between 0 and 1.")

    try:
        # 1. Take the sizes
        original_width, original_height = img_pil.size

        # 2. Calculate new dimensions
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # 3. Resize the image
        # Use Image.LANCZOS (or Image.ANTIALIAS) for high-quality downscaling
        resized_image = img_pil.resize((new_width, new_height), Image.LANCZOS)

        # 4. & 5. Create and fill the new canvas
        final_image = Image.new("RGB", (original_width, original_height), fill_color)

        # 6. Calculate paste position
        paste_x = (original_width - new_width) // 2
        paste_y = (original_height - new_height) // 2

        # 7. Paste the resized image
        final_image.paste(resized_image, (paste_x, paste_y))

        # Return the padded image
        return final_image

    except Exception as e:
        print(f"An error occurred: {e}")
