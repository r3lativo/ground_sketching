# src/helpers/utils.py

import base64
import io
import logging
from PIL import Image
import yaml
import pdb
import csv
import os
from datetime import datetime

logger = logging.getLogger(__name__)

LOG_HEADER = [
    "timestamp", "status", "initial_prompt", "final_prompt",
    "negative_prompt", "output_file", "latency_s"
]


def setup_logging(log_file='logs/app.log', level=logging.INFO):
    """Sets up basic logging to file and console."""
    # Ensure logs directory exists
    import os
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging configured.")

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
def load_config(config_path='config/server_config.yaml'):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config or 'image_gen_service' not in config:
            raise ValueError("Config file is missing 'image_gen_service' section.")
        return config
    except Exception as e:
        raise # Critical error, stop startup


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