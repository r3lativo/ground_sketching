# test_pipeline.py

import asyncio
import os
import logging
from PIL import Image
import time

# Assuming your source structure is correct
from src.api_clients import call_edit_prompt_enhancer, call_image_gen
from src.utils import setup_logging, pil_to_base64, base64_to_pil, load_config

# --- Configuration ---
# Ensure logs directory exists for this script's log
config = {}
LOG_FILE = 'logs/test_pipeline.log'
OUTPUT_DIR = 'output'
# Create dummy input files if they don't exist, or point to real test images
# Using the example from the original nunchaku script
# Make sure these images exist or replace paths
INPUT_IMAGE_PATHS = [
    "data/man.png",
    "data/puppy.png",
    "data/sofa.png",
]
INITIAL_PROMPT = "Let the man in image 1 lie on the sofa in image 3, and let the puppy in image 2 lie on the floor to sleep."
OUTPUT_FILENAME = f"test_output_{int(time.time())}.png"

# --- Setup ---
setup_logging(log_file=LOG_FILE)
logger = logging.getLogger(__name__)
os.makedirs(OUTPUT_DIR, exist_ok=True)
load_config()

async def run_test():
    """Runs a single test case through the enhance -> generate pipeline."""
    logger.info("--- Starting Test Pipeline ---")

    # 1. Load and Encode Input Images
    logger.info(f"Loading input images: {INPUT_IMAGE_PATHS}")
    input_images_pil = []
    input_images_b64 = []
    try:
        for path in INPUT_IMAGE_PATHS:
            if not os.path.exists(path):
                logger.error(f"Input image not found: {path}. Skipping test.")
                # Create dummy images if needed for basic testing:
                # img = Image.new('RGB', (60, 30), color = 'red')
                # img.save(path)
                # logger.info(f"Created dummy image at {path}")
                # input_images_pil.append(img)
                return # Exit if images not found
            img = Image.open(path).convert("RGB")
            input_images_pil.append(img)
            logger.debug(f"Loaded image {path} with size {img.size}")

        input_images_b64 = [pil_to_base64(img) for img in input_images_pil]
        logger.info(f"Successfully encoded {len(input_images_b64)} images to Base64.")

    except Exception as e:
        logger.error(f"Failed to load or encode input images: {e}", exc_info=True)
        return

    # 2. Enhance Prompt
    logger.info(f"Initial prompt: '{INITIAL_PROMPT}'")
    enhanced_prompt = None
    try:
        enhanced_prompt = await call_edit_prompt_enhancer(INITIAL_PROMPT, input_images_b64)
        if enhanced_prompt:
            logger.info(f"Enhanced prompt: '{enhanced_prompt}'")
        else:
            logger.warning("Prompt enhancement failed or returned empty. Using initial prompt.")
            enhanced_prompt = INITIAL_PROMPT # Fallback
    except Exception as e:
        logger.error(f"Error calling prompt enhancer: {e}", exc_info=True)
        logger.warning("Using initial prompt due to enhancer error.")
        enhanced_prompt = INITIAL_PROMPT # Fallback

    # 3. Generate Image
    output_image_b64 = None
    try:
        logger.info("Calling image generation service...")
        output_image_b64 = await call_image_gen(enhanced_prompt, input_images_b64)
    except Exception as e:
        logger.error(f"Error calling image generation service: {e}", exc_info=True)
        return

    # 4. Decode and Save Output
    if output_image_b64:
        logger.info("Image generation successful. Decoding and saving...")
        try:
            output_image_pil = base64_to_pil(output_image_b64)
            save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
            output_image_pil.save(save_path)
            logger.info(f"Output image saved successfully to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to decode or save output image: {e}", exc_info=True)
    else:
        logger.error("Image generation service did not return an image.")

    logger.info("--- Test Pipeline Finished ---")


if __name__ == "__main__":
    # Ensure the servers are running before executing this script
    print(f"Make sure both servers (Image Gen on port {config.get('image_gen_service',{}).get('port', 8000)} and Prompt Enhancer on port {config.get('prompt_enhancer_service',{}).get('port', 8001)}) are running.")
    print("Executing test pipeline...")
    asyncio.run(run_test())
    print("Test script finished. Check logs/test_pipeline.log and the output/ directory.")
