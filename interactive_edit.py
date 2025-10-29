# interactive_edit.py

import asyncio
import os
import logging
import time
from PIL import Image
import io # Needed for dummy image
import argparse
import re # <-- Added for filename parsing
from typing import Optional

# Assuming your source structure is correct
from src.api_clients import (
    call_text_prompt_enhancer,
    call_edit_prompt_enhancer,
    call_image_gen
)
from src.utils import setup_logging, pil_to_base64, base64_to_pil, log_experiment_step

# --- Configuration ---
LOG_FILE = 'logs/interactive_session.csv'
OUTPUT_DIR = 'output/interactive'
DUMMY_IMAGE_SIZE = (512, 512) # Small size for the initial blank image

# --- Setup ---
setup_logging(log_file='logs/interactive_run.log')
logger = logging.getLogger(__name__)
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def main_interactive_loop(initial_image_path: Optional[str] = None):
    """Runs the interactive image generation and editing loop."""
    logger.info("--- Starting Interactive Session ---")
    print("\nWelcome to the Interactive Image Editor!")
    print("Type 'quit' at any prompt to exit.")

    current_image_path = None
    current_image_pil = None
    step_count = 0 # Default start

    # --- Pre-loop: Load initial image if provided ---
    if initial_image_path:
        try:
            if not os.path.exists(initial_image_path):
                raise FileNotFoundError(f"Image file not found at {initial_image_path}")
            
            logger.info(f"Loading initial image from: {initial_image_path}")
            current_image_pil = Image.open(initial_image_path).convert("RGB")
            current_image_path = initial_image_path
            
            print(f"\nSuccessfully loaded initial image: {initial_image_path}")

            # --- New logic to parse step count from filename ---
            filename = os.path.basename(initial_image_path)
            # Try to match the format 'step_001_...'
            match = re.match(r"step_(\d+)_.*", filename)
            
            if match:
                last_step = int(match.group(1))
                step_count = last_step + 1 # Start at the *next* step
                logger.info(f"Parsed filename. Resuming from step {step_count}.")
                print(f"Resuming from step {step_count}.")
            else:
                logger.warning(f"Could not parse step number from filename '{filename}'. Starting edits at step 1.")
                step_count = 1 # Default behavior if image is loaded but name format is unknown
                print("You can now start editing (starting at step 1).")
            # --- End new logic ---

        except Exception as e:
            logger.error(f"Failed to load initial image '{initial_image_path}': {e}", exc_info=True)
            print(f"\n[Error] Could not load initial image: {e}")
            print("Starting with a blank canvas instead.\n")
            # On failure, step_count remains 0, and loop starts at T2I

    # --- Main Loop ---
    while True:
        step_start_time = time.time()
        log_data = {
            "negative_prompt": "text, blurry ",
        }
        status = "fail"

        try:
            # --- Get User Input ---
            if step_count == 0:
                # Text-to-Image block
                initial_prompt = input("Enter the initial text prompt to generate an image: ")
                if initial_prompt.lower() == 'quit': break
                log_data["initial_prompt"] = initial_prompt

                logger.info("Creating dummy blank image for initial generation.")
                current_image_pil = Image.new('RGB', DUMMY_IMAGE_SIZE, color='white')
                input_images_b64 = [pil_to_base64(current_image_pil)]

                logger.info("Calling text prompt enhancer...")
                final_prompt = await call_text_prompt_enhancer(initial_prompt)
                if not final_prompt:
                    logger.warning("Text enhancement failed. Using initial prompt.")
                    final_prompt = initial_prompt
                log_data["final_prompt"] = final_prompt

            else:
                # Image-to-Image Edit block
                print(f"\nCurrent image: {current_image_path} (editing for step {step_count})")
                edit_prompt = input("Enter the edit instruction (or 'quit'): ")
                if edit_prompt.lower() == 'quit': break
                log_data["initial_prompt"] = edit_prompt

                if current_image_pil is None:
                    logger.error("Cannot edit, no current image available.")
                    print("Error: No image to edit. Exiting.")
                    break

                input_images_b64 = [pil_to_base64(current_image_pil)]

                logger.info("Calling edit prompt enhancer...")
                final_prompt = await call_edit_prompt_enhancer(edit_prompt, input_images_b64)
                if not final_prompt:
                    logger.warning("Edit enhancement failed. Using initial prompt.")
                    final_prompt = edit_prompt
                log_data["final_prompt"] = final_prompt

            # --- Generate/Edit Image ---
            logger.info(f"Calling image generation service with final prompt: '{final_prompt[:100]}...'")
            output_image_b64 = await call_image_gen(final_prompt, input_images_b64)

            if output_image_b64:
                # CLEANUP
                def cleanup(outpt_img):
                    from PIL import Image, ImageFilter
                    WHITE = (255, 255, 255)
                    BLACK = (0, 0, 0)
                    allowed_colors=[WHITE, BLACK]
                    img = outpt_img.convert("RGB")
                    pixels = img.load() 
                    for i in range(img.width):
                        for j in range(img.height):
                            current_color = pixels[i, j]
                            if current_color not in allowed_colors:
                                if current_color[0] > 180 and current_color[1] > 180 and current_color[2] > 180:
                                    pixels[i, j] = WHITE
                                elif current_color[0] < 50 and current_color[1] < 50 and current_color[2] < 50:
                                    pixels[i, j] = BLACK
                                # else:
                                #     pixels[i, j] = WHITE
                    return img

                # --- Decode and Save ---
                try:
                    output_image_pil = base64_to_pil(output_image_b64)
                    cleaned_output_image_pil = cleanup(output_image_pil)

                    timestamp = int(time.time())
                    # The filename will now correctly use the incremented step_count
                    output_filename = f"step_{step_count:03d}_{timestamp}.png"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    cleaned_output_image_pil.save(output_path)

                    logger.info(f"Output image saved successfully to: {output_path}")
                    print(f"Image saved as: {output_path}")

                    # Update state for next iteration
                    current_image_path = output_path
                    current_image_pil = cleaned_output_image_pil
                    log_data["output_file"] = output_path
                    status = "success"
                    step_count += 1 # Increment for the *next* loop

                except Exception as e:
                    logger.error(f"Failed to decode or save output image: {e}", exc_info=True)
                    print(f"Error saving image: {e}")
            else:
                logger.error("Image generation service did not return an image.")
                print("Error: Image generation failed.")

        except KeyboardInterrupt:
            logger.warning("User interrupted session.")
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in step {step_count}: {e}", exc_info=True)
            print(f"An error occurred: {e}")

        finally:
            log_data["status"] = status
            log_data["latency_s"] = time.time() - step_start_time
            log_experiment_step(LOG_FILE, log_data)
            # Correct logging of the step that just finished
            current_logged_step = step_count if status == 'fail' else step_count - 1
            logger.info(f"Step {current_logged_step} logged with status: {status}")

    logger.info("--- Interactive Session Ended ---")
    print("\nGoodbye!")


if __name__ == "__main__":
    print(f"Make sure both servers (Image Gen and Prompt Enhancer) are running.")
    
    parser = argparse.ArgumentParser(
        description="Run an interactive image editing session.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-i", "--image",
        type=str,
        default=None,
        help="Optional path to an initial image to start editing.\n"
             "If the filename is 'step_NNN_...', editing will resume at step NNN+1.\n"
             "If not provided, the session starts with text-to-image."
    )
    args = parser.parse_args()

    print("Starting interactive session...")
    try:
        asyncio.run(main_interactive_loop(initial_image_path=args.image))
    except Exception as e:
        logger.critical(f"Interactive loop failed critically: {e}", exc_info=True)
    finally:
        print("Interactive script finished.")
