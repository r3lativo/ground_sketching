# interactive_edit.py

import asyncio
import os
import logging
import time
from PIL import Image
import io # Needed for dummy image

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
# Using the main logger setup, redirect output if desired
setup_logging(log_file='logs/interactive_run.log')
logger = logging.getLogger(__name__)
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def main_interactive_loop():
    """Runs the interactive image generation and editing loop."""
    logger.info("--- Starting Interactive Session ---")
    print("\nWelcome to the Interactive Image Editor!")
    print("Type 'quit' at any prompt to exit.")

    current_image_path = None
    current_image_pil = None
    step_count = 0

    while True:
        step_start_time = time.time()
        log_data = {
            "negative_prompt": "text, blurry ", # Default negative prompt used by generator
        }
        status = "fail" # Default status

        try:
            # --- Get User Input ---
            if step_count == 0:
                initial_prompt = input("Enter the initial text prompt to generate an image: ")
                if initial_prompt.lower() == 'quit': break
                log_data["initial_prompt"] = initial_prompt

                # Create a dummy blank image for the first "edit" (T2I simulation)
                logger.info("Creating dummy blank image for initial generation.")
                current_image_pil = Image.new('RGB', DUMMY_IMAGE_SIZE, color='white')
                input_images_b64 = [pil_to_base64(current_image_pil)]

                # --- Enhance Text Prompt (for initial generation) ---
                logger.info("Calling text prompt enhancer...")
                final_prompt = await call_text_prompt_enhancer(initial_prompt)
                if not final_prompt:
                    logger.warning("Text enhancement failed. Using initial prompt.")
                    final_prompt = initial_prompt # Fallback
                log_data["final_prompt"] = final_prompt

            else:
                print(f"\nCurrent image: {current_image_path}")
                edit_prompt = input("Enter the edit instruction (or 'quit'): ")
                if edit_prompt.lower() == 'quit': break
                log_data["initial_prompt"] = edit_prompt

                if current_image_pil is None:
                    logger.error("Cannot edit, no current image available.")
                    print("Error: No image to edit. Exiting.")
                    break

                input_images_b64 = [pil_to_base64(current_image_pil)]

                # --- Enhance Edit Prompt (for subsequent edits) ---
                logger.info("Calling edit prompt enhancer...")
                final_prompt = await call_edit_prompt_enhancer(edit_prompt, input_images_b64)
                if not final_prompt:
                    logger.warning("Edit enhancement failed. Using initial prompt.")
                    final_prompt = edit_prompt # Fallback
                log_data["final_prompt"] = final_prompt

            # --- Generate/Edit Image ---
            logger.info(f"Calling image generation service with final prompt: '{final_prompt[:100]}...'")
            output_image_b64 = await call_image_gen(final_prompt, input_images_b64)

            if output_image_b64:
                # --- Decode and Save ---
                try:
                    output_image_pil = base64_to_pil(output_image_b64)
                    timestamp = int(time.time())
                    output_filename = f"step_{step_count:03d}_{timestamp}.png"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    output_image_pil.save(output_path)

                    logger.info(f"Output image saved successfully to: {output_path}")
                    print(f"Image saved as: {output_path}")

                    # Update state for next iteration
                    current_image_path = output_path
                    current_image_pil = output_image_pil
                    log_data["output_file"] = output_path
                    status = "success"
                    step_count += 1

                except Exception as e:
                    logger.error(f"Failed to decode or save output image: {e}", exc_info=True)
                    print(f"Error saving image: {e}")
                    # Keep previous image state if save fails
            else:
                logger.error("Image generation service did not return an image.")
                print("Error: Image generation failed.")
                # Keep previous image state

        except KeyboardInterrupt:
            logger.warning("User interrupted session.")
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in step {step_count}: {e}", exc_info=True)
            print(f"An error occurred: {e}")
            # Optionally break or try to continue? For now, continue loop state.

        finally:
            # --- Log Step Result ---
            log_data["status"] = status
            log_data["latency_s"] = time.time() - step_start_time
            log_experiment_step(LOG_FILE, log_data)
            logger.info(f"Step {step_count-1 if status=='success' else step_count} logged with status: {status}")

    logger.info("--- Interactive Session Ended ---")
    print("\nGoodbye!")


if __name__ == "__main__":
    # Ensure the servers are running before executing this script
    print(f"Make sure both servers (Image Gen and Prompt Enhancer) are running.")
    print("Starting interactive session...")
    try:
        asyncio.run(main_interactive_loop())
    except Exception as e:
        logger.critical(f"Interactive loop failed critically: {e}", exc_info=True)
    finally:
        print("Interactive script finished.")