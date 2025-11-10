# interactive_edit.py

import asyncio
import os
import logging
import time
from PIL import Image
import io
import argparse
import re
from typing import Optional, Tuple
import sys
import datetime

from src.api_clients import (
    call_text_prompt_polisher,
    call_edit_prompt_polisher,
    call_image_gen
)
from src.utils import (
    setup_logging,
    pil_to_base64,
    base64_to_pil,
    log_experiment_step,
    update_last_log_comment,
    load_config,
    clean_image_artifacts,
    check_server
)

# --- Configuration ---
server_config = load_config(config_path='config/server_config.yaml')
interactive_config = load_config(config_path='config/interactive_config.yaml')

# --- Setup ---
os.makedirs(interactive_config.get('output_dir'), exist_ok=True)

def load_initial_state(initial_image_path: Optional[str]) -> Tuple[Optional[Image.Image], Optional[str], int]:
    """
    Loads the initial image if provided and determines the starting step count.
    """
    if not initial_image_path:
        return None, None, 0

    try:
        if not os.path.exists(initial_image_path):
            raise FileNotFoundError(f"Image file not found at {initial_image_path}")
        
        current_image_pil = Image.open(initial_image_path).convert("RGB")
        current_image_path = initial_image_path
        print(f"\nSuccessfully loaded initial image: {initial_image_path}")

        # Parse step count from filename
        filename = os.path.basename(initial_image_path)
        match = re.match(r"step_(\d+)_.*", filename)
        
        if match:
            last_step = int(match.group(1))
            step_count = last_step + 1
            print(f"Resuming from step {step_count}.")
        else:
            print(f"Could not parse step number from filename '{filename}'.")
            step_count = 1
            print("You can now start editing (starting at step 1).")
        
        return current_image_pil, current_image_path, step_count

    except Exception as e:
        print(f"\n[Error] Could not load initial image: {e}")
        print("Starting with a blank canvas instead.\n")
        return None, None, 0


async def handle_text_to_image_step(direct: bool) -> Tuple[Optional[str], Optional[list], Optional[str], bool]:
    """
    Handles the T2I step: gets prompt, creates dummy image, and polishes prompt.
    Returns: (step_start_time, final_prompt, initial_prompt, user_quit)
    """
    initial_prompt = input("Enter the initial text prompt to generate an image: ")
    if initial_prompt.lower() == 'quit':
        return time.time(), None, None, True
    
    step_start_time = time.time()

    final_prompt = initial_prompt
    if not direct:
        print("Calling text prompt polisher...")
        final_prompt = await call_text_prompt_polisher(initial_prompt)
    else:
        print("Bypassing text prompt polisher!")
    
    if not final_prompt:
        print("Text enhancement failed. Using initial prompt.")
        final_prompt = initial_prompt

    return step_start_time, final_prompt, initial_prompt, False


async def handle_text_and_image_to_image_step(direct: bool, current_image_pil: Image.Image) -> Tuple[Optional[str], Optional[list], Optional[str], bool]:
    """
    Handles the TI2I step: gets prompt, uses current image, and polishes prompt.
    Returns: (step_start_time, final_prompt, input_images_b64, initial_prompt, user_quit)
    """
    edit_prompt = input("Enter the edit instruction (or 'quit'): ")
    if edit_prompt.lower() == 'quit':
        return time.time(), None, None, None, True
    
    step_start_time = time.time()

    input_images_b64 = [pil_to_base64(current_image_pil)]

    final_prompt = edit_prompt
    if not direct:
        print("Calling edit prompt polisher...")
        final_prompt = await call_edit_prompt_polisher(edit_prompt, input_images_b64)
    else:
        print("Bypassing edit prompt polisher!")

    if not final_prompt:
        print("Edit enhancement failed. Using initial prompt.")
        final_prompt = edit_prompt
    
    return step_start_time, final_prompt, input_images_b64, edit_prompt, False


def process_and_save_output(output_image_b64: str, no_cleanup: bool, step_count: int) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Decodes, cleans (if not no_cleanup), and saves the output image.
    Returns: (output_pil_image, output_path) or (None, None) on failure.
    """
    try:
        output_image_pil = base64_to_pil(output_image_b64)
        
        cleaned_output_image_pil = output_image_pil
        if not no_cleanup:
            print("Cleaning image artifacts...")
            cleaned_output_image_pil = clean_image_artifacts(output_image_pil)
        else:
            print("Skipping artifact cleanup (--no_cleanup).")

        timestamp_str = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
        output_filename = f"step_{step_count:03d}_{timestamp_str}.png"
        output_path = os.path.join(interactive_config.get('output_dir'), output_filename)
        
        cleaned_output_image_pil.save(output_path)
        print(f"Output image saved successfully to: {output_path}")
        
        return cleaned_output_image_pil, output_path
        
    except Exception as e:
        print(f"Error saving image: {e}")
        return None, None


def log_and_print_step_summary(log_data: dict, status: str, step_count: int, step_start_time: float):
    """
    Logs the step data to the experiment file and prints the time summary.
    """
    log_data["status"] = status
    experiment_file = interactive_config.get('experiment_file')
    
    # Ensure experiment directory exists
    os.makedirs(os.path.dirname(experiment_file), exist_ok=True)
    
    log_experiment_step(experiment_file, log_data)
    
    # Get the step number that just finished
    current_logged_step = step_count if status == 'fail' else step_count - 1

    step_end_time = time.time()
    duration = step_end_time - step_start_time
    print(f"\nStep {current_logged_step} completed in {duration:.2f} seconds.")


# --- Refactored Main Loop ---

async def main_interactive_loop(initial_image_path: Optional[str] = None, direct: bool = False, no_cleanup: bool = True):
    """Runs the interactive image generation and editing loop."""
    print("\nWelcome to the Interactive Image Editor!")
    print("Type 'quit' at any prompt to exit.")

    # --- Pre-loop: Load initial state ---
    current_image_pil, current_image_path, step_count = load_initial_state(initial_image_path)

    # --- Main Loop ---
    while True:
        user_quit = False
        log_data = {
            "negative_prompt": server_config["image_gen_client"]["negative_prompt"],
            "true_cfg_scale": server_config["image_gen_client"]["true_cfg_scale"],
            "num_inference_steps": server_config["image_gen_client"]["num_inference_steps"],
            "IMGseed": server_config["image_gen_client"]["seed"],
            "VLseed": server_config["prompt_polisher_client"]["seed"],
            "top_p": server_config["prompt_polisher_client"]["top_p"],
            "temperature": server_config["prompt_polisher_client"]["temperature"],
        }
        status = "fail"

        try:
            # --- 0. Assume no image  ---
            input_images_b64 = None
            
            # --- 1. Get Prompt and Inputs ---
            if step_count == 0:
                # Text-to-Image block
                step_start_time, final_prompt, initial_prompt, user_quit = await handle_text_to_image_step(direct)
            else:
                # Image-to-Image Edit block
                if current_image_pil is None:
                    print("Error: No image to edit. Exiting.")
                    break
                
                print(f"\nCurrent image: {current_image_path} (editing for step {step_count})")
                step_start_time, final_prompt, input_images_b64, initial_prompt, user_quit = await handle_text_and_image_to_image_step(direct, current_image_pil)
            
            if user_quit:
                break

            log_data["initial_prompt"] = initial_prompt
            log_data["final_prompt"] = final_prompt

            # --- 2. Generate Image ---
            print("Calling image generator...")
            output_image_b64 = await call_image_gen(final_prompt, input_images_b64)

            # --- 3. Process and Save Output ---
            if output_image_b64:
                new_image_pil, new_image_path = process_and_save_output(output_image_b64, no_cleanup, step_count)

                if new_image_pil:
                    # Update state for next iteration
                    current_image_path = new_image_path
                    current_image_pil = new_image_pil
                    log_data["output_file"] = new_image_path
                    status = "success"
                    step_count += 1
                else:
                    print("Error processing or saving image.")
            else:
                print("Error: Image generation failed.")

        except KeyboardInterrupt:
            print("\nExiting...")
            user_quit = True
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        finally:
            # --- 4. Log and Summarize ---
            if not user_quit:
                
                # First, save the log as-is
                log_and_print_step_summary(log_data, status, step_count, step_start_time)

                # THEN, if successful, ask for the optional comment
                if status == "success":
                    try:
                        comment = input("###\nOPTIONAL COMMENT on output (press Enter to skip): ")
                        if comment:
                            # If comment provided, call the update function
                            experiment_file = interactive_config.get('experiment_file')
                            update_last_log_comment(experiment_file, comment)
                    
                    except (KeyboardInterrupt, EOFError):
                        # If user quits *during* comment prompt, the log is already
                        # safely saved. We just print a newline and let the
                        # loop exit or continue.
                        print("\nSkipping comment. Log was already saved.")
                        pass # The main loop will handle the exit
                
    print("\nGoodbye!")


# --- Argument Parser and Main Execution ---

def get_args():
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
    parser.add_argument(
        "-d", "--direct",
        action='store_true',
        help="Bypass the prompt polisher if called (ergo, if true)."
    )
    parser.add_argument(
        "--no-cleanup",
        action='store_true',
        help="Bypass artifact cleanup and save the raw model output."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # Check Image Gen Server
    img_gen_ok = check_server(
        server_config["image_gen_service"]["host"],
        server_config["image_gen_service"]["port"]
    )
    
    # Check Prompt Polisher Server if it is needed
    if not args.direct:
        polisher_ok = check_server(
            server_config["prompt_polisher_service"]["host"],
            server_config["prompt_polisher_service"]["port"]
        )

    if not (img_gen_ok and polisher_ok):
        print("\n[Error] One or more required services are down. Exiting.")
        sys.exit(1) # Exit with an error code
    
    try:
        asyncio.run(main_interactive_loop(initial_image_path=args.image, direct=args.direct, no_cleanup=args.no_cleanup))
    except Exception as e:
        print(f"An unexpected critical error occurred: {e}")
