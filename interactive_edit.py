# interactive_edit.py

import asyncio
import os
import logging
import time
from PIL import Image
import io
import argparse
import re
from typing import Optional, Tuple, List
import sys
import datetime

from src.api_clients import (
    call_prompt_polisher,
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

# --- Helper Functions ---
def _get_user_input(prompt_message: str) -> Tuple[Optional[str], bool]:
    """Gets text input from the user and checks for 'quit' command."""
    # Adds a newline for better spacing in the terminal
    user_input = input(f"\n{prompt_message} (or 'quit'): ")
    if user_input.lower() == 'quit':
        return None, True
    return user_input, False

async def _get_polished_prompt(
    initial_prompt: str,
    images_b64: Optional[List[str]] = None,
    direct: bool = False
) -> str:
    """
    Handles the logic of either bypassing or calling the prompt polisher
    and includes fallback to the initial prompt.
    """
    # This function now clearly explains what it's doing.
    if direct:
        print("Bypassing prompt polisher (--direct).")
        return initial_prompt

    print("Calling prompt polisher...")
    try:
        # Use the consolidated function. It handles None for images correctly.
        polished_prompt = await call_prompt_polisher(
            utterance=initial_prompt,
            images=images_b64
        )
        
        if polished_prompt:
            print(f"Polisher successful. Using new prompt.")
            return polished_prompt
        else:
            # Clearer error message
            print("Polishing failed or returned empty. Using initial prompt.")
            return initial_prompt
            
    except Exception as e:
        print(f"An error occurred during polishing: {e}. Using initial prompt.")
        return initial_prompt

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


async def handle_text_to_image_step(direct: bool) -> Tuple[Optional[str], Optional[str], Optional[float], bool]:
    """
    Handles the T2I step: gets prompt and polishes it.
    Returns: (final_prompt, initial_prompt, step_start_time, user_quit)
    """
    initial_prompt, user_quit = _get_user_input("Enter the initial text prompt to generate an image")
    if user_quit:
        return None, None, None, True
    
    # Start timer *after* user input, to time the API calls
    step_start_time = time.time()
    
    final_prompt = await _get_polished_prompt(
        initial_prompt=initial_prompt,
        direct=direct
    )
    
    return final_prompt, initial_prompt, step_start_time, False


async def handle_text_and_image_to_image_step(
    direct: bool, 
    current_image_pil: Image.Image
) -> Tuple[Optional[str], Optional[List[str]], Optional[str], Optional[float], bool]:
    """
    Handles the TI2I step: gets prompt, uses current image, and polishes prompt.
    Returns: (final_prompt, input_images_b64, initial_prompt, step_start_time, user_quit)
    """
    edit_prompt, user_quit = _get_user_input("Enter the edit instruction")
    if user_quit:
        return None, None, None, None, True

    # Start timer *after* user input
    step_start_time = time.time()
    
    # The API client expects raw b64, not the data URI
    input_images_b64 = [pil_to_base64(current_image_pil)]

    final_prompt = await _get_polished_prompt(
        initial_prompt=edit_prompt,
        images_b64=input_images_b64,
        direct=direct
    )
    
    return final_prompt, input_images_b64, edit_prompt, step_start_time, False


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
            "VLseed": server_config["vlm_client"]["seed"],
            "top_p": server_config["vlm_client"]["top_p"],
            "temperature": server_config["vlm_client"]["temperature"],
        }
        status = "fail"

        try:
            # --- 0. Assume no image  ---
            input_images_b64 = None
            
            # --- 1. Get Prompt and Inputs ---
            if step_count == 0:
                # Text-to-Image block
                # Note the new variable order from the refactored function
                final_prompt, initial_prompt, step_start_time, user_quit = await handle_text_to_image_step(direct)
            else:
                # Image-to-Image Edit block
                if current_image_pil is None:
                    print("Error: No image to edit. Exiting.")
                    break
                
                print(f"\nCurrent image: {current_image_path} (editing for step {step_count})")
                # Note the new variable order from the refactored function
                final_prompt, input_images_b64, initial_prompt, step_start_time, user_quit = await handle_text_and_image_to_image_step(direct, current_image_pil)
            
            if user_quit:
                break
            
            # This check is now needed since step_start_time is set *after* the quit check
            if not step_start_time:
                break # Should not happen, but good safety
                
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
    failed_services = []

    # --- Check Image Gen Server (Always required) ---
    img_gen_conf = server_config["image_gen_service"]
    if not check_server(img_gen_conf["host"], img_gen_conf["port"]):
        failed_services.append(
            f"Image Gen service at http://{img_gen_conf['host']}:{img_gen_conf['port']}"
        )

    # --- Check Prompt Polisher (Only if not --direct) ---
    if not args.direct:
        polisher_conf = server_config["vlm_service"]
        if not check_server(polisher_conf["gateway"]["host"], polisher_conf["gateway"]["port"]):
            failed_services.append(
                f"Prompt Polisher service at http://{polisher_conf["gateway"]['host']}:{polisher_conf["gateway"]['port']}"
            )

    # --- Report results ---
    if failed_services:
        print("\n[Error] One or more required services are down:")
        for service_msg in failed_services:
            print(f"- {service_msg}")
        
        print("Exiting.")
        sys.exit(1)
    
    print("\nAll required services are running.")

    # --- Run Main Loop ---
    try:
        asyncio.run(main_interactive_loop(initial_image_path=args.image, direct=args.direct, no_cleanup=args.no_cleanup))
    except Exception as e:
        print(f"An unexpected critical error occurred: {e}")
