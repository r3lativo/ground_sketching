# interactive_edit.py

import asyncio
import os
import logging
import time
from PIL import Image
import io
import argparse
import re
from typing import Optional
import sys
import datetime
import socket

from src.api_clients import (
    call_text_prompt_polisher,
    call_edit_prompt_polisher,
    call_image_gen
)
from src.utils import setup_logging, pil_to_base64, base64_to_pil, log_experiment_step, load_config, clean_image_artifacts

# --- Configuration ---
server_config = load_config(config_path='config/server_config.yaml')
interactive_config = load_config(config_path='config/interactive_config.yaml')
dummy_img_height = interactive_config.get('dummy_img_height')
dummy_img_size = (dummy_img_height, dummy_img_height)

# --- Setup ---
os.makedirs(interactive_config.get('output_dir'), exist_ok=True)

def check_server(host: str, port: int, timeout: int = 3) -> bool:
    """
    Synchronously checks if a server is reachable at a given host and port.
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

async def main_interactive_loop(initial_image_path: Optional[str] = None, direct: bool = False, raw: bool = False):
    """Runs the interactive image generation and editing loop."""
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
            
            current_image_pil = Image.open(initial_image_path).convert("RGB")
            current_image_path = initial_image_path
            
            print(f"\nSuccessfully loaded initial image: {initial_image_path}")

            # --- Parse step count from filename ---
            filename = os.path.basename(initial_image_path)
            # Try to match the format 'step_001_...'
            match = re.match(r"step_(\d+)_.*", filename)
            
            if match:
                last_step = int(match.group(1))
                step_count = last_step + 1 # Start at the *next* step
                print(f"Resuming from step {step_count}.")
            else:
                print(f"Could not parse step number from filename '{filename}'. Starting edits at step 1.")
                step_count = 1 # Default behavior if image is loaded but name format is unknown
                print("You can now start editing (starting at step 1).")

        except Exception as e:
            print(f"\n[Error] Could not load initial image: {e}")
            print("Starting with a blank canvas instead.\n")
            # On failure, step_count remains 0, and loop starts at T2I

    # --- Main Loop ---
    while True:
        user_quit = False

        step_start_time = time.time()
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
            # --- Get User Input ---
            if step_count == 0:
                # Text-to-Image block
                initial_prompt = input("Enter the initial text prompt to generate an image: ")
                
                if initial_prompt.lower() == 'quit':
                    user_quit = True
                    break

                log_data["initial_prompt"] = initial_prompt

                print("Creating dummy blank image for initial generation.")
                current_image_pil = Image.new('RGB', dummy_img_size, color='white')
                input_images_b64 = [pil_to_base64(current_image_pil)]

                if not direct:
                    print("Calling text prompt polisher...")
                    final_prompt = await call_text_prompt_polisher(initial_prompt)
                    print(f"Final prompt:\n{final_prompt}")
                else:
                    print("Bypassing text prompt polisher!")
                    final_prompt = initial_prompt
                
                if not final_prompt:
                    print("Text enhancement failed. Using initial prompt.")
                    final_prompt = initial_prompt
                log_data["final_prompt"] = final_prompt

            else:
                # Image-to-Image Edit block
                print(f"\nCurrent image: {current_image_path} (editing for step {step_count})")
                edit_prompt = input("Enter the edit instruction (or 'quit'): ")

                if edit_prompt.lower() == 'quit':
                    user_quit = True
                    break
                
                log_data["initial_prompt"] = edit_prompt

                if current_image_pil is None:
                    print("Error: No image to edit. Exiting.")
                    break

                input_images_b64 = [pil_to_base64(current_image_pil)]

                if not direct:
                    print("Calling edit prompt polisher...")
                    final_prompt = await call_edit_prompt_polisher(edit_prompt, input_images_b64)
                else:
                    print("Bypassing edit prompt polisher!")
                    final_prompt = edit_prompt

                if not final_prompt:
                    print("Edit enhancement failed. Using initial prompt.")
                    final_prompt = edit_prompt
                log_data["final_prompt"] = final_prompt

            # --- Generate/Edit Image ---
            print("Calling image generator...")
            output_image_b64 = await call_image_gen(final_prompt, input_images_b64)

            if output_image_b64:

                # --- Decode and Save ---
                try:
                    output_image_pil = base64_to_pil(output_image_b64)
                    
                    if not raw:
                        # Remove artifacts from the image
                        cleaned_output_image_pil = clean_image_artifacts(output_image_pil)
                    else:
                        cleaned_output_image_pil = output_image_pil

                    timestamp_str = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
                    # The filename will now correctly use the incremented step_count
                    output_filename = f"step_{step_count:03d}_{timestamp_str}.png"
                    output_path = os.path.join(interactive_config.get('output_dir'), output_filename)
                    cleaned_output_image_pil.save(output_path)

                    print(f"Output image saved successfully to: {output_path}")

                    # Update state for next iteration
                    current_image_path = output_path
                    current_image_pil = cleaned_output_image_pil
                    log_data["output_file"] = output_path
                    status = "success"
                    step_count += 1 # Increment for the *next* loop

                except Exception as e:
                    print(f"Error saving image: {e}")
            else:
                print("Error: Image generation failed.")

        except KeyboardInterrupt:
            print("\nExiting...")
            user_quit = True
            break
        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            if not user_quit:
                log_data["status"] = status
                experiment_file = interactive_config.get('experiment_file')
                os.makedirs(os.path.dirname(experiment_file), exist_ok=True)
                log_experiment_step(experiment_file, log_data)
                # Correct logging of the step that just finished
                current_logged_step = step_count if status == 'fail' else step_count - 1

                step_end_time = time.time()
                duration = step_end_time - step_start_time
                print(f"\nStep {current_logged_step} completed in {duration:.2f} seconds.")

    print("\nGoodbye!")


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
        help="Bypass the prompt polisher if called (ergo, if true)"
    )
    parser.add_argument(
        "-r", "--raw",
        action='store_true',
        help="Bypass the artifact cleanup if called (ergo, if true)"
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
    
    # Check Prompt Polisher Server
    polisher_ok = check_server(
        server_config["prompt_polisher_service"]["host"],
        server_config["prompt_polisher_service"]["port"]
    )

    if not (img_gen_ok and polisher_ok):
        print("\n[Error] One or more required services are down. Exiting.")
        sys.exit(1) # Exit with an error code
    
    try:
        asyncio.run(main_interactive_loop(initial_image_path=args.image, direct=args.direct, raw=args.raw))
    except Exception as e:
        logger.critical(f"Interactive loop failed critically: {e}", exc_info=True)
