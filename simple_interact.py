"""
A simple tool to interact directly with the prompt polisher services.

- If you run it without arguments, it will use the Text-to-Text polisher.
  (python simple_polisher_tool.py)

- If you provide an image, it will use the Text-and-Image-to-Text (edit) polisher.
  (python simple_polisher_tool.py -i /path/to/your/image.png)
"""

import asyncio
import os
import sys
import argparse
from PIL import Image
from typing import Optional

from src.api_clients import call_prompt_polisher
from src.utils import (
    load_config,
    pil_to_base64,
    check_server
)

# --- Configuration ---
server_config = load_config(config_path='config/server_config.yaml')


def load_image(image_path: Optional[str]) -> Optional[Image.Image]:
    """Loads an image if the path is valid."""
    if not image_path:
        return None
    try:
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return None
        
        image_pil = Image.open(image_path).convert("RGB")
        print(f"\nSuccessfully loaded image: {image_path}")
        return image_pil
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

async def run_polisher_loop(image_pil: Optional[Image.Image]):
    """Main interactive loop for polishing prompts."""
    
    image_b64_list = None
    if image_pil:
        print("--- Mode: Edit Prompt Polisher (Text + Image) ---")
        # The edit polisher API expects a list of base64 images
        image_b64_list = [pil_to_base64(image_pil)]
    else:
        print("--- Mode: Text Prompt Polisher (Text Only) ---")

    while True:
        try:
            # Set the correct prompt text based on the mode
            if image_pil:
                prompt = input("Enter EDIT instruction (or 'quit'): ")
            else:
                prompt = input("Enter TEXT prompt (or 'quit'): ")
            
            if prompt.lower() == 'quit':
                break
            if not prompt.strip():
                continue

            print("... Calling prompt polisher ...")
            
            polished_prompt = None
            if image_pil:
                # Use Edit Polisher (Text + Image)
                polished_prompt = await call_prompt_polisher(prompt, image_b64_list)
            else:
                # Use Text Polisher (Text only)
                polished_prompt = await call_prompt_polisher(prompt)

            if polished_prompt:
                print("\n=== Polished Prompt ===")
                print(polished_prompt)
                print("=======================\n")
            else:
                print("Error: Polishing failed or returned an empty result.\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            break
    
    print("\nGoodbye!")

# --- Argument Parser and Main Execution ---

def get_args():
    parser = argparse.ArgumentParser(
        description="Interact directly with the VL prompt polisher.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-i", "--image",
        type=str,
        default=None,
        help="Optional path to an image.\n"
             "If provided, uses the 'edit' polisher (text+image).\n"
             "If omitted, uses the 'text' polisher (text only)."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Check Polisher Server
    print("Checking prompt polisher service...")
    polisher_ok = check_server(
        server_config["vlm_service"]["gateway"]["host"],
        server_config["vlm_service"]["gateway"]["port"]
    )

    if not polisher_ok:
        print("\n[Error] The prompt polisher service is down. Exiting.")
        sys.exit(1)
    
    print("Service is up.")
    
    # Load the image (if any)
    image = load_image(args.image)
    
    try:
        # Run the main loop
        asyncio.run(run_polisher_loop(image))
    except Exception as e:
        print(f"An unexpected critical error occurred: {e}")
