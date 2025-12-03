import pandas as pd
import argparse
import asyncio
import base64
import io
import json
from pathlib import Path
from PIL import Image
import os
from typing import List

from utils import (
    pil_to_base64,
    check_server,
    load_config
)
from api_clients import (
    call_prompt_summarizer,
    call_image_captioner,
    call_fact_checker
)

server_config = load_config(config_path='config/server_config.yaml')

def get_args():
    parser = argparse.ArgumentParser(
        description="Run an interactive image editing session.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-i", "--image",
        type=str,
        default=None,
        help="Path to an image to summarize.\n"
             "If the filename is 'step_NNN_...', editing will resume at step NNN+1.\n"
             "If not provided, the session starts with text-to-image."
    )
    parser.add_argument(
        "-p", "--prompts",
        type=str,
        default=None,
        help="Path to the initial prompts to summarize.\n"
    )
    args = parser.parse_args()
    return args

async def validate_image(image_path: str, prompts_to_summarize: List[str]):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return False

    try:
        current_image_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return False

    input_images_b64 = [pil_to_base64(current_image_pil)]
    
    print("Generating image caption...")
    caption = await call_image_captioner(input_images_b64)
    caption = caption.strip()
    if not caption:
        print("Failed to generate caption.")
        return False
    
    print("Summarizing prompts...")
    print('???', prompts_to_summarize)
    summaries = await call_prompt_summarizer(prompts_to_summarize)

    if not summaries:
        print("Failed to summarize prompts.")
        return False
    summaries= summaries.split('\\n')
    print('!!!!!',summaries)
    print(f"Checking facts against caption {caption}")

    for summary in summaries:
        summary = summary.strip()
        if not summary:
            continue
        is_similar_response = await call_fact_checker(summary, caption)

        if not is_similar_response:
            print("Fact check returned None/Error.")
            return False

        print(f"{summary} : {is_similar_response}\n")
        if 'false' in is_similar_response.lower():
            print(f"mismatch found on fact: {summary}")
            return False
    
    return True

if __name__ == '__main__':
    args = get_args()

    polisher_ok = check_server(
            server_config["prompt_polisher_service"]["host"],
            server_config["prompt_polisher_service"]["port"]
        )
    
    if not polisher_ok:
        print("\n[Error] VLLM server is down. Exiting.")
        sys.exit(1) # Exit with an error code

    try:
        result = asyncio.run(validate_image(args.image, args.prompts.split('\n')))
        print(f"The validation result is: {result}")
    except KeyboardInterrupt:
        print("\nValidation cancelled.")







    