# src/api_clients.py

import httpx
import logging
import yaml
import os
import json
from typing import List, Optional, Dict, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape
import torch

from src.utils import setup_logging, load_config

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = logging.getLogger(__name__)

try:
    # Load Server Config
    config = load_config(config_path='config/server_config.yaml')

    # Construct the base URL for the OpenAI compatible endpoint
    image_gen_url = f"http://{config['image_gen_service']['host']}:{config['image_gen_service']['port']}"
    polisher_api_base_url = f"http://{config['prompt_polisher_service']['host']}:{config['prompt_polisher_service']['port']}"
    polisher_model_id = config['prompt_polisher_service']['model_id']

    logger.info(f"Image Gen API URL: {image_gen_url}")
    logger.info(f"Prompt Polisher API Base URL: {polisher_api_base_url}")
    logger.info(f"Using Polisher Model ID: {polisher_model_id}")

    # Load Jinja Templates for System Prompts
    env = Environment(
        loader=FileSystemLoader(config['prompt_polisher_client']['jinja_env']),
        autoescape=select_autoescape(['html', 'xml'])
    )
    polish_template = env.get_template(config['prompt_polisher_client']['polish_template'])
    edit_template = env.get_template(config['prompt_polisher_client']['edit_template'])
    
    # Render templates
    polish_system_prompt = polish_template.render().strip()
    edit_system_prompt = edit_template.render().strip()
    logger.info("System prompts loaded from Jinja2 templates.")

except FileNotFoundError:
    logger.error(f"Configuration or template file not found in {config_PATH} or config/. Using defaults.")
except KeyError as e:
    logger.error(f"Missing key in configuration file {config_PATH}: {e}. Using defaults.")
except Exception as e:
    logger.error(f"Error loading config or templates: {e}", exc_info=True)


# --- Image Generation Client ---

async def call_image_gen(prompt: str, base64_images: List[str], timeout: int = 120) -> Optional[str]:
    """Calls the image generation service API."""
    endpoint = f"{image_gen_url}/img_generate"
    image_gen_config = config.get('image_gen_client', {})
    payload = {
        "prompt": prompt,
        "images": base64_images,
        "seed": image_gen_config.get("seed"),
        "true_cfg_scale": image_gen_config.get("true_cfg_scale"),
        "negative_prompt": image_gen_config.get("negative_prompt"),
        "num_inference_steps": image_gen_config.get("num_inference_steps"),
    }
    logger.info(f"Sending request to Image Gen API: {endpoint} with prompt '{prompt[:50]}...'")
    try:
        async with httpx.AsyncClient() as client_http:
            response = await client_http.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            if "image" in result and isinstance(result["image"], str):
                logger.info("Image generation successful.")
                return result["image"]
            else:
                logger.error(f"Image Gen API returned unexpected response format: {result}")
                return None
    except httpx.HTTPStatusError as e:
        logger.error(f"Image Gen API request failed with status {e.response.status_code}: {e.response.text}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Error connecting to Image Gen API at {endpoint}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during image generation call: {e}", exc_info=True)
        return None


# --- Prompt Polisher Client (Text-Only Beautification via HTTPX) ---

async def call_text_prompt_polisher(
    initial_prompt: str,
    #magic_prompt: str,
    #max_tokens: int,
    #temperature: float,
    timeout: int = 120
) -> Optional[str]:
    """
    Calls the vLLM OpenAI-compatible API via HTTPX to rewrite a text-only prompt.
    """
    endpoint = f"{polisher_api_base_url}/generate"
    logger.info(f"Sending request to Prompt Polisher (Text): {endpoint} with prompt '{initial_prompt[:50]}...'")
    prompt_polisher_client_config = config["prompt_polisher_client"]
    messages = [
        {"role": "system", "content": polish_system_prompt},
        {"role": "user", "content": initial_prompt}
    ]

    payload = {
        "messages": messages,
        "seed": prompt_polisher_client_config["seed"],
        "top_p": prompt_polisher_client_config["top_p"],
        "temperature": prompt_polisher_client_config["temperature"],
        "max_tokens": prompt_polisher_client_config["max_tokens"],
    }

    try:
        async with httpx.AsyncClient() as client_http:
            response = await client_http.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()

            # Navigate the OpenAI API response structure
            if result.get("text") and len(result["text"]) > 0:
                # message = result.get("text")
                # polished_prompt = message.get("content")
                polished_prompt = result.get("text")
                if polished_prompt and isinstance(polished_prompt, str):
                    final_prompt = polished_prompt.strip().replace("\n", " ")
                    # final_prompt = polished_prompt + magic_prompt
                    logger.info(f"Text Prompt enhancement successful. New prompt: '{final_prompt[:100]}...'")
                    return final_prompt
                else:
                    logger.warning("Prompt Polisher (Text) returned empty content.")
                    return None
            else:
                logger.error(f"Prompt Polisher (Text) API returned unexpected response format: {result}")
                return None

    except httpx.HTTPStatusError as e:
        logger.error(f"Prompt Polisher (Text) API request failed with status {e.response.status_code}: {e.response.text}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Error connecting to Prompt Polisher API at {endpoint}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during text prompt enhancement call: {e}", exc_info=True)
        return None


# --- Prompt Polisher EDIT Client (Multimodal Edit Polishing via HTTPX) ---

async def call_edit_prompt_polisher(
    initial_prompt: str,
    base64_images: List[str],
    timeout: int = 120,
) -> Optional[str]:
    """
    Calls the vLLM OpenAI-compatible API via HTTPX to rewrite an edit instruction,
    sending images and text.
    """
    endpoint = f"{polisher_api_base_url}/edit"
    logger.info(f"Sending request to Prompt Polisher (Edit): {endpoint} with prompt '{initial_prompt[:50]}...' and {len(base64_images)} image(s).")

    # Construct multimodal message content
    content: List[Dict[str, Any]] = []
    for img_b64 in base64_images:
        content.append({
            "type": "image",
            "image": f"data:image/jpeg;base64,{img_b64}"
        })
    content.append({"type": "text", "text": initial_prompt})

    prompt_polisher_client_config = config["prompt_polisher_client"]

    messages = [
        {"role": "system", "content": edit_system_prompt},
        {"role": "user", "content": content}
    ]

    payload = {
        "messages": messages,
        "seed": prompt_polisher_client_config["seed"],
        "top_p": prompt_polisher_client_config["top_p"],
        "temperature": prompt_polisher_client_config["temperature"],
        "max_tokens": prompt_polisher_client_config["max_tokens"],
    }

    try:
        async with httpx.AsyncClient() as client_http:
            response = await client_http.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()

            # Navigate the OpenAI API response structure
            if result.get("text") and len(result["text"]) > 0:
                # message = result.get("text")
                # enhanced_prompt_raw = message.get("content")
                enhanced_prompt_raw = result.get("text")
                
                # --- Mimic JSON parsing attempt ---
                if enhanced_prompt_raw and isinstance(enhanced_prompt_raw, str):
                    polished_text = None
                    try:
                        cleaned_result = enhanced_prompt_raw.strip().replace('```json','').replace('```','')
                        result_json = json.loads(cleaned_result)
                        if isinstance(result_json, dict) and 'Rewritten' in result_json:
                            polished_text = result_json['Rewritten']
                        else:
                            logger.warning("Polisher response parsed as JSON but missing 'Rewritten' key. Using raw response.")
                            polished_text = enhanced_prompt_raw
                    except json.JSONDecodeError:
                        logger.debug("Polisher response is not JSON, assuming direct rewritten prompt.")
                        polished_text = enhanced_prompt_raw

                    if polished_text:
                        polished_text = polished_text.strip().replace("\n", " ")
                        logger.info(f"Edit Prompt enhancement successful. New prompt: '{polished_text[:100]}...'")
                        return polished_text
                    else:
                        logger.warning("Prompt Polisher (Edit) returned an empty or invalid response after processing.")
                        return None
                else:
                    logger.warning("Prompt Polisher (Edit) returned empty content.")
                    return None
                # --- End JSON parsing ---
            else:
                logger.error(f"Prompt Polisher (Edit) API returned unexpected response format: {result}")
                return None

    except httpx.HTTPStatusError as e:
        logger.error(f"Prompt Polisher (Edit) API request failed with status {e.response.status_code}: {e.response.text}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Error connecting to Prompt Polisher API at {endpoint}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during edit prompt enhancement call: {e}", exc_info=True)
        return None


# --- Example Usage (Optional, for testing this file directly) ---
async def _test_clients():
    # Placeholder: Replace with actual base64 image data for testing
    test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" # Example: 1x1 black pixel PNG

    print("\n--- Testing Image Gen Client ---")
    gen_result = await call_image_gen("make it blue", [test_image_b64])
    if gen_result:
        print(f"Image Gen returned base64 string starting with: {gen_result[:60]}...")
    else:
        print("Image Gen call failed.")

    print("\n--- Testing Edit Prompt Polisher Client (via HTTPX) ---")
    edit_enhance_result = await call_edit_prompt_polisher("add hat", [test_image_b64])
    if edit_enhance_result: print(f"Edit Polisher returned: {edit_enhance_result}")
    else: print("Edit Prompt Polisher call failed.")

    print("\n--- Testing Text Prompt Polisher Client (via HTTPX) ---")
    text_enhance_result = await call_text_prompt_polisher("a cat sitting")
    if text_enhance_result: print(f"Text Polisher returned: {text_enhance_result}")
    else: print("Text Prompt Polisher call failed.")

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print("Running API client tests...")
    # Make sure servers are running before uncommenting:
    # asyncio.run(_test_clients())
    print("Testing setup complete (tests commented out). Remember to start servers first.")
