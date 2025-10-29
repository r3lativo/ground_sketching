# src/api_clients.py

import httpx
import logging
import yaml
import os
import json
from typing import List, Optional, Dict, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape
import torch # Added for seed generation fallback in call_image_gen

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = logging.getLogger(__name__)

# --- Configuration & Template Loading ---
CONFIG = {}
CONFIG_PATH = 'config/server_config.yaml'
IMAGE_GEN_URL = "http://localhost:8000" # Default
ENHANCER_API_BASE_URL = "http://localhost:8001/" # Default
ENHANCER_MODEL_ID = "default_model_id" # Default
POLISH_SYSTEM_PROMPT = "You are a helpful assistant." # Default
EDIT_SYSTEM_PROMPT = "You are a helpful assistant for editing." # Default

try:
    # Load Server Config
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = yaml.safe_load(f)
    IMAGE_GEN_URL = f"http://{CONFIG['image_gen_service']['host']}:{CONFIG['image_gen_service']['port']}"
    # Construct the base URL for the OpenAI compatible endpoint
    ENHANCER_API_BASE_URL = f"http://{CONFIG['prompt_enhancer_service']['host']}:{CONFIG['prompt_enhancer_service']['port']}"
    ENHANCER_MODEL_ID = CONFIG['prompt_enhancer_service']['model_id']

    logger.info(f"Image Gen API URL: {IMAGE_GEN_URL}")
    logger.info(f"Prompt Enhancer API Base URL: {ENHANCER_API_BASE_URL}")
    logger.info(f"Using Enhancer Model ID: {ENHANCER_MODEL_ID}")

    # Load Jinja Templates for System Prompts
    env = Environment(
        loader=FileSystemLoader('config'),
        autoescape=select_autoescape(['html', 'xml'])
    )
    polish_template = env.get_template('polish.jinja2')
    edit_template = env.get_template('polish_edit.jinja2')
    POLISH_SYSTEM_PROMPT = polish_template.render().strip()
    EDIT_SYSTEM_PROMPT = edit_template.render().strip()
    logger.info("System prompts loaded from Jinja2 templates.")

except FileNotFoundError:
    logger.error(f"Configuration or template file not found in {CONFIG_PATH} or config/. Using defaults.")
except KeyError as e:
    logger.error(f"Missing key in configuration file {CONFIG_PATH}: {e}. Using defaults.")
except Exception as e:
    logger.error(f"Error loading config or templates: {e}", exc_info=True)


# --- Image Generation Client ---

async def call_image_gen(
    prompt: str,
    base64_images: List[str],
    timeout: int = 120
) -> Optional[str]:
    """Calls the image generation service API."""
    endpoint = f"{IMAGE_GEN_URL}/img_generate"
    image_gen_config = CONFIG.get('image_gen_client', {})
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


# --- Prompt Enhancer Client (Text-Only Beautification via HTTPX) ---

async def call_text_prompt_enhancer(
    initial_prompt: str,
    #magic_prompt: str,
    #max_tokens: int,
    #temperature: float,
    timeout: int = 120
) -> Optional[str]:
    """
    Calls the vLLM OpenAI-compatible API via HTTPX to rewrite a text-only prompt.
    """
    endpoint = f"{ENHANCER_API_BASE_URL}/generate"
    logger.info(f"Sending request to Prompt Enhancer (Text): {endpoint} with prompt '{initial_prompt[:50]}...'")
    prompt_enhancer_client_config = CONFIG["prompt_enhancer_client"]
    messages = [
        {"role": "system", "content": POLISH_SYSTEM_PROMPT},
        {"role": "user", "content": initial_prompt}
    ]

    payload = {
        "messages": messages,
        "seed": prompt_enhancer_client_config["seed"],
        "top_p": prompt_enhancer_client_config["top_p"],
        "temperature": prompt_enhancer_client_config["temperature"],
        "max_tokens": prompt_enhancer_client_config["max_tokens"],
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
                    logger.warning("Prompt Enhancer (Text) returned empty content.")
                    return None
            else:
                logger.error(f"Prompt Enhancer (Text) API returned unexpected response format: {result}")
                return None

    except httpx.HTTPStatusError as e:
        logger.error(f"Prompt Enhancer (Text) API request failed with status {e.response.status_code}: {e.response.text}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Error connecting to Prompt Enhancer API at {endpoint}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during text prompt enhancement call: {e}", exc_info=True)
        return None


# --- Prompt Enhancer Client (Multimodal Edit Polishing via HTTPX) ---

async def call_edit_prompt_enhancer(
    initial_prompt: str,
    base64_images: List[str],
    timeout: int = 120,
) -> Optional[str]:
    """
    Calls the vLLM OpenAI-compatible API via HTTPX to rewrite an edit instruction,
    sending images and text.
    """
    endpoint = f"{ENHANCER_API_BASE_URL}/edit"
    logger.info(f"Sending request to Prompt Enhancer (Edit): {endpoint} with prompt '{initial_prompt[:50]}...' and {len(base64_images)} image(s).")

    # Construct multimodal message content
    content: List[Dict[str, Any]] = []
    for img_b64 in base64_images:
        content.append({
            "type": "image",
            "image": f"data:image/jpeg;base64,{img_b64}"
        })
    content.append({"type": "text", "text": initial_prompt})

    prompt_enhancer_client_config = CONFIG["prompt_enhancer_client"]

    messages = [
        {"role": "system", "content": EDIT_SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ]

    payload = {
        "messages": messages,
        "seed": prompt_enhancer_client_config["seed"],
        "top_p": prompt_enhancer_client_config["top_p"],
        "temperature": prompt_enhancer_client_config["temperature"],
        "max_tokens": prompt_enhancer_client_config["max_tokens"],
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
                            logger.warning("Enhancer response parsed as JSON but missing 'Rewritten' key. Using raw response.")
                            polished_text = enhanced_prompt_raw
                    except json.JSONDecodeError:
                        logger.debug("Enhancer response is not JSON, assuming direct rewritten prompt.")
                        polished_text = enhanced_prompt_raw

                    if polished_text:
                        polished_text = polished_text.strip().replace("\n", " ")
                        logger.info(f"Edit Prompt enhancement successful. New prompt: '{polished_text[:100]}...'")
                        return polished_text
                    else:
                        logger.warning("Prompt Enhancer (Edit) returned an empty or invalid response after processing.")
                        return None
                else:
                    logger.warning("Prompt Enhancer (Edit) returned empty content.")
                    return None
                # --- End JSON parsing ---
            else:
                logger.error(f"Prompt Enhancer (Edit) API returned unexpected response format: {result}")
                return None

    except httpx.HTTPStatusError as e:
        logger.error(f"Prompt Enhancer (Edit) API request failed with status {e.response.status_code}: {e.response.text}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Error connecting to Prompt Enhancer API at {endpoint}: {e}")
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

    print("\n--- Testing Edit Prompt Enhancer Client (via HTTPX) ---")
    edit_enhance_result = await call_edit_prompt_enhancer("add hat", [test_image_b64])
    if edit_enhance_result: print(f"Edit Enhancer returned: {edit_enhance_result}")
    else: print("Edit Prompt Enhancer call failed.")

    print("\n--- Testing Text Prompt Enhancer Client (via HTTPX) ---")
    text_enhance_result = await call_text_prompt_enhancer("a cat sitting")
    if text_enhance_result: print(f"Text Enhancer returned: {text_enhance_result}")
    else: print("Text Prompt Enhancer call failed.")

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print("Running API client tests...")
    # Make sure servers are running before uncommenting:
    # asyncio.run(_test_clients())
    print("Testing setup complete (tests commented out). Remember to start servers first.")

