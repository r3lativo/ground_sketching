# src/api_clients.py

import httpx
import openai
import logging
import yaml
import os
import json # For parsing JSON in original polish_edit logic
from typing import List, Optional, Dict, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.utils import load_config

logger = logging.getLogger(__name__)

# --- Configuration & Template Loading ---
config = load_config('config/server_config.yaml')
image_gen_config = config.get("image_gen_service")
enhancer_config = config.get("prompt_enhancer_service")

IMAGE_GEN_URL = "http://localhost:8000" # Default
ENHANCER_URL = "http://localhost:8001/v1"  # Default
ENHANCER_MODEL_ID = "default_model_id" # Default
POLISH_SYSTEM_PROMPT = "You are a helpful assistant." # Default
EDIT_SYSTEM_PROMPT = "You are a helpful assistant for editing." # Default

try:
    IMAGE_GEN_URL = f"http://{image_gen_config['host']}:{image_gen_config['port']}"
    ENHANCER_URL = f"http://{enhancer_config['host']}:{enhancer_config['port']}/v1"
    ENHANCER_MODEL_ID = enhancer_config['model_id']
    logger.info(f"Image Gen API URL: {IMAGE_GEN_URL}")
    logger.info(f"Prompt Enhancer API URL: {ENHANCER_URL}")
    logger.info(f"Using Enhancer Model ID: {ENHANCER_MODEL_ID}")

    # Load Jinja Templates for System Prompts
    env = Environment(
        loader=FileSystemLoader('config'), # Look for templates in 'config/'
        autoescape=select_autoescape(['html', 'xml']) # Basic autoescaping
    )
    polish_template = env.get_template('polish.jinja2')
    edit_template = env.get_template('polish_edit.jinja2')
    # Render static templates immediately (they don't have variables here)
    POLISH_SYSTEM_PROMPT = polish_template.render().strip()
    EDIT_SYSTEM_PROMPT = edit_template.render().strip()
    logger.info("System prompts loaded from Jinja2 templates.")

except FileNotFoundError:
    logger.error(f"Configuration or template file not or config/. Using defaults.")
except KeyError as e:
    logger.error(f"Missing key in configuration file: {e}. Using defaults.")
except Exception as e:
    logger.error(f"Error loading config or templates: {e}", exc_info=True)


# --- OpenAI Client (for vLLM) ---
# Initialize OpenAI client pointing to the local vLLM server
client = openai.AsyncClient(base_url=ENHANCER_URL, api_key="dummy")


# --- Image Generation Client ---

async def call_image_gen(
    prompt: str,
    base64_images: List[str],
    timeout: int = 120
) -> Optional[str]:
    """Calls the image generation service API."""
    endpoint = f"{IMAGE_GEN_URL}/img_generate"
    payload = {
        "prompt": prompt,
        "images": base64_images,
        "seed": image_gen_config.get("seed", 0),
        "true_cfg_scale": image_gen_config.get("true_cfg_scale", 1.0),
        "negative_prompt": image_gen_config.get("negative_prompt", " "),
        "num_inference_steps": image_gen_config.get("num_inference_steps", 8),
    }
    logger.info(f"Sending request to Image Gen API: {endpoint} with prompt '{prompt[:50]}...'")
    try:
        async with httpx.AsyncClient() as client_http: # Renamed to avoid clash with openai client
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


# --- Prompt Enhancer Client (Multimodal Edit Polishing) ---

async def call_edit_prompt_enhancer(
    initial_prompt: str,
    base64_images: List[str],
    max_tokens: int = 300,
    temperature: float = 0.2
) -> Optional[str]:
    """
    Calls the vLLM prompt enhancer API to rewrite an *edit* instruction,
    mimicking the original polish_edit_prompt logic.
    """
    logger.info(f"Sending request to Prompt Enhancer (Edit): {ENHANCER_URL} with prompt '{initial_prompt[:50]}...'")

    # Construct multimodal message content
    content: List[Dict[str, Any]] = []
    for img_b64 in base64_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })
    # Add the final instruction text AFTER the images
    content.append({"type": "text", "text": initial_prompt})

    messages = [
        {"role": "system", "content": EDIT_SYSTEM_PROMPT}, # Loaded from polish_edit.jinja2
        {"role": "user", "content": content}
    ]

    try:
        response = await client.chat.completions.create(
            model=ENHANCER_MODEL_ID,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        enhanced_prompt = response.choices[0].message.content

        # --- Mimic original polish_edit_prompt JSON parsing ---
        # The original code expected the VL model to return JSON.
        # Let's try to parse, but fall back if it's just text.
        if enhanced_prompt:
            polished_text = None
            try:
                # Attempt to remove markdown and parse as JSON
                cleaned_result = enhanced_prompt.strip().replace('```json','').replace('```','')
                result_json = json.loads(cleaned_result)
                if isinstance(result_json, dict) and 'Rewritten' in result_json:
                    polished_text = result_json['Rewritten']
                else:
                     logger.warning("Enhancer response parsed as JSON but missing 'Rewritten' key. Using raw response.")
                     polished_text = enhanced_prompt # Fallback
            except json.JSONDecodeError:
                # If it's not JSON, assume the model followed the "output ONLY the rewritten instruction" rule
                logger.debug("Enhancer response is not JSON, assuming direct rewritten prompt.")
                polished_text = enhanced_prompt # Fallback

            if polished_text:
                polished_text = polished_text.strip().replace("\n", " ")
                logger.info(f"Edit Prompt enhancement successful. New prompt: '{polished_text[:100]}...'")
                return polished_text
            else:
                 logger.warning("Prompt Enhancer (Edit) returned an empty or invalid response after processing.")
                 return None
        else:
             logger.warning("Prompt Enhancer (Edit) returned an empty response.")
             return None

    except openai.APIConnectionError as e:
        logger.error(f"Failed to connect to Prompt Enhancer API at {ENHANCER_URL}: {e}")
        return None
    except openai.APIStatusError as e:
        logger.error(f"Prompt Enhancer API returned an error status {e.status_code}: {e.response}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during edit prompt enhancement call: {e}", exc_info=True)
        return None


# --- Prompt Enhancer Client (Text-Only Beautification) ---

async def call_text_prompt_enhancer(
    initial_prompt: str,
    magic_prompt: str = ", Ultra HD, 4K, cinematic composition", # Default magic prompt
    max_tokens: int = 250,
    temperature: float = 0.7 # Higher temperature for more creative beautification
) -> Optional[str]:
    """
    Calls the vLLM prompt enhancer API to rewrite a *text-only* prompt,
    mimicking the original polish_prompt logic.
    """
    logger.info(f"Sending request to Prompt Enhancer (Text): {ENHANCER_URL} with prompt '{initial_prompt[:50]}...'")

    # Construct the final prompt including the user input for the LLM
    # Note: We append the user input within the user message here.
    user_content_text = f"{initial_prompt}"

    messages = [
        {"role": "system", "content": POLISH_SYSTEM_PROMPT}, # Loaded from polish.jinja2
        {"role": "user", "content": user_content_text} # Just text input
    ]

    try:
        response = await client.chat.completions.create(
            model=ENHANCER_MODEL_ID, # Using the same VL model, but treating it as text-only
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        polished_prompt = response.choices[0].message.content
        if polished_prompt:
             polished_prompt = polished_prompt.strip().replace("\n", " ")
             # Append the magic prompt like in the original code
             final_prompt = polished_prompt + magic_prompt
             logger.info(f"Text Prompt enhancement successful. New prompt: '{final_prompt[:100]}...'")
             return final_prompt
        else:
             logger.warning("Prompt Enhancer (Text) returned an empty response.")
             return None

    except openai.APIConnectionError as e:
        logger.error(f"Failed to connect to Prompt Enhancer API at {ENHANCER_URL}: {e}")
        return None
    except openai.APIStatusError as e:
        logger.error(f"Prompt Enhancer API returned an error status {e.status_code}: {e.response}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during text prompt enhancement call: {e}", exc_info=True)
        return None
