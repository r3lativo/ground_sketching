# src/api_clients.py

import httpx
import logging
import yaml
import os
import json
from typing import List, Optional, Dict, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape
import torch
from PIL import Image

from src.utils import setup_logging, load_config, pil_to_base64, json_parser, thinking_parser

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
    polish_template = env.get_template(config['prompt_polisher_client']['polish_t'])
    edit_template = env.get_template(config['prompt_polisher_client']['edit_t'])

    contextual_polish_template = env.get_template(config['prompt_polisher_client']['contextual_polish_t'])
    contextual_edit_template = env.get_template(config['prompt_polisher_client']['contextual_edit_t'])
    
    # Render templates
    polish_system_prompt = polish_template.render()
    edit_system_prompt = edit_template.render()
    contextual_polish_system_prompt = contextual_polish_template.render()
    contextual_edit_system_prompt = contextual_edit_template.render()
    logger.info("System prompts loaded from Jinja2 templates.")

except FileNotFoundError:
    logger.error(f"Configuration or template file not found in {config_PATH} or config/. Using defaults.")
except KeyError as e:
    logger.error(f"Missing key in configuration file {config_PATH}: {e}. Using defaults.")
except Exception as e:
    logger.error(f"Error loading config or templates: {e}", exc_info=True)


# --- Image Generation Client ---

async def call_image_gen(prompt: str, base64_images: Optional[List[str]], timeout: int = 120) -> Optional[str]:
    """Calls the image generation service API."""
    endpoint = f"{image_gen_url}/img_generate"
    image_gen_config = config.get('image_gen_client', {})
    payload = {
        "prompt": prompt,
        "seed": image_gen_config.get("seed"),
        "true_cfg_scale": image_gen_config.get("true_cfg_scale"),
        "negative_prompt": image_gen_config.get("negative_prompt"),
        "num_inference_steps": image_gen_config.get("num_inference_steps"),
    }

    # Add the image if present, else generate a dummy white image.
    if base64_images:
        payload["images"] = base64_images
    else:
        dummy_image_pil = Image.new('RGB', (1024,1024), color='white')
        payload["images"] = [pil_to_base64(dummy_image_pil)]
    
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
            raw_text = result.get("text")
            if raw_text and len(raw_text) > 0:
                
                # 1. Parse the raw text into thinking and answer
                parsed_output = thinking_parser(raw_text)
                thinking_part = parsed_output.get("thinking")
                answer_part = parsed_output.get("answer")
                
                print(f"Thinking:\n{thinking_part}\nAnswer:{answer_part}")

                # 2. Log the thinking part if it exists
                if thinking_part:
                    logger.info(f"Prompt Polisher (Text) thought: {thinking_part[:150]}...")

                # 3. Check if the answer part is valid
                if answer_part and isinstance(answer_part, str):
                    
                    # 4. Define the final_prompt based on the cleaned answer
                    final_prompt = answer_part
                    
                    logger.info(f"Text Prompt enhancement successful. New prompt: '{final_prompt[:100]}...'")
                    return final_prompt
                else:
                    logger.warning("Prompt Polisher (Text) returned empty content after parsing.")
                    return None
            else:
                logger.error(f"Prompt Polisher (Text) API returned empty or invalid text field: {result}")
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
            raw_text = result.get("text")
            if raw_text and len(raw_text) > 0:
                
                # 1. Parse the raw text into thinking and answer
                parsed_output = thinking_parser(raw_text)
                thinking_part = parsed_output.get("thinking")
                answer_part = parsed_output.get("answer")

                print(f"Thinking:\n{thinking_part}\nAnswer:{answer_part}")

                # 2. Log the thinking part if it exists
                if thinking_part:
                    logger.info(f"Prompt Polisher (Edit) thought: {thinking_part[:150]}...")

                # 3. Check if the answer part is valid
                if answer_part and isinstance(answer_part, str):
                    
                    # 4. Try to parse json from the final answer
                    final_prompt = json_parser(answer_part, 'Rewritten')
                    
                    logger.info(f"Edit Prompt enhancement successful. New prompt: '{final_prompt[:100]}...'")
                    
                    return final_prompt
                else:
                    logger.warning("Prompt Polisher (Edit) returned empty content after parsing.")
                    return None
            else:
                logger.error(f"Prompt Polisher (Edit) API returned empty or invalid text field: {result}")
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


# --- Prompt Polisher Client (Contextual Scene CREATION) ---

async def call_contextual_prompt_polisher(
    context: List[str],
    utterance: str,
    timeout: int = 120
) -> Optional[str]:
    """
    Calls the vLLM API to create the first prompt in a chunk,
    using the new contextual template. (Text-in, Text-out)
    """
    endpoint = f"{polisher_api_base_url}/generate"
    logger.info(f"Sending request to Contextual Polisher (Create): {endpoint} for utterance '{utterance[:50]}...'")

    content = f"Conversation Context:\n{context}\nTarget Utterance:\n{utterance}\nRewritten Prompt:\n"
    messages = [
        {"role": "system", "content": contextual_polish_system_prompt},
        {"role": "user", "content": content}
    ]

    prompt_polisher_client_config = config["prompt_polisher_client"]

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

            if result.get("text") and len(result["text"]) > 0:
                polished_prompt = result.get("text")
                if polished_prompt and isinstance(polished_prompt, str):
                    final_prompt = polished_prompt.strip().replace("\n", " ")
                    logger.info(f"Contextual (Create) enhancement successful. New prompt: '{final_prompt[:100]}...'")
                    return final_prompt
                else:
                    logger.warning("Contextual (Create) Polisher returned empty content.")
                    return None
            else:
                logger.error(f"Contextual (Create) Polisher API returned unexpected format: {result}")
                return None

    except httpx.HTTPStatusError as e:
        logger.error(f"Contextual (Create) Polisher API request failed with status {e.response.status_code}: {e.response.text}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Error connecting to Contextual (Create) Polisher API at {endpoint}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during contextual (create) call: {e}", exc_info=True)
        return None


# --- Prompt Polisher Client (Contextual Scene EDIT) ---

async def call_contextual_edit_prompt_polisher(
    context: List[str],
    utterance: str,
    previous_prompts: List[str],
    timeout: int = 120,
) -> Optional[str]:
    """
    Calls the vLLM API to edit a prompt, using the new contextual
    edit template with a history of previous prompts. (Text-in, Text-out)
    """
    endpoint = f"{polisher_api_base_url}/generate"
    logger.info(f"Sending request to Contextual Polisher (Edit): {endpoint} for utterance '{utterance[:50]}...'")

    content = f"Conversation Context:\n{context}\nTarget Utterance:\n{utterance}\nPrevious Prompts:\n{previous_prompts}\nRewritten:\n"
    messages = [
        {"role": "system", "content": contextual_edit_system_prompt},
        {"role": "user", "content": content}
    ]
    
    prompt_polisher_client_config = config["prompt_polisher_client"]

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

            if result.get("text") and len(result["text"]) > 0:
                enhanced_prompt_raw = result.get("text")
                
                if enhanced_prompt_raw and isinstance(enhanced_prompt_raw, str):
                    polished_text = None
                    polished_text = json_parser(enhanced_prompt_raw, 'Rewritten')

                    if polished_text:
                        polished_text = polished_text.strip().replace("\n", " ")
                        logger.info(f"Contextual (Edit) enhancement successful. New prompt: '{polished_text[:100]}...'")
                        return polished_text
                    else:
                        logger.warning("Contextual (Edit) Polisher returned empty or invalid response.")
                        return None
                else:
                    logger.warning("Contextual (Edit) Polisher returned empty content.")
                    return None
            else:
                logger.error(f"Contextual (Edit) Polisher API returned unexpected format: {result}")
                return None

    except httpx.HTTPStatusError as e:
        logger.error(f"Contextual (Edit) Polisher API request failed with status {e.response.status_code}: {e.response.text}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Error connecting to Contextual (Edit) Polisher API at {endpoint}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during contextual (edit) call: {e}", exc_info=True)
        return None
