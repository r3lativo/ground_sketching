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

# --- Configuration Loading ---
try:
    # Load Server Config
    config = load_config(config_path='config/server_config.yaml')

    # Construct the base URL for the OpenAI compatible endpoint
    image_gen_url = f"http://{config['image_gen_service']['host']}:{config['image_gen_service']['port']}"
    polisher_api_base_url = f"http://{config['vlm_service']['gateway']['host']}:{config['vlm_service']['gateway']['port']}"
    polisher_model_id = config['vlm_service']['backend']['model_id']

    logger.info(f"Image Gen API URL: {image_gen_url}")
    logger.info(f"Prompt Polisher API Base URL: {polisher_api_base_url}")
    logger.info(f"Using Polisher Model ID: {polisher_model_id}")

    # Load Jinja Templates for System Prompts
    env = Environment(
        loader=FileSystemLoader(config['vlm_client']['jinja']['env']),
        autoescape=select_autoescape(['html', 'xml'])
    )
    polish_template = env.get_template(config['vlm_client']['jinja']['polish_t'])
    edit_template = env.get_template(config['vlm_client']['jinja']['edit_t'])
    contextual_polish_template = env.get_template(config['vlm_client']['jinja']['contextual_polish_t'])
    contextual_edit_template = env.get_template(config['vlm_client']['jinja']['contextual_edit_t'])
    
    # Render templates
    polish_system_prompt = polish_template.render()
    edit_system_prompt = edit_template.render()
    contextual_polish_system_prompt = contextual_polish_template.render()
    contextual_edit_system_prompt = contextual_edit_template.render()
    logger.info("System prompts loaded from Jinja2 templates.")

except FileNotFoundError:
    logger.error(f"Configuration or template file not found. Using defaults.")
except KeyError as e:
    logger.error(f"Missing key in configuration file: {e}. Using defaults.")
except Exception as e:
    logger.error(f"Error loading config or templates: {e}", exc_info=True)


# --- Image Generation Client ---

async def call_image_gen(prompt: str, base64_images: Optional[List[str]], timeout: int = 300) -> Optional[str]:
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


# --- Consolidated Prompt Polisher Client ---

async def call_prompt_polisher(
    utterance: str,
    images: Optional[List[str]] = None,
    context: Optional[List[str]] = None,
    previous_prompts: Optional[List[str]] = None,
    timeout: int = 300
) -> Optional[str]:
    """
    Consolidated function to call the vLLM prompt polisher.
    
    It intelligently selects the correct endpoint, system prompt,
    and payload structure based on the arguments provided.
    
    - utterance (str): The user's latest text instruction.
    - images (List[str], optional): A list of Base64 *without* data:image header.
    - context (List[str], optional): The conversational context.
    - previous_prompts (List[str], optional): A history of generated prompts.
    """
    
    messages = []
    endpoint_path = "/generate"  # Default to text-only endpoint
    post_process_fn = lambda raw_text: raw_text.strip() # Default post-processing
    
    # --- 1. Determine API Configuration based on inputs ---

    if images:
        # --- Case 1: Multimodal Edit (Needs /edit endpoint) ---
        logger.info(f"Polisher Case: Multimodal Edit (Utterance: '{utterance[:50]}...')")
        endpoint_path = "/edit"
        
        content: List[Dict[str, Any]] = []
        for img_b64 in images:
            # Add the data:image header required by the gateway
            content.append({"type": "image", "image": f"data:image/jpeg;base64,{img_b64}"})
        content.append({"type": "text", "text": utterance})
        
        messages = [
            {"role": "system", "content": edit_system_prompt},
            {"role": "user", "content": content}
        ]
        
        # Define the post-processing function for this case
        def post_process_fn(raw_text):
            parsed = thinking_parser(raw_text)
            logger.info(f"Polisher (Edit) thought: {parsed.get('thinking', 'N/A')[:150]}...")
            print(f"Thinking:\n{parsed.get('thinking')}\nAnswer:{parsed.get('answer')}")
            answer = parsed.get("answer")
            if answer:
                return json_parser(answer, 'Rewritten') # Expects JSON
            return None

    elif context and previous_prompts is not None:
        # --- Case 2: Contextual Edit (Text-only, uses /generate) ---
        logger.info(f"Polisher Case: Contextual Edit (Utterance: '{utterance[:50]}...')")
        content_str = f"Conversation Context:\n{context}\nTarget Utterance:\n{utterance}\nPrevious Prompts:\n{previous_prompts}\nRewritten:\n"
        messages = [
            {"role": "system", "content": contextual_edit_system_prompt},
            {"role": "user", "content": content_str}
        ]
        
        def post_process_fn(raw_text):
            parsed = json_parser(raw_text, 'Rewritten') # Expects JSON
            if parsed:
                return parsed.strip().replace("\n", " ")
            return None

    elif context:
        # --- Case 3: Contextual Create (Text-only, uses /generate) ---
        logger.info(f"Polisher Case: Contextual Create (Utterance: '{utterance[:50]}...')")
        content_str = f"Conversation Context:\n{context}\nTarget Utterance:\n{utterance}\nRewritten Prompt:\n"
        messages = [
            {"role": "system", "content": contextual_polish_system_prompt},
            {"role": "user", "content": content_str}
        ]
        
        def post_process_fn(raw_text):
            return raw_text.strip().replace("\n", " ") # Expects plain text

    else:
        # --- Case 4: Simple Text-Only Polish (uses /generate) ---
        logger.info(f"Polisher Case: Simple Text-Only (Utterance: '{utterance[:50]}...')")
        messages = [
            {"role": "system", "content": polish_system_prompt},
            {"role": "user", "content": utterance}
        ]
        
        def post_process_fn(raw_text):
            parsed = thinking_parser(raw_text)
            logger.info(f"Polisher (Text) thought: {parsed.get('thinking', 'N/A')[:150]}...")
            print(f"Thinking:\n{parsed.get('thinking')}\nAnswer:{parsed.get('answer')}")
            return parsed.get("answer") # Expects plain text


    # --- 2. Build the Final Payload ---
    endpoint = f"{polisher_api_base_url}{endpoint_path}"
    prompt_polisher_client_config = config["vlm_client"]
    payload = {
        "messages": messages,
        "seed": prompt_polisher_client_config.get("seed"),
        "top_p": prompt_polisher_client_config.get("top_p"),
        "temperature": prompt_polisher_client_config.get("temperature"),
        "max_tokens": prompt_polisher_client_config.get("max_tokens"),
    }

    # --- 3. Make the API Call (Centralized Logic) ---
    logger.info(f"Sending request to Polisher API: {endpoint}")
    try:
        async with httpx.AsyncClient() as client_http:
            response = await client_http.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            
            raw_text = result.get("text")
            if not raw_text or len(raw_text) == 0:
                logger.error(f"Polisher API at {endpoint} returned empty or invalid text field: {result}")
                return None
            
            # --- 4. Post-process the response ---
            final_prompt = post_process_fn(raw_text)
            
            if final_prompt:
                logger.info(f"Polisher successful. New prompt: '{final_prompt[:100]}...'")
                return final_prompt
            else:
                logger.warning(f"Polisher at {endpoint} returned empty content after parsing raw text: '{raw_text[:100]}...'")
                return None

    except httpx.HTTPStatusError as e:
        logger.error(f"Polisher API at {endpoint} failed with status {e.response.status_code}: {e.response.text}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Error connecting to Polisher API at {endpoint}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during polisher call: {e}", exc_info=True)
        return None

# --- OLD FUNCTIONS REMOVED ---
# call_text_prompt_polisher, call_edit_prompt_polisher,
# call_contextual_prompt_polisher, and call_contextual_edit_prompt_polisher
# are now all replaced by the single 'call_prompt_polisher' function above.
