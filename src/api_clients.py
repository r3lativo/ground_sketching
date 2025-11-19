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
setup_logging(log_file='logs/api_clients.log', log_to_console=False)

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

    initial_start_template = env.get_template(config['vlm_client']['jinja']['initial_start_t'])
    initial_edit_template = env.get_template(config['vlm_client']['jinja']['initial_edit_t'])

    final_start_template = env.get_template(config['vlm_client']['jinja']['final_start_t'])
    final_edit_template = env.get_template(config['vlm_client']['jinja']['final_edit_t'])
    
    # Render templates
    polish_system_prompt = polish_template.render()
    edit_system_prompt = edit_template.render()

    initial_start_system_prompt = initial_start_template.render()
    inital_edit_system_prompt = initial_edit_template.render()

    final_start_system_prompt = final_start_template.render()
    final_edit_system_prompt = final_edit_template.render()
    
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
    
    logger.info(f"Sending request to Image Gen API: {endpoint} with prompt '{prompt.replace('\n', ' ')}'")
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


# --- Universal Post-Processing Function ---

def _post_process_polisher_response(
    raw_text: str,
    json_key: Optional[str] = None
) -> Optional[str]:
    """
    Consolidated post-processing for all polisher responses.
    1. Always separates 'thinking' from 'answer'.
    2. Logs the 'thinking' part.
    3. If a 'json_key' is given, attempts to parse the 'answer' as JSON.
    4. Returns the final, clean answer.
    """
    if not raw_text:
        return None

    # 1. Always run thinking_parser first
    parsed_output = thinking_parser(raw_text)
    thinking_part = parsed_output.get("thinking")
    answer_part = parsed_output.get("answer")

    # Print the thinking/answer to the console
    print(f"\nThinking:\n{thinking_part}")
    print(f"\nAnswer:\n{answer_part}")

    # 2. Log the thinking part (with newlines replaced for clean logging)
    if thinking_part:
        logger.info(f"Polisher thought: {thinking_part.replace('\n', ' ')}")
    
    # 3. Check if we have an answer to process
    if not answer_part:
        logger.warning("Polisher returned no answer after parsing </think>.")
        return None

    # 4. Attempt to parse JSON *if* a key was provided
    if json_key:
        final_prompt = json_parser(answer_part, json_key)
        # json_parser returns the original text on failure, so we check
        # if it's *still* JSON (which means failure)
        if final_prompt.strip().startswith('{'):
            logger.warning(f"JSON key '{json_key}' not found. Returning raw answer.")
            return answer_part.strip()
        return final_prompt
    
    # 5. If no JSON key, just return the clean answer
    return answer_part.strip()


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
    json_key_to_parse = None # Default to expecting plain text
    
    # --- 1. Determine API Configuration based on inputs ---

    if images:
        # --- Case 1: Multimodal Edit (uses /edit endpoint) ---
        logger.info(f"Polisher Case: Multimodal Edit (Utterance: '{utterance.replace('\n', ' ')}')")
        endpoint_path = "/edit"
        json_key_to_parse = "Rewritten" # Expects JSON
        
        content: List[Dict[str, Any]] = []
        for img_b64 in images:
            content.append({"type": "image", "image": f"data:image/jpeg;base64,{img_b64}"})
        content.append({"type": "text", "text": utterance})
        
        messages = [
            {"role": "system", "content": final_edit_system_prompt},
            {"role": "user", "content": content}
        ]

    elif context and previous_prompts is not None:
        # --- Case 2: Initial Edit (Text-only, uses /generate) ---
        logger.info(f"Polisher Case: Contextual Edit (Utterance: '{utterance.replace('\n', ' ')}')")
        json_key_to_parse = "Rewritten" # Expects JSON
        
        content_str = f"Conversation Context:\n{context}\nTarget Utterance:\n{utterance}\nPrevious Prompts:\n{previous_prompts}\nRewritten:\n"
        messages = [
            {"role": "system", "content": inital_edit_system_prompt},
            {"role": "user", "content": content_str}
        ]

    elif context:
        # --- Case 3: Initial Start (Text-only, uses /generate) ---
        logger.info(f"Polisher Case: Contextual Create (Utterance: '{utterance.replace('\n', ' ')}')")
        # No JSON key, expects plain text
        
        content_str = f"Conversation Context:\n{context}\nTarget Utterance:\n{utterance}\nRewritten Prompt:\n"
        messages = [
            {"role": "system", "content": initial_start_system_prompt},
            {"role": "user", "content": content_str}
        ]

    else:
        # --- Case 4: Simple Text-Only Polish (uses /generate) ---
        logger.info(f"Polisher Case: Simple Text-Only (Utterance: '{utterance.replace('\n', ' ')}')")
        # No JSON key, expects plain text
        
        messages = [
            {"role": "system", "content": final_start_system_prompt},
            {"role": "user", "content": utterance}
        ]

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
            final_prompt = _post_process_polisher_response(raw_text, json_key_to_parse)
            
            if final_prompt:
                # Log the final, clean prompt
                logger.info(f"Polisher successful. New prompt: '{final_prompt.replace('\n', ' ')}'")
                return final_prompt
            else:
                logger.warning(f"Polisher at {endpoint} returned empty content after parsing raw text: '{raw_text.replace('\n', ' ')}'")
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
