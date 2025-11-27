# src/api_clients.py

import httpx
import logging
from typing import List, Optional, Dict, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image

from src.utils import setup_logging, load_config, pil_to_base64
from src.strategies import (
    PromptStrategy, 
    TextCreateStrategy, 
    TextEditStrategy, 
    MultimodalEditStrategy
)

try:
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except ImportError:
    pass

logger = logging.getLogger(__name__)
# Assuming setup_logging is called by the main entry point, but keeping this for standalone usage validity
setup_logging(log_file='logs/api_clients.log', log_to_console=False)

# --- Configuration & Strategy Initialization ---

# Global Config
config = {}
STRATEGIES: Dict[str, PromptStrategy] = {}
image_gen_url = ""
polisher_api_base_url = ""

try:
    # 1. Load Configs
    config = load_config(config_path='config/server_config.yaml')
    config.update(load_config(config_path='config/experiment_config.yaml'))

    # 2. Setup URLs
    image_gen_url = f"http://{config['image_gen_service']['host']}:{config['image_gen_service']['port']}"
    polisher_api_base_url = f"http://{config['vlm_service']['gateway']['host']}:{config['vlm_service']['gateway']['port']}"
    
    logger.info(f"Image Gen API URL: {image_gen_url}")
    logger.info(f"Prompt Polisher API Base URL: {polisher_api_base_url}")

    # 3. Setup Jinja2
    env = Environment(
        loader=FileSystemLoader(config['experiment']['jinja']['env']),
        autoescape=select_autoescape(['html', 'xml'])
    )

    # 4. Instantiate Strategies with System Prompts
    
    # META EXTRACTION
    STRATEGIES['meta_extraction'] = TextEditStrategy(
        env.get_template(config['experiment']['jinja']['new_or_continue']).render()
    )

    # ORACLE STRATEGIES
    STRATEGIES['oracle_create_context'] = TextCreateStrategy(
        env.get_template(config['experiment']['jinja']['initial_start_o']).render()
    )
    STRATEGIES['oracle_edit_context'] = TextEditStrategy(
        env.get_template(config['experiment']['jinja']['initial_edit_o']).render()
    )
    # The 'simple' cases often use the final templates or specific ones
    STRATEGIES['oracle_simple'] = TextCreateStrategy(
        env.get_template(config['experiment']['jinja']['final_start_t']).render()
    )
    
    # REAL STRATEGIES
    STRATEGIES['real_create_context'] = TextCreateStrategy(
        env.get_template(config['experiment']['jinja']['initial_start_r2']).render()
    )
    STRATEGIES['real_edit_context'] = TextEditStrategy(
        env.get_template(config['experiment']['jinja']['initial_edit_r']).render()
    )
    STRATEGIES['real_simple'] = TextCreateStrategy(
        env.get_template(config['experiment']['jinja']['initial_start_r1']).render()
    )

    # SHARED / FINAL STRATEGIES (Multimodal is usually consistent across conditions)
    STRATEGIES['multimodal_edit'] = MultimodalEditStrategy(
        env.get_template(config['experiment']['jinja']['final_edit_t']).render()
    )
    
    logger.info("VLM Strategies initialized successfully.")

except Exception as e:
    logger.error(f"Error loading config or strategies: {e}", exc_info=True)


# --- Strategy Factory ---

async def get_strategy_vlm(
    is_oracle: bool,
    utterance: Optional[str] = None,
    context: Optional[List[str]] = None,
    previous_prompts: Optional[List[str]] = None,
    has_images: Optional[bool] = False,
    timeout: int = 300
) -> (PromptStrategy, str):
    """
    Meta Extraction + Strategy selection based on inputs.
    call vlm and send prompt to choose NEW or CONTINUE
    based on CONTEXT and PREVIOUS PROMPTS and UTTERANCE
    """

    # If image, we know it's MM edit
    if has_images:
        return STRATEGIES['multimodal_edit'], 'multimodal_edit'
    
    if not context:
        if is_oracle:
            return STRATEGIES['oracle_simple'], 'oracle_simple'
        else:
            return STRATEGIES['real_simple'], 'real_simple'

    choice = None
    strategy = STRATEGIES['meta_extraction']
    client_config = config.get("vlm_client", {})
    endpoint = f"{polisher_api_base_url}{strategy.endpoint_suffix}"
    
    # DEBUG LOG
    logger.info(f"DEBUG: Executing meta_extraction")

    # 1. Build Payload
    payload = strategy.build_payload(
        utterance=utterance,
        config=client_config,
        context=context,
        previous_prompts=previous_prompts,
        images=None
    )

    # 2. Call to META EXTRACTION
    logger.info(f"Call to META EXTRACTION via VLM ({strategy.endpoint_suffix})")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
                        
            # 3. Process Response
            choice = strategy.process_response(result.get("text"))
            # TODO deal with <meta> etc. (TO IMPLEMENT)
            logger.info(f"CHOICE: {choice}")

    except Exception as e:
        logger.error(f"VLM Request failed: {e}", exc_info=True)
    
    # Process the choice
    if is_oracle:
        if choice == '[NEW]':
            return STRATEGIES['oracle_create_context'], 'oracle_create_context'
        elif choice == '[CONTINUE]':
            return STRATEGIES['oracle_edit_context'], 'oracle_edit_context'
        else:
            print("BAD CHOICE")
            return None, None
    else:
        if choice == '[NEW]':
            return STRATEGIES['real_create_context'], 'real_create_context'
        elif choice == '[CONTINUE]':
            return STRATEGIES['real_edit_context'], 'real_edit_context'
        else:
            print("BAD CHOICE")
            return None, None


# --- VLM Client (The Strategy Executor) ---

async def execute_vlm_strategy(
    strategy: PromptStrategy,
    utterance: str,
    context: Optional[List[str]] = None,
    previous_prompts: Optional[List[str]] = None,
    images: Optional[List[str]] = None,
    timeout: int = 300
) -> Optional[str]:
    """
    Executes a specific PromptStrategy against the VLM API.
    """
    client_config = config.get("vlm_client", {})
    endpoint = f"{polisher_api_base_url}{strategy.endpoint_suffix}"
    
    # DEBUG LOG
    logger.info(f"DEBUG: Executing Strategy: {type(strategy).__name__}")

    # 1. Build Payload
    payload = strategy.build_payload(
        utterance=utterance,
        config=client_config,
        context=context,
        previous_prompts=previous_prompts,
        images=images
    )

    # 2. Network Call
    logger.info(f"Sending request to VLM ({strategy.endpoint_suffix})")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            
            raw_text = result.get("text")
            
            # 3. Process Response
            final_prompt = strategy.process_response(raw_text)
            
            if final_prompt:
                return final_prompt
            else:
                logger.warning(f"Strategy returned None after processing raw text.")
                return None

    except Exception as e:
        logger.error(f"VLM Request failed: {e}", exc_info=True)
        return None


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

    if base64_images:
        payload["images"] = base64_images
    else:
        dummy_image_pil = Image.new('RGB', (1024,1024), color='white')
        payload["images"] = [pil_to_base64(dummy_image_pil)]
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            if "image" in result and isinstance(result["image"], str):
                return result["image"]
            return None
    except Exception as e:
        logger.error(f"Image Gen failed: {e}", exc_info=True)
        return None
