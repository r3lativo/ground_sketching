# src/api_clients.py

import httpx
import logging
from typing import List, Optional, Dict, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image

# Removed global setup_logging call to prevent side effects on import
from src.utils import load_config, pil_to_base64
from src.strategies import (
    PromptStrategy, 
    TextCreateStrategy, 
    TextEditStrategy, 
    MultimodalEditStrategy,
    MetaStrategy
)

try:
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except ImportError:
    pass

logger = logging.getLogger(__name__)

class APIClient:
    """
    Manages connections to the VLM and Image Generation services.
    Encapsulates configuration, Jinja2 environment, and prompt strategies.
    """

    def __init__(self, server_config_path: str, experiment_config_path: str):
        self.config = {}
        self.strategies: Dict[str, PromptStrategy] = {}
        self.image_gen_url = ""
        self.polisher_api_base_url = ""
        
        # Load Configuration
        self._load_configurations(server_config_path, experiment_config_path)
        
        # Initialize Components
        self._init_jinja_and_strategies()

    def _load_configurations(self, server_conf: str, exp_conf: str):
        try:
            self.config = load_config(config_path=server_conf)
            self.config.update(load_config(config_path=exp_conf))

            # Setup URLs
            self.image_gen_url = f"http://{self.config['image_gen_service']['host']}:{self.config['image_gen_service']['port']}"
            self.polisher_api_base_url = f"http://{self.config['vlm_service']['gateway']['host']}:{self.config['vlm_service']['gateway']['port']}"
            
            logger.info(f"API Client Initialized.")
            logger.info(f"Image Gen URL: {self.image_gen_url}")
            logger.info(f"VLM Base URL: {self.polisher_api_base_url}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}", exc_info=True)
            raise

    def _init_jinja_and_strategies(self):
        try:
            env = Environment(
                loader=FileSystemLoader(self.config['experiment']['jinja']['env']),
                autoescape=select_autoescape(['html', 'xml'])
            )

            # Initialize Strategies with Templates
            self.strategies['meta_extraction'] = MetaStrategy(
                env.get_template(self.config['experiment']['jinja']['new_or_continue']).render()
            )
            self.strategies['oracle_create_context'] = TextCreateStrategy(
                env.get_template(self.config['experiment']['jinja']['initial_start_o']).render()
            )
            self.strategies['oracle_edit_context'] = TextEditStrategy(
                env.get_template(self.config['experiment']['jinja']['initial_edit_o']).render()
            )
            self.strategies['oracle_simple'] = TextCreateStrategy(
                env.get_template(self.config['experiment']['jinja']['final_start_t']).render()
            )
            self.strategies['real_create_context'] = TextCreateStrategy(
                env.get_template(self.config['experiment']['jinja']['initial_start_r2']).render()
            )
            self.strategies['real_edit_context'] = TextEditStrategy(
                env.get_template(self.config['experiment']['jinja']['initial_edit_r']).render()
            )
            self.strategies['real_simple'] = TextCreateStrategy(
                env.get_template(self.config['experiment']['jinja']['initial_start_r1']).render()
            )
            self.strategies['multimodal_edit'] = MultimodalEditStrategy(
                env.get_template(self.config['experiment']['jinja']['final_edit_t']).render()
            )
            
        except Exception as e:
            logger.error(f"Error initializing Jinja2 or Strategies: {e}", exc_info=True)
            raise

    # --- Strategy Factory Method ---

    async def get_meta_and_strategy(
        self,
        is_oracle: bool,
        utterance: Optional[str] = None,
        context: Optional[List[str]] = None,
        previous_prompts: Optional[List[str]] = None,
        has_images: Optional[bool] = False,
        timeout: int = 300
    ) -> (PromptStrategy, str, Optional[str], Optional[str]):
        """
        Determines the appropriate strategy and extracts meta-information.
        """
        # If image exists, force MM edit
        if has_images:
            return self.strategies['multimodal_edit'], 'multimodal_edit', None, None

        choice = None
        strategy = self.strategies['meta_extraction']
        client_config = self.config.get("vlm_client", {})
        endpoint = f"{self.polisher_api_base_url}{strategy.endpoint_suffix}"
        
        # Build Payload
        payload = strategy.build_payload(
            utterance=utterance,
            config=client_config,
            context=context,
            previous_prompts=previous_prompts,
            images=None
        )

        # Call Meta Extraction
        meta_info = None
        imagery_utterance = None

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(endpoint, json=payload, timeout=timeout)
                response.raise_for_status()
                result = response.json()
                
                response_dict = strategy.process_response(result.get("text"))
                if response_dict:
                    choice = response_dict.get('action')
                    meta_info = response_dict.get('meta') if response_dict.get('meta') else None
                    imagery_utterance = response_dict.get('imagery_utterance') if response_dict.get('imagery_utterance') else None

        except Exception as e:
            logger.error(f"Meta Extraction Request failed: {e}", exc_info=True)
            return None, None, None, None
        
        # Select Strategy based on Choice
        if is_oracle:
            if choice == '[NEW]':
                target = 'oracle_create_context' if context else 'oracle_simple'
                return self.strategies[target], target, meta_info, imagery_utterance
            elif choice == '[CONTINUE]':
                return self.strategies['oracle_edit_context'], 'oracle_edit_context', meta_info, imagery_utterance
        else:
            if choice == '[NEW]':
                target = 'real_create_context' if context else 'real_simple'
                return self.strategies[target], target, meta_info, imagery_utterance
            elif choice == '[CONTINUE]':
                return self.strategies['real_edit_context'], 'real_edit_context', meta_info, imagery_utterance

        logger.warning(f"Unknown choice or state: {choice}")
        return None, None, None, None

    # --- Execution Methods ---

    async def execute_vlm_strategy(
        self,
        strategy: PromptStrategy,
        utterance: str,
        context: Optional[List[str]] = None,
        previous_prompts: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        timeout: int = 300
    ) -> Optional[str]:
        
        client_config = self.config.get("vlm_client", {})
        endpoint = f"{self.polisher_api_base_url}{strategy.endpoint_suffix}"
        
        payload = strategy.build_payload(
            utterance=utterance,
            config=client_config,
            context=context,
            previous_prompts=previous_prompts,
            images=images
        )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(endpoint, json=payload, timeout=timeout)
                response.raise_for_status()
                result = response.json()
                return strategy.process_response(result.get("text"))
        except Exception as e:
            logger.error(f"VLM Strategy Execution failed: {e}", exc_info=True)
            return None

    async def call_image_gen(
        self, 
        prompt: str, 
        base64_images: Optional[List[str]], 
        timeout: int = 300
    ) -> Optional[str]:
        
        endpoint = f"{self.image_gen_url}/img_generate"
        image_gen_config = self.config.get('image_gen_client', {})
        
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
            # Create a white dummy image if none provided but expected by API
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
