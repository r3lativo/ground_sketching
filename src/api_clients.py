# src/api_clients.py

import httpx
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image

from src.utils import load_config, pil_to_base64
from src.strategies import (
    PromptStrategy, 
    TextCreateStrategy, 
    TextEditStrategy, 
    MultimodalEditStrategy,
    MetaStrategy,
    # New Strategies
    SummarizeStrategy,
    CaptionStrategy,
    FactCheckStrategy
)

logger = logging.getLogger(__name__)

# --- Constants & Data Structures ---

class Action:
    NEW = '[NEW]'
    CONTINUE = '[CONTINUE]'
    SKIP = '[SKIP]'

@dataclass
class StrategyDecision:
    """Holds the result of the meta-extraction and strategy selection process."""
    action: Optional[str] = None
    strategy: Optional[PromptStrategy] = None
    strategy_name: Optional[str] = None
    meta_info: Optional[Dict[str, Any]] = None
    imagery_utterance: Optional[str] = None

    def to_dict(self):
        """
        Converts the decision to a dictionary for logging.
        """
        return {
            "action": self.action,
            "strategy_name": self.strategy_name,
            "meta_info": self.meta_info,
            "imagery_utterance": self.imagery_utterance
        }

# --- Main Client ---

class APIClient:
    """
    Manages connections to the VLM and Image Generation services.
    """

    def __init__(self, server_config_path: str, experiment_config_path: str):
        self.config = {}
        self.strategies: Dict[str, PromptStrategy] = {}
        self.image_gen_url = ""
        self.polisher_api_base_url = ""
        self.client: Optional[httpx.AsyncClient] = None
        
        self._load_configurations(server_config_path, experiment_config_path)
        self._init_jinja_and_strategies()

    async def __aenter__(self):
        # Connection pooling limits to prevent opening too many file descriptors
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
        self.client = httpx.AsyncClient(limits=limits, timeout=60.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
            logger.info("API Client session closed.")

    # --- Initialization Helpers ---

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
            jinja_conf = self.config['experiment']['jinja']
            env = Environment(
                loader=FileSystemLoader(jinja_conf['env']),
                autoescape=select_autoescape(['html', 'xml'])
            )

            def load_strat(cls, template_key):
                # Ensure the key exists in config, otherwise it might raise KeyError
                return cls(env.get_template(jinja_conf[template_key]).render())

            # Initialize Strategies
            self.strategies = {
                # Existing
                'meta_extraction': load_strat(MetaStrategy, 'meta_extraction'),
                'oracle_create_context': load_strat(TextCreateStrategy, 'initial_start_o'),
                'oracle_edit_context': load_strat(TextEditStrategy, 'initial_edit_o'),
                'oracle_simple': load_strat(TextCreateStrategy, 'final_start_t'),
                'real_create_context': load_strat(TextCreateStrategy, 'initial_start_r2'),
                'real_edit_context': load_strat(TextEditStrategy, 'initial_edit_r'),
                'real_simple': load_strat(TextCreateStrategy, 'initial_start_r1'),
                'multimodal_edit': load_strat(MultimodalEditStrategy, 'final_edit_t'),
                
                # New Strategies
                'summarize': load_strat(SummarizeStrategy, 'summarize_t'),
                'caption': load_strat(CaptionStrategy, 'caption_t'),
                'fact_check': load_strat(FactCheckStrategy, 'fact_t'),
            }
            
        except Exception as e:
            logger.error(f"Error initializing Strategies: {e}", exc_info=True)
            raise

    # --- Core Logic ---

    async def get_meta_and_strategy(
        self,
        is_oracle: bool,
        utterance: Optional[str] = None,
        context: Optional[List[str]] = None,
        previous_prompts: Optional[List[str]] = None,
        has_images: Optional[bool] = False,
        timeout: int = 300
    ) -> StrategyDecision:
        
        if not self.client:
            raise RuntimeError("Client not initialized. Use 'async with APIClient(...)'.")

        # 1. Immediate Short-circuit: Multimodal
        if has_images:
            return StrategyDecision(
                action='multimodal_edit', 
                strategy=self.strategies['multimodal_edit'], 
                strategy_name='multimodal_edit'
            )

        # 2. Fetch Meta Information (Network Call)
        response_data = await self._fetch_meta_info(utterance, context, previous_prompts, timeout)
        
        if not response_data:
            return StrategyDecision() # Returns empty/None values safely

        action = response_data.get('action')
        meta_info = response_data.get('meta')
        imagery_utterance = response_data.get('imagery_utterance')

        # 3. Determine Strategy Name dynamically
        strategy_name = None
        prefix = "oracle" if is_oracle else "real"
        
        if action == Action.NEW:
            # If context exists, we "create context", otherwise "simple"
            suffix = "create_context" if context else "simple"
            strategy_name = f"{prefix}_{suffix}"

        elif action == Action.CONTINUE:
            strategy_name = f"{prefix}_edit_context"

        # 4. Resolve Strategy Object
        selected_strategy = self.strategies.get(strategy_name) if strategy_name else None

        return StrategyDecision(
            action=action,
            strategy=selected_strategy,
            strategy_name=strategy_name,
            meta_info=meta_info,
            imagery_utterance=imagery_utterance
        )

    async def _fetch_meta_info(self, utterance, context, previous_prompts, timeout) -> Optional[Dict]:
        """Helper to handle the specific Meta Extraction API call."""
        strategy = self.strategies['meta_extraction']
        client_config = self.config.get("vlm_client", {})
        endpoint = f"{self.polisher_api_base_url}{strategy.endpoint_suffix}"

        payload = strategy.build_payload(
            utterance=utterance, config=client_config, context=context,
            previous_prompts=previous_prompts, images=None
        )

        try:
            response = await self.client.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            return strategy.process_response(response.json().get("text"))
        except Exception as e:
            logger.error(f"Meta Extraction Request failed: {e}", exc_info=True)
            return None

    async def execute_vlm_strategy(
        self, strategy: PromptStrategy, utterance: str, 
        context: Optional[List[str]] = None, 
        previous_prompts: Optional[List[str]] = None, 
        images: Optional[List[str]] = None, timeout: int = 300
    ) -> Optional[str]:
        
        if not self.client: raise RuntimeError("Client not initialized.")
        
        client_config = self.config.get("vlm_client", {})
        endpoint = f"{self.polisher_api_base_url}{strategy.endpoint_suffix}"
        payload = strategy.build_payload(utterance, client_config, context, previous_prompts, images)

        try:
            response = await self.client.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            return strategy.process_response(response.json().get("text"))
        except Exception as e:
            logger.error(f"VLM Strategy Execution failed ({strategy.__class__.__name__}): {e}", exc_info=True)
            return None

    async def call_image_gen(self, prompt: str, base64_images: Optional[List[str]], timeout: int = 300) -> Optional[str]:
        if not self.client: raise RuntimeError("Client not initialized.")

        endpoint = f"{self.image_gen_url}/img_generate"
        cfg = self.config.get('image_gen_client', {})
        
        if not base64_images:
            dummy_image = Image.new('RGB', (1024, 1024), color='white')
            base64_images = [pil_to_base64(dummy_image)]

        payload = {
            "prompt": prompt,
            "images": base64_images,
            "seed": cfg.get("seed"),
            "true_cfg_scale": cfg.get("true_cfg_scale"),
            "negative_prompt": cfg.get("negative_prompt"),
            "num_inference_steps": cfg.get("num_inference_steps"),
        }
        
        try:
            response = await self.client.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            return result.get("image")
        except Exception as e:
            logger.error(f"Image Gen failed: {e}", exc_info=True)
            return None

    # --- NEW CALLS ---

    async def call_prompt_summarizer(self, context: List[str], timeout: int = 300) -> Optional[str]:
        """
        Summarizes a list of prompts.
        It should tell us what important objects are there by the end of the prompt.
        Uses 'summarize' strategy (TextCreateStrategy-like).
        The list of prompts is passed as 'context'.
        """
        logger.info("Calling Prompt Summarizer...")
        return await self.execute_vlm_strategy(
            strategy=self.strategies['summarize'],
            utterance="", # Utterance not used in this specific strategy's build_user_content
            context=context,
            timeout=timeout
        )

    async def call_image_captioner(self, base64_images: List[str], timeout: int = 300) -> Optional[str]:
        """
        Captions the provided images.
        Uses 'caption' strategy (MultimodalEditStrategy-like).
        """
        logger.info("Calling Image Captioner...")
        return await self.execute_vlm_strategy(
            strategy=self.strategies['caption'],
            utterance="Now describe this image in detail.", # Passed as text input to MM strategy
            images=base64_images,
            timeout=timeout
        )

    async def call_fact_checker(self, fact: str, caption: str, timeout: int = 360) -> str:
        """
        Checks if the fact is contained in the caption.
        Uses 'fact_check' strategy.
        Passes Fact as 'utterance' and Caption inside 'context' list.
        """
        logger.info("Calling Fact Checker...")
        result = await self.execute_vlm_strategy(
            strategy=self.strategies['fact_check'],
            utterance=fact,
            context=[caption], # Strategy expects context list
            timeout=timeout
        )
        # Fallback if result is None
        return result if result else "False"
