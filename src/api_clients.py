# src/api_clients.py

import httpx
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Union
from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log
)

from src.utils import load_config, pil_to_base64
from src.strategies import (
    PromptStrategy, 
    TextCreateStrategy, 
    TextEditStrategy, 
    MultimodalEditStrategy,
    MetaStrategy,
    SummarizeStrategy,
    CaptionStrategy,
    FactCheckStrategy,
    TripletsExtractionStrategy
)

logger = logging.getLogger(__name__)

# --- 1. CLEAN RETRY SETUP ---

def _should_retry_error(exception: BaseException) -> bool:
    """
    Predicate to decide if we should retry.
    - Retry: Network timeouts, Connection errors, 5xx Server errors.
    - Fail Fast: 4xx Client errors (Bad Request, Unauthorized), generic logic errors.
    """
    # 1. Network / Transport Errors (Retry these)
    if isinstance(exception, (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError)):
        return True
    
    # 2. HTTP Status Errors (Retry 5xx, Fail 4xx)
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code >= 500
        
    return False

def get_retry_config(min_wait=2, max_wait=20, attempts=3):
    """
    Returns a configured decorator.
    Usage: @get_retry_config(min_wait=5)
    """
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception(_should_retry_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True  # Important: If retries fail, raise the actual exception
    )

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
    frame_meta: Optional[str] = None
    relation: Optional[str] = None
    imagery: Optional[str] = None

    def to_dict(self):
        return {
            "action": self.action,
            "strategy_name": self.strategy_name,
            "frame_meta": self.frame_meta,
            "relation": self.relation,
            "imagery": self.imagery
        }

# --- Main Client ---

class APIClient:
    """Manages connections to the services."""

    def __init__(self, server_config_path: str, experiment_config_path: str):
        self.config = {}
        self.strategies: Dict[str, PromptStrategy] = {}
        self.image_gen_url = ""
        self.vllm_api_url = ""
        self.client: Optional[httpx.AsyncClient] = None
        
        self._load_configurations(server_config_path, experiment_config_path)
        self._init_jinja_and_strategies()

    async def __aenter__(self):
        # Connection pooling limits to prevent opening too many file descriptors
        limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
        self.client = httpx.AsyncClient(limits=limits, timeout=None) 
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
            logger.info("[API CLIENTS] API Client session closed.")

    # --- Initialization Helpers ---

    def _load_configurations(self, server_conf: str, exp_conf: str):
        try:
            self.config = load_config(config_path=server_conf)
            self.config.update(load_config(config_path=exp_conf))
            self.image_gen_url = f"http://{self.config['image_gen_service']['host']}:{self.config['image_gen_service']['port']}"
            self.vllm_api_url = f"http://{self.config['vlm_service']['gateway']['host']}:{self.config['vlm_service']['gateway']['port']}"
            logger.info(f"[API CLIENTS] Initialized. Img: {self.image_gen_url}, vLLM: {self.vllm_api_url}")
        except Exception as e:
            logger.error(f"[API CLIENTS] Config load failed: {e}", exc_info=True)
            raise

    def _init_jinja_and_strategies(self):
        try:
            jinja_conf = self.config['experiment']['jinja']
            env = Environment(
                loader=FileSystemLoader(jinja_conf['env']),
                autoescape=select_autoescape(['html', 'xml'])
            )

            def load_strat(cls, template_key):
                return cls(env.get_template(jinja_conf[template_key]).render())

            self.strategies = {
                'meta_extraction': load_strat(MetaStrategy, 'meta_extraction'),
                'oracle_create_context': load_strat(TextCreateStrategy, 'initial_start_o'),
                'oracle_edit_context': load_strat(TextEditStrategy, 'initial_edit_o'),
                'oracle_simple': load_strat(TextCreateStrategy, 'final_start_t'),
                'real_create_context': load_strat(TextCreateStrategy, 'initial_start_r2'),
                'real_edit_context': load_strat(TextEditStrategy, 'initial_edit_r'),
                'real_simple': load_strat(TextCreateStrategy, 'initial_start_r1'),
                'multimodal_edit': load_strat(MultimodalEditStrategy, 'final_edit_t'),
                'summarize': load_strat(SummarizeStrategy, 'summarize_t'),
                'caption': load_strat(CaptionStrategy, 'caption_t'),
                'fact_check': load_strat(FactCheckStrategy, 'fact_t'),
                'triplets_extraction': load_strat(TripletsExtractionStrategy, 'triplets_extraction_t'),
            }
        except Exception as e:
            logger.error(f"[API CLIENTS] Strategies Init Failed: {e}", exc_info=True)
            raise

    # --- Core Logic ---

    async def get_meta_and_strategy(
        self,
        is_oracle: bool,
        utterance: Optional[str] = None,
        context: Optional[List[str]] = None,
        previous_prompts: Optional[List[str]] = None,
        has_images: Optional[bool] = False,
        timeout: Optional[float] = None
    ) -> StrategyDecision:
        
        if not self.client: raise RuntimeError("Client not initialized.")

        # 1. Immediate Short-circuit: Multimodal
        if has_images:
            return StrategyDecision(
                action='multimodal_edit', 
                strategy=self.strategies['multimodal_edit'], 
                strategy_name='multimodal_edit'
            )
        # 2. Fetch Meta Information (Network Call)
        try:
            response_data = await self._fetch_meta_info(utterance, context, previous_prompts, timeout)
        except Exception as e:
            logger.error(f"[API] Meta Extraction failed after retries: {e}")
            return StrategyDecision()

        if not response_data:
            return StrategyDecision()

        action = response_data.get('action')

        # 3. Determine Strategy Name dynamically
        strategy_name = None
        prefix = "oracle" if is_oracle else "real"
        
        if action == Action.NEW:
            suffix = "create_context" if context else "simple"
            strategy_name = f"{prefix}_{suffix}"
        elif action == Action.CONTINUE:
            strategy_name = f"{prefix}_edit_context"

        return StrategyDecision(
            action=action,
            strategy=self.strategies.get(strategy_name),
            strategy_name=strategy_name,
            frame_meta=response_data.get('frame_meta'),
            relation=response_data.get('relation'),
            imagery=response_data.get('imagery')
        )

    @get_retry_config(min_wait=2, max_wait=10)
    async def _fetch_meta_info(self, utterance, context, previous_prompts, timeout: Optional[float] = None) -> Optional[Dict]:
        """Helper to handle the specific Meta Extraction API call."""
        strategy = self.strategies['meta_extraction']
        client_config = self.config.get("vlm_client", {})
        endpoint = f"{self.vllm_api_url}{strategy.endpoint_suffix}"

        payload = strategy.build_payload(utterance=utterance, config=client_config, context=context, previous_prompts=previous_prompts)
        
        response = await self.client.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status() # Raises Error -> Caught by @retry -> Retried if 500
        return strategy.process_response(response.json().get("text"))

    @get_retry_config(min_wait=5, max_wait=30)
    async def execute_vlm_strategy(
        self, strategy: PromptStrategy, utterance: Optional[str] = "",
        context: Optional[List[str]] = None, previous_prompts: Optional[List[str]] = None,
        images: Optional[List[str]] = None, timeout: Optional[float] = None, **kwargs
    ) -> Optional[Any]:
        
        if not self.client: raise RuntimeError("Client not initialized.")
        
        client_config = self.config.get("vlm_client", {})
        if hasattr(strategy, 'default_params'):
            client_config.update(strategy.default_params)

        endpoint = f"{self.vllm_api_url}{strategy.endpoint_suffix}"
        payload = strategy.build_payload(utterance=utterance, config=client_config, context=context, previous_prompts=previous_prompts, images=images, **kwargs)

        response = await self.client.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        
        resp_json = response.json()
        raw_text = resp_json.get("text", resp_json.get("content", ""))
        return strategy.process_response(raw_text)

    @get_retry_config(min_wait=5, max_wait=60)
    async def call_image_gen(
        self, prompt: str, base64_images: Optional[List[str]],
        seed: Optional[int] = None, timeout: Optional[float] = None
    ) -> Optional[str]:
        if not self.client: raise RuntimeError("Client not initialized.")

        endpoint = f"{self.image_gen_url}/img_generate"
        cfg = self.config.get('image_gen_client', {})
        current_seed = seed if seed is not None else cfg.get("seed")

        if not base64_images:
            dummy_image = Image.new('RGB', (1024, 1024), color='white')
            base64_images = [pil_to_base64(dummy_image)]

        payload = {
            "prompt": prompt, "images": base64_images, "seed": current_seed,
            "true_cfg_scale": cfg.get("true_cfg_scale"), "negative_prompt": cfg.get("negative_prompt"),
            "num_inference_steps": cfg.get("num_inference_steps"),
        }

        logger.info(f"[API] Generating Img: {prompt}... (Seed: {seed})")
        response = await self.client.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json().get("image")

    # --- VISUAL VERIFICATION & FAITHFULNESS ---

    @get_retry_config(min_wait=5, max_wait=30)
    async def call_prompt_summarizer(self, context: List[str], timeout: Optional[float] = None) -> List[str]:
        result = await self.execute_vlm_strategy(
            strategy=self.strategies['summarize'],
            utterance="", context=context, timeout=timeout
        )
        return result if isinstance(result, list) else []

    @get_retry_config(min_wait=5, max_wait=30)
    async def call_visual_verifier(self, facts: List[str], base64_image: str, timeout: Optional[float] = None) -> List[Dict]:
        """
        Verifies a list of facts against an image.
        Returns a detailed list of dicts: {'fact': str, 'box': [y,x,y,x], 'verdict': bool}
        """
        result = await self.execute_vlm_strategy(
            strategy=self.strategies['fact_check'],
            utterance=str(facts), images=[base64_image], timeout=timeout
        )
        return result if isinstance(result, list) else []

    @get_retry_config(min_wait=5, max_wait=30)
    async def call_image_captioner(self, base64_images: List[str], timeout: Optional[float] = None) -> Optional[str]:
        """Captions the provided images."""
        return await self.execute_vlm_strategy(
            strategy=self.strategies['caption'],
            utterance="Now describe this image in detail.", images=base64_images, timeout=timeout
        )

    # No decorator here because it handles logic, not direct IO
    async def verify_image_faithfulness(
        self, base64_image: str, facts: Optional[List[str]] = None,
        context: Optional[List[str]] = None, timeout: Optional[float] = None
    ) -> Tuple[float, List[Dict]]:
        """
        Manages verification and calculates the score.
        
        Args:
            base64_image: The image to verify.
            facts: (Optional) Pre-computed list of atomic facts.
            context: (Optional) If facts aren't provided, use this prompt history to generate them.
            
        Returns:
            score: Float (0.0 to 1.0) representing % of confirmed facts.
            details: List of verification dicts (useful for debugging/logging).
        """
        if not facts:
            if not context: raise ValueError("Requires facts or context.")
            facts = await self.call_prompt_summarizer(context, timeout=timeout)
        
        if not facts: return 0.0, []

        verification_results = await self.call_visual_verifier(facts, base64_image, timeout=timeout)
        if not verification_results: return 0.0, []

        true_count = sum(1 for item in verification_results if item.get("verdict") is True)
        return (true_count / len(facts)), verification_results

    @get_retry_config(min_wait=5, max_wait=30)
    async def call_triplets_extraction(self, utterance: str, context: dict, timeout: Optional[float] = None) -> List[Tuple[str]]:
        """Tries to extract triplets from a relation and the given context."""
        return await self.execute_vlm_strategy(
            strategy=self.strategies['triplets_extraction'],
            utterance=utterance, context=context, timeout=timeout
        )
