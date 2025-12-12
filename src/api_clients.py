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
    FactCheckStrategy,
    TripletsExtractionStrategy
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
    frame_meta: Optional[str] = None
    relation: Optional[str] = None
    imagery: Optional[str] = None

    def to_dict(self):
        """
        Converts the decision to a dictionary for logging.
        """
        return {
            "action": self.action,
            "strategy_name": self.strategy_name,
            "frame_meta": self.frame_meta,
            "relation": self.relation,
            "imagery": self.imagery
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

            # Setup URLs
            self.image_gen_url = f"http://{self.config['image_gen_service']['host']}:{self.config['image_gen_service']['port']}"
            self.polisher_api_base_url = f"http://{self.config['vlm_service']['gateway']['host']}:{self.config['vlm_service']['gateway']['port']}"
            
            logger.info(f"[API CLIENTS] API Client Initialized.")
            logger.info(f"[API CLIENTS] Image Gen URL: {self.image_gen_url}")
            logger.info(f"[API CLIENTS] VLM Base URL: {self.polisher_api_base_url}")
            
        except Exception as e:
            logger.error(f"[API CLIENTS] Failed to load configuration: {e}", exc_info=True)
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

                'triplets_extraction': load_strat(TripletsExtractionStrategy, 'triplets_extraction_t'),
            }
            
        except Exception as e:
            logger.error(f"[API CLIENTS] Error initializing Strategies: {e}", exc_info=True)
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
        frame_meta = response_data.get('frame_meta')
        relation = response_data.get('relation')
        imagery = response_data.get('imagery')

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
            frame_meta=frame_meta,
            relation=relation,
            imagery=imagery
        )

    async def _fetch_meta_info(self, utterance, context, previous_prompts, timeout: Optional[float] = None) -> Optional[Dict]:
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
            logger.error(f"[API CLIENTS] Meta Extraction Request failed: {e}", exc_info=True)
            return None

    async def execute_vlm_strategy(
        self, 
        strategy: PromptStrategy, 
        utterance: Optional[str] = "",
        context: Optional[List[str]] = None,
        previous_prompts: Optional[List[str]] = None,
        images: Optional[List[str]] = None, 
        timeout: Optional[float] = None,
        **kwargs
    ) -> Optional[Any]:
        
        if not self.client: raise RuntimeError("Client not initialized.")
        
        client_config = self.config.get("vlm_client", {})
        
        if hasattr(strategy, 'default_params'):
            client_config.update(strategy.default_params)

        endpoint = f"{self.polisher_api_base_url}{strategy.endpoint_suffix}"
        
        payload = strategy.build_payload(
            utterance=utterance, 
            config=client_config, 
            context=context, 
            previous_prompts=previous_prompts, 
            images=images, 
            **kwargs 
        )

        try:
            response = await self.client.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            
            # Helper to safely get text
            resp_json = response.json()
            # Some VLMs return 'content', others 'text'.
            raw_text = resp_json.get("text", resp_json.get("content", ""))
            
            return strategy.process_response(raw_text)
        except Exception as e:
            logger.error(f"[API CLIENTS] VLM Strategy Execution failed ({strategy.__class__.__name__}): {e}", exc_info=True)
            return None

    async def call_image_gen(self, prompt: str, base64_images: Optional[List[str]], seed: Optional[int] = None, timeout: Optional[float] = None) -> Optional[str]:
        if not self.client: raise RuntimeError("Client not initialized.")

        endpoint = f"{self.image_gen_url}/img_generate"
        cfg = self.config.get('image_gen_client', {})
        
        # Use passed seed if valid, otherwise fallback to config
        current_seed = seed if seed is not None else cfg.get("seed")

        if not base64_images:
            dummy_image = Image.new('RGB', (1024, 1024), color='white')
            base64_images = [pil_to_base64(dummy_image)]

        payload = {
            "prompt": prompt,
            "images": base64_images,
            "seed": current_seed,
            "true_cfg_scale": cfg.get("true_cfg_scale"),
            "negative_prompt": cfg.get("negative_prompt"),
            "num_inference_steps": cfg.get("num_inference_steps"),
        }

        logger.info(f"[API CLIENTS] Generating image for: {prompt[:20]}... (Seed: {seed})")
        
        try:
            response = await self.client.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            return result.get("image")
        except Exception as e:
            logger.error(f"[API CLIENTS] Image Gen failed: {e}", exc_info=True)
            return None

    # --- VISUAL VERIFICATION & FAITHFULNESS ---

    async def call_prompt_summarizer(self, context: List[str], timeout: Optional[float] = None) -> List[str]:
        """
        Decomposes the prompt history into a list of atomic visual facts.
        """
        logger.info("[API CLIENTS] Calling Prompt Summarizer (Fact Decomposition)...")
        # Uses 'summarize' strategy which now returns List[str] via json_parser
        result = await self.execute_vlm_strategy(
            strategy=self.strategies['summarize'],
            utterance="", 
            context=context,
            timeout=timeout
        )
        # Ensure we return a list, even if strategy fails
        return result if isinstance(result, list) else []

    async def call_visual_verifier(self, facts: List[str], base64_image: str, timeout: Optional[float] = None) -> List[Dict]:
        """
        Verifies a list of facts against an image.
        Returns a detailed list of dicts: {'fact': str, 'box': [y,x,y,x], 'verdict': bool}
        """
        logger.info("[API CLIENTS] Calling Visual Verifier (Fact Check)...")
        
        # Convert list to string for the prompt
        facts_str = str(facts)
        
        # Uses 'fact_check' strategy which returns List[Dict] via json_parser
        result = await self.execute_vlm_strategy(
            strategy=self.strategies['fact_check'],
            utterance=facts_str, 
            images=[base64_image],
            timeout=timeout
        )
        return result if isinstance(result, list) else []

    async def call_image_captioner(self, base64_images: List[str], timeout: Optional[float] = None) -> Optional[str]:
        """
        Captions the provided images.
        Uses 'caption' strategy (MultimodalEditStrategy-like).
        """
        logger.info("[API CLIENTS] Calling Image Captioner...")
        return await self.execute_vlm_strategy(
            strategy=self.strategies['caption'],
            utterance="Now describe this image in detail.", # Passed as text input to MM strategy
            images=base64_images,
            timeout=timeout
        )

    async def verify_image_faithfulness(
        self, 
        base64_image: str, 
        facts: Optional[List[str]] = None, 
        context: Optional[List[str]] = None,
        timeout: Optional[float] = None
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
        # 1. Decompose
        # If facts are provided (e.g. calculated once per user), use them. 
        # Otherwise, derive them from context.
        if not facts:
            if not context:
                raise ValueError("[API CLIENTS] verify_image_faithfulness requires either 'facts' or 'context'.")
            facts = await self.call_prompt_summarizer(context, timeout=timeout)
        
        if not facts:
            logger.warning("[API CLIENTS] No facts available for verification. Returning 0.0.")
            return 0.0, []

        # 2. Verify
        verification_results = await self.call_visual_verifier(facts, base64_image, timeout=timeout)
        
        if not verification_results:
             return 0.0, []

        # 3. Score
        # We calculate the score as: (True Verdicts / Total Facts)
        true_count = 0
        for item in verification_results:
            # We check for explicit True verdict. 
            # (Optional: You could also check if 'box' is not [0,0,0,0] for extra strictness)
            if item.get("verdict") is True:
                # if str(item.get("box")) != "[0,0,0,0]"
                true_count += 1
        
        final_score = true_count / len(facts) if facts else 0.0
        
        return final_score, verification_results

    async def call_triplets_extraction(self, relation: str, context: dict, timeout: Optional[float] = None) -> List[Tuple[str]]:
        """
        Tries to extract triplets from a relation and the given context.
        """
        logger.info("[API CLIENTS] Calling Triplets Extraction...")
        result = await self.execute_vlm_strategy(
            strategy=self.strategies['triplets_extraction'],
            utterance="",
            relation=relation,
            context=context,
            timeout=timeout
        )
        return result
