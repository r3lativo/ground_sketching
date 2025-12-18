# src/api_clients.py

import httpx
import logging
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Union, Type
from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log
)
from pydantic import BaseModel, ValidationError, Field, TypeAdapter

from src.utils import load_config, pil_to_base64, thinking_parser
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

# --- PYDANTIC SCHEMAS ---

class MetaResponse(BaseModel):
    """Schema for the Meta Extraction strategy"""
    frame_meta: Optional[str] = None
    action: str  # Mandatory
    relation: Optional[str] = None
    imagery: Optional[str] = None

class VerificationFact(BaseModel):
    """Schema for a single verification item"""
    fact: str
    verdict: bool
    box: Optional[List[int]] = None

class Triplet(BaseModel):
    """Schema for Triplet Extraction"""
    subject: str
    predicate: str
    object: str

class TextResponse(BaseModel):
    scene: str

class SummarizeResponse(BaseModel):
    facts: List[str]

class FactCheckResponse(BaseModel):
    verification: List[VerificationFact]

class CaptionResponse(BaseModel):
    caption: str

class TripletResponse(BaseModel):
    triplets: List[Triplet]

# ----------------------------

@dataclass
class StrategyDecision:
    """Holds the result of the meta-extraction and strategy selection process."""
    action: Optional[str] = None
    strategy: Optional[PromptStrategy] = None
    strategy_name: Optional[str] = None
    frame_meta: Optional[str] = None
    relation: Optional[str] = None
    imagery: Optional[str] = None
    validation_schema: Optional = None

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

    def _generate_friendly_error_msg(self, e: ValidationError, schema: Any, raw_text: str) -> str:
        """
        Translates Pydantic errors into clear instructions for the LLM.
        Also reconstructs a simple schema example so the model knows what to fix.
        """
        error_report = []
        
        # 1. Analyze the Errors
        for err in e.errors():
            loc = " -> ".join([str(x) for x in err['loc']])
            msg = err['msg']
            
            # Case A: Root Level Error (Model returned raw text instead of JSON)
            if err['type'] == 'model_type' and not loc:
                error_report.append(f"CRITICAL: You returned a raw string, but a JSON Object is required.")
                
            # Case B: Missing Field (Model used wrong key)
            elif err['type'] == 'missing':
                error_report.append(f"MISSING KEY: The JSON is missing the required field '{loc}'.")
                
            # Case C: Generic Field Error
            else:
                target = f"Field '{loc}'" if loc else "Response"
                error_report.append(f"FORMAT ERROR: {target} - {msg}")

        # 2. Dynamic Schema Explanation
        # We assume 'schema' is either a TypeAdapter or a BaseModel class
        try:
            # Handle TypeAdapter vs BaseModel differences
            if isinstance(schema, TypeAdapter):
                json_schema = schema.json_schema()
            elif hasattr(schema, 'model_json_schema'):
                json_schema = schema.model_json_schema()
            else:
                json_schema = {}

            # Extract Properties to show the model what is valid
            properties = json_schema.get('properties', {})
            required = json_schema.get('required', [])
            
            schema_desc = "REQUIRED JSON STRUCTURE:\n{"
            for prop, details in properties.items():
                is_req = " (REQUIRED)" if prop in required else ""
                prop_type = details.get('type', 'any')
                schema_desc += f'\n  "{prop}": <{prop_type}>{is_req},'
            schema_desc += "\n}"
            
        except Exception:
            schema_desc = "Required Format: Valid JSON matching the requested schema."

        # 3. Construct Final Message
        final_msg = (
            f"Your response failed validation.\n"
            f"{'\n'.join(error_report)}\n\n"
            f"{schema_desc}\n\n"
            f"Output a valid answer in JSON format."
        )
        return final_msg

    # --- Sanitization & Parsing ---

    def _sanitize_for_reflection(self, raw_text: str, max_preview_length: int = 2000) -> str:
        """
        Prepares a malformed response for the reflection step.
        Handles the specific case where 'thinking' is unclosed.
        """
        if not raw_text: 
            return "[EMPTY RESPONSE]"
        
        # 1. Check for Runaway Thought (Unclosed Tag)
        # If <think> exists but </think> does not, the model timed out or rambled.
        # We must NOT treat the thought as the answer.
        if "<think>" in raw_text and "</think>" not in raw_text:
            help_text = (
                "[ERROR: Generation truncated. You started <think> but never finished. No answer found.]\n"
                "[RETRY: BE SHORTER IN THINKING. Use a maximum of 500 tokens. Make sure to output the final thinking token </think>!]\n"
                "[BE DECISIVE: Once you have decided something, DO NOT GO BACK ON IT.]"
            )
            return help_text

        # 2. Standard Parse
        parsed = thinking_parser(raw_text)
        answer = parsed.get("answer", "").strip()

        # 3. Fallback: If parser failed to find separation
        if not answer and not parsed.get("thinking"):
            answer = raw_text

        # 4. Truncate huge answers to save context
        if len(answer) > max_preview_length:
            answer = answer[:max_preview_length] + "... [TRUNCATED]"
            
        return answer if answer else "[NO ANSWER FOUND]"

    def _validate_output(self, parsed_data: Any, schema: Any) -> Any:
        """
        Unified validation helper for Pydantic V1/V2 & TypeAdapter.
        """
        if not schema:
            return parsed_data

        if isinstance(schema, TypeAdapter):
            validated = schema.validate_python(parsed_data)
            if isinstance(validated, list):
                return [v.model_dump() if hasattr(v, 'model_dump') else v for v in validated]
            return validated.model_dump() if hasattr(validated, 'model_dump') else validated

        if hasattr(schema, 'model_validate'):
            return schema.model_validate(parsed_data).model_dump()
            
        # Legacy Pydantic V1
        return schema(**parsed_data).dict()


    async def get_meta_and_strategy(
        self,
        is_oracle: bool,
        utterance: Optional[str] = None,
        context: Optional[List[str]] = None,
        previous_prompts: Optional[List[str]] = None,
        has_images: Optional[bool] = False,
        timeout: Optional[float] = 480
    ) -> StrategyDecision:
        
        if not self.client: raise RuntimeError("Client not initialized.")

        # Immediate Short-circuit: Multimodal
        if has_images:
            return StrategyDecision(
                action='multimodal_edit', 
                strategy=self.strategies['multimodal_edit'], 
                strategy_name='multimodal_edit'
            )
        # Fetch Meta Information
        meta_validator = TypeAdapter(MetaResponse) if TypeAdapter else None

        try:
            response_data = await self.execute_vlm_strategy(
                strategy=self.strategies['meta_extraction'],
                utterance=utterance,
                context=context,
                previous_prompts=previous_prompts,
                timeout=timeout,
                validation_schema=meta_validator
            )
        except Exception as e:
            logger.error(f"[API] Meta Extraction failed: {e}")
            return StrategyDecision()

        # Safety
        if not response_data:
            return StrategyDecision(action=Action.SKIP)

        action = response_data.get('action')

        # Determine Strategy Name dynamically
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
            imagery=response_data.get('imagery'),
            validation_schema=TypeAdapter(TextResponse) if TypeAdapter else None
        )

    @get_retry_config(min_wait=5, max_wait=30)
    async def execute_vlm_strategy(
        self, strategy: PromptStrategy, utterance: Optional[str] = "",
        context: Optional[List[str]] = None, previous_prompts: Optional[List[str]] = None,
        images: Optional[List[str]] = None, timeout: Optional[float] = 480, 
        validation_schema: Any = None, max_correction_attempts: int = 2,
        **kwargs
    ) -> Optional[Any]:
        
        if not self.client: raise RuntimeError("Client not initialized.")
        
        # 0. Setup Configuration
        client_config = self.config.get("vlm_client", {})
        if hasattr(strategy, 'default_params'):
            client_config.update(strategy.default_params)
        endpoint = f"{self.vllm_api_url}{strategy.endpoint_suffix}"
        
        # 1. Build Base Payload (Snapshot)
        base_payload = strategy.build_payload(
            utterance=utterance, config=client_config, context=context, 
            previous_prompts=previous_prompts, images=images, **kwargs
        )
        current_payload = base_payload.copy()
        current_payload['messages'] = [dict(m) for m in base_payload['messages']]

        for attempt in range(max_correction_attempts + 1):
            try:
                # A. Network Request
                response = await self.client.post(endpoint, json=current_payload, timeout=timeout)
                response.raise_for_status()
                
                resp_json = response.json()
                raw_text = resp_json.get("text", resp_json.get("content", ""))

                # --- Runaway Thought Check ---
                # Before parsing, check if we hit the "infinite thinking" bug.
                # If <think> is unclosed, strategies.process_response will fail or return garbage.
                if "<think>" in raw_text and "</think>" not in raw_text:
                    raise ValidationError("Runaway thought detected (Unclosed <think> tag).")

                # B. Parsing
                parsed_data = strategy.process_response(raw_text)

                # C. Validation
                return self._validate_output(parsed_data, validation_schema)

            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                # Capture technical error for logs
                try: error_json = e.json()
                except: error_json = str(e)
                
                logger.warning(f"[API] Validation Failed (Attempt {attempt+1}): {error_json}")

                if attempt >= max_correction_attempts:
                    logger.error("[API] Max retries exhausted.")
                    return None

                # --- GENERATE FRIENDLY MESSAGE ---
                # Translate the error into Natural Language + Context
                friendly_error_msg = error_json # Fallback
                
                if isinstance(e, ValidationError) and validation_schema:
                    friendly_error_msg = self._generate_friendly_error_msg(e, validation_schema, raw_text)

                # --- RETRY LOGIC ---
                
                # Strategy 1: Reflection (Gentle Correction)
                if attempt == 0:
                    sanitized_answer = self._sanitize_for_reflection(raw_text)
                    current_payload['messages'].append({
                        "role": "assistant", 
                        "content": sanitized_answer 
                    })
                    # Send the FRIENDLY message instead of the raw technical dump
                    current_payload['messages'].append({
                        "role": "user", 
                        "content": friendly_error_msg 
                    })

                # Strategy 2: Fresh Start (Hard Reset)
                else:
                    current_payload['messages'] = [dict(m) for m in base_payload['messages']]
                    current_payload['messages'].append({
                        "role": "user", 
                        "content": (
                            f"Previous attempts failed.\n"
                            f"{friendly_error_msg}\n"
                            "Ignore previous thoughts. Output ONLY valid JSON."
                        )
                    })

        return None

    @get_retry_config(min_wait=5, max_wait=60)
    async def call_image_gen(
        self, prompt: str, base64_images: Optional[List[str]],
        seed: Optional[int] = None, timeout: Optional[float] = 120
    ) -> Optional[str]:
        if not self.client: raise RuntimeError("Client not initialized.")
        endpoint = f"{self.image_gen_url}/img_generate"
        cfg = self.config.get('image_gen_client', {})
        current_seed = seed if seed is not None else cfg.get("seed")

        if not base64_images:
            dummy_image = Image.new('RGB', (1024, 1024), color='white')
            base64_images = [pil_to_base64(dummy_image)]

        clean_prompt = prompt[:2000] if len(prompt) > 2000 else prompt

        payload = {
            "prompt": clean_prompt, "images": base64_images, "seed": current_seed,
            "true_cfg_scale": cfg.get("true_cfg_scale"), "negative_prompt": cfg.get("negative_prompt"),
            "num_inference_steps": cfg.get("num_inference_steps"),
        }

        logger.info(f"[API] Generating Img: {prompt}... (Seed: {seed})")
        response = await self.client.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json().get("image")

    # --- VISUAL VERIFICATION & FAITHFULNESS ---

    @get_retry_config(min_wait=5, max_wait=30)
    async def call_prompt_summarizer(self, context: List[str], timeout: Optional[float] = 480) -> List[str]:
        validator = TypeAdapter(SummarizeResponse)
        
        result = await self.execute_vlm_strategy(
            strategy=self.strategies['summarize'],
            utterance="", context=context, timeout=timeout,
            validation_schema=validator
        )
        
        if result and isinstance(result, dict):
            return result.get('facts', [])
        return []

    @get_retry_config(min_wait=5, max_wait=30)
    async def call_visual_verifier(self, facts: List[str], base64_image: str, timeout: Optional[float] = 480) -> List[Dict]:
        validator = TypeAdapter(FactCheckResponse)
            
        result = await self.execute_vlm_strategy(
            strategy=self.strategies['fact_check'],
            utterance=str(facts), 
            images=[base64_image], 
            timeout=timeout,
            validation_schema=validator
        )
        
        if result and isinstance(result, dict):
            return result.get('verification', [])
        return []

    @get_retry_config(min_wait=5, max_wait=30)
    async def call_image_captioner(self, base64_images: List[str], timeout: Optional[float] = 480) -> Optional[str]:
        validator = TypeAdapter(CaptionResponse)
        
        result = await self.execute_vlm_strategy(
            strategy=self.strategies['caption'],
            utterance="Now describe this image in detail.",
            images=base64_images,
            timeout=timeout,
            validation_schema=validator
        )
        
        if result and isinstance(result, dict):
            return result.get('caption')
        return None

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

        # Add isinstance check to prevent crash on malformed lists
        true_count = sum(
            1 for item in verification_results 
            if isinstance(item, dict) and item.get("verdict") is True
        )
        
        return (true_count / len(facts)), verification_results

    @get_retry_config(min_wait=5, max_wait=30)
    async def call_triplets_extraction(self, utterance: str, context: dict, timeout: Optional[float] = 480) -> List[Dict]:
        validator = TypeAdapter(TripletResponse)

        result = await self.execute_vlm_strategy(
            strategy=self.strategies['triplets_extraction'],
            utterance=utterance, context=context, timeout=timeout,
            validation_schema=validator
        )
        
        if result and isinstance(result, dict):
            return result.get('triplets', [])
        return []
