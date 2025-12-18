# src/mock_api_clients.py

import asyncio
import logging
import random
import json
from typing import List, Optional, Dict, Any, Tuple

from src.api_clients import APIClient, Action 
from src.utils import mock_creation_logic, mock_gen_logic

logger = logging.getLogger(__name__)

class MockAPIClient(APIClient):
    """
    Enhanced Mock Client.
    Executes the REAL strategy construction and parsing logic,
    but replaces the HTTP network call with a simulated raw response.
    """

    async def __aenter__(self):
        logger.info("[MOCK] Starting Mock API Session...")
        self.client = "MockActive"
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.info("[MOCK] Closing Mock API Session.")
        self.client = None

    async def _fetch_meta_info(self, utterance, context, previous_prompts, timeout: Optional[float] = None) -> Optional[Dict]:
        """Route it through the generic executor."""
        strategy = self.strategies['meta_extraction']
        return await self.execute_vlm_strategy(strategy, utterance, context, previous_prompts, timeout=timeout)

    async def execute_vlm_strategy(
        self, 
        strategy, 
        utterance: Optional[str] = "",
        context: Optional[List[str]] = None,
        previous_prompts: Optional[List[str]] = None,
        images: Optional[List[str]] = None, 
        timeout: Optional[float] = None,
        validation_schema: Any = None, 
        max_correction_attempts: int = 2,
        **kwargs
    ) -> Any:
        
        # 1. TEST PAYLOAD CONSTRUCTION
        # This catches bugs in 'build_user_content' before they hit production
        client_config = self.config.get("vlm_client", {})
        if hasattr(strategy, 'default_params'):
            client_config.update(strategy.default_params)
            
        try:
            payload = strategy.build_payload(
                utterance=utterance, 
                config=client_config, 
                context=context, 
                previous_prompts=previous_prompts, 
                images=images, 
                **kwargs
            )
            print(payload)
        except Exception as e:
            logger.error(f"[MOCK] CRITICAL: Strategy {strategy.__class__.__name__} failed to build payload! Error: {e}")
            raise e

        # 2. SIMULATE LATENCY
        await asyncio.sleep(0.01)

        # 3. GENERATE RAW RESPONSE (Simulating VLM Output Text)
        raw_text = self._generate_fake_raw_text(strategy, utterance)

        # 4. EXECUTE PARSING & VALIDATION CHAIN
        # We run the actual parser to ensure the mock data is compatible with the strategy logic
        try:
            parsed_data = strategy.process_response(raw_text)
            
            # Use the parent class's validation logic
            return self._validate_output(parsed_data, validation_schema)
            
        except Exception as e:
            logger.error(f"[MOCK] Validation/Parsing failed on mock data: {e}")
            logger.error(f"[MOCK] Raw Mock Text: {raw_text}")
            raise e

    async def call_image_gen(
        self,
        prompt: str,
        base64_images: Optional[List[str]],
        seed: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> Optional[str]:
        await asyncio.sleep(0.1)
        logger.info(f"[MOCK] Generating image for: {prompt}... (Seed: {seed})")
        return mock_gen_logic(prompt)

    # --- INTERNAL HELPER: Generates RAW STRING responses ---

    def _generate_fake_raw_text(self, strategy, utterance):
        
        strat_name = strategy.__class__.__name__
        logger.info(f"[MOCK] Simulating VLM output for {strat_name}...")

        base_thought = "<think>\nMocking the reasoning process...\n</think>\n"

        # 1. META STRATEGY
        if strat_name == 'MetaStrategy':
            valid_actions = [Action.NEW, Action.CONTINUE, Action.SKIP]
            selected_action = random.choices(valid_actions, weights=[40, 30, 30], k=1)[0]
            data = {
                "frame_meta": "Mock Frame Meta",
                "relation": "Mock Relation",
                "imagery": f"Mock imagery for {utterance[:10]}",
                "action": selected_action
            }
            return f"{base_thought}```json\n{json.dumps(data)}\n```"

        # 2. TEXT & REFINEMENT STRATEGIES
        # Added 'MultimodalEditStrategy' here because it is used for Refinement and expects 'scene'
        elif strat_name in ['TextCreateStrategy', 'TextEditStrategy', 'MultimodalEditStrategy']:
            raw_mock_text = mock_creation_logic(utterance or "Generic Prompt")
            
            # Identify refinement specifically for clarity in logs
            if strat_name == 'MultimodalEditStrategy':
                raw_mock_text = f"Refined: {raw_mock_text}"

            # Wrap in the TextResponse schema
            data = {"scene": raw_mock_text}
            return f"{base_thought}```json\n{json.dumps(data)}\n```"

        # 3. SUMMARIZE STRATEGY
        elif strat_name == 'SummarizeStrategy':
            data = {"facts": ["Fact A", "Fact B"]}
            return f"{base_thought}```json\n{json.dumps(data)}\n```"

        # 4. FACT CHECK STRATEGY
        elif strat_name == 'FactCheckStrategy':
            data = {
                "verification": [{"fact": "Fact A", "verdict": True, "box": [0,0,10,10]}]
            }
            return f"{base_thought}```json\n{json.dumps(data)}\n```"

        # 5. TRIPLETS STRATEGY
        elif strat_name == 'TripletsExtractionStrategy':
            triplets_list = [{"subject": "A", "predicate": "B", "object": "C"}]
            data = {"triplets": triplets_list}
            return f"{base_thought}```json\n{json.dumps(data)}\n```"

        # 6. CAPTION STRATEGY
        elif strat_name == 'CaptionStrategy':
            data = {"caption": "A mock caption."}
            return f"{base_thought}```json\n{json.dumps(data)}\n```"

        else:
            return f"{base_thought}Mock response for {strat_name}"
