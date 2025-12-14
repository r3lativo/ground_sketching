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
    Executes the REAL strategy construction logic to catch bugs in strategies.py,
    but skips the actual HTTP network call.
    """

    async def __aenter__(self):
        logger.info("[MOCK] Starting Mock API Session...")
        self.client = "MockActive"
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.info("[MOCK] Closing Mock API Session.")
        self.client = None

    async def _fetch_meta_info(self, utterance, context, previous_prompts, timeout: Optional[float] = None) -> Optional[Dict]:
        """
        Route it through generic helper logic.
        """
        strategy = self.strategies['meta_extraction']
        return self._generate_fake_response(strategy, utterance)

    async def execute_vlm_strategy(
        self, 
        strategy, 
        utterance: Optional[str] = "",
        context: Optional[List[str]] = None,
        previous_prompts: Optional[List[str]] = None,
        images: Optional[List[str]] = None, 
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        
        # 1. Run the actual build_payload to catch argument errors
        client_config = self.config.get("vlm_client", {})
        if hasattr(strategy, 'default_params'):
            client_config.update(strategy.default_params)
            
        try:
            # We discard the result, we just want to ensure it builds without error
            _ = strategy.build_payload(
                utterance=utterance, 
                config=client_config, 
                context=context, 
                previous_prompts=previous_prompts, 
                images=images, 
                **kwargs
            )
        except Exception as e:
            logger.error(f"[MOCK] CRITICAL: Strategy {strategy.__class__.__name__} failed to build payload! Error: {e}")
            raise e

        # 2. SIMULATE LATENCY
        await asyncio.sleep(0.01)

        # 3. RETURN FAKE DATA (Based on strategy type)
        return self._generate_fake_response(strategy, utterance)

    # --- We still mock image gen because it has no strategy logic ---
    async def call_image_gen(self, prompt: str, base64_images: Optional[List[str]], seed: Optional[int] = None, timeout: Optional[float] = None) -> Optional[str]:
        await asyncio.sleep(0.1)
        logger.info(f"[MOCK] Generating image for: {prompt}... (Seed: {seed})")
        return mock_gen_logic(prompt)

    # --- IMPROVEMENT ---
    # We do NOT override call_triplets_extraction, call_visual_verifier, etc.
    # We let the parent class (APIClient) run those methods.
    # The parent class calls 'execute_vlm_strategy', so we ONLY override that.

    def _generate_fake_response(self, strategy, utterance):
        """Internal helper to route fake responses."""
        
        strat_name = strategy.__class__.__name__
        logger.info(f"[MOCK] simulating {strat_name}...")

        if strat_name == 'MetaStrategy':
            # Weights: 40% NEW, 30% CONTINUE, 30% SKIP
            valid_actions = [Action.NEW, Action.CONTINUE, Action.SKIP]
            selected_action = random.choices(valid_actions, weights=[40, 30, 30], k=1)[0]
            
            # Construct a valid JSON string that the parser expects
            fake_json = json.dumps({
                "frame_meta": "Mock Frame Meta",
                "relation": "Mock Relation",
                "imagery": f"Mock imagery for {utterance[:10]}",
                "action": selected_action
            })
            return strategy.process_response(f"<think>...</think>```json{fake_json}```")

        elif strat_name == 'SummarizeStrategy':
            return ["Fact A: The object is red.", "Fact B: The sky is blue."]
            
        elif strat_name == 'FactCheckStrategy':
            return [{"fact": "Fact A", "verdict": True, "box": [0,0,10,10]}]
            
        elif strat_name == 'TripletsExtractionStrategy':
            return [{"subject": "A", "predicate": "touches", "object": "B"}]
            
        elif strat_name == 'CaptionStrategy':
            return "A mock caption of the image."
            
        else:
            # Default text creation
            return mock_creation_logic(utterance or "Generic Prompt")

    async def call_triplets_extraction(self, utterance: str, context: dict, timeout: Optional[float] = None) -> List[Tuple[str]]:
        """Tries to extract triplets from a relation and the given context."""
        print(self.strategies['triplets_extraction'].build_user_content(utterance, context))
        return await self.execute_vlm_strategy(
            strategy=self.strategies['triplets_extraction'],
            utterance=utterance, context=context, timeout=timeout
        )
