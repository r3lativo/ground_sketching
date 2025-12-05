# src/mock_api_clients.py

import asyncio
import logging
import random
from typing import List, Optional, Dict, Any

# [CRITICAL] Import Action to recognize SKIP/NEW/CONTINUE constants
from src.api_clients import APIClient, Action 
from src.utils import mock_creation_logic, mock_gen_logic
from src.strategies import (
    SummarizeStrategy, 
    FactCheckStrategy, 
    CaptionStrategy
)

logger = logging.getLogger(__name__)

class MockAPIClient(APIClient):
    """
    A drop-in replacement for APIClient that uses your local mock logic.
    """

    async def __aenter__(self):
        logger.info("[MOCK] Starting Mock API Session...")
        self.client = "MockActive"
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.info("[MOCK] Closing Mock API Session.")
        self.client = None

    async def _fetch_meta_info(self, utterance, context, previous_prompts, timeout) -> Optional[Dict]:
        """
        Simulates the Meta Extraction VLM call.
        """
        await asyncio.sleep(0.05)

        # Use Action constants. 
        # Weights: 20% NEW, 30% CONTINUE, 50% SKIP
        valid_actions = [Action.NEW, Action.CONTINUE, Action.SKIP]
        selected_action = random.choices(valid_actions, weights=[20, 30, 50], k=1)[0]

        mock_response = f"<think>Mocking meta decision...</think><answer>\n<action>{selected_action}</action>\n<meta>Scene: Mock Location</meta>\n<imagery>Detailed shot of {utterance}</imagery></answer>"
        
        return self.strategies['meta_extraction'].process_response(mock_response)

    async def execute_vlm_strategy(
        self, strategy, utterance, context=None, previous_prompts=None, images=None, timeout=300
    ) -> Optional[str]:
        await asyncio.sleep(0.05) 

        # Handle Auxiliary Strategies
        if isinstance(strategy, SummarizeStrategy):
            return "A mock summary."
        if isinstance(strategy, FactCheckStrategy):
            return random.choice(["True", "False"])
        if isinstance(strategy, CaptionStrategy):
            return "A mock caption."

        # Handle Prompt Generation using your utils logic
        logger.info(f"[MOCK] Creating prompt for: {utterance[:20]}...")
        generated_prompt = mock_creation_logic(utterance)
        
        return f"<think>Using utils logic</think>\n{generated_prompt}"

    async def call_image_gen(self, prompt: str, base64_images: Optional[List[str]], timeout: int = 300) -> Optional[str]:
        await asyncio.sleep(0.1)
        logger.info(f"[MOCK] Generating image for: {prompt[:20]}...")
        return mock_gen_logic(prompt)