# src/mock_api_clients.py

import asyncio
import logging
import random
from typing import List, Optional, Dict, Any, Tuple

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

    async def _fetch_meta_info(self, utterance, context, previous_prompts, timeout: Optional[float] = None) -> Optional[Dict]:
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
        self, strategy, utterance, context=None, previous_prompts=None, images=None, timeout: Optional[float] = None
    ) -> Any:
        # NOTE: This is a fallback. The specific call_* methods below override this for specific tasks.
        await asyncio.sleep(0.05) 

        # Handle Prompt Generation using your utils logic
        logger.info(f"[MOCK] Creating prompt for: {utterance[:20]}...")
        generated_prompt = mock_creation_logic(utterance)
        
        return f"<think>Using utils logic</think>\n{generated_prompt}"

    async def call_image_gen(self, prompt: str, base64_images: Optional[List[str]], seed: Optional[int] = None, timeout: Optional[float] = None) -> Optional[str]:
        await asyncio.sleep(0.1)
        logger.info(f"[MOCK] Generating image for: {prompt[:20]}... (Seed: {seed})")
        return mock_gen_logic(prompt)

    # --- NEW MOCK FUNCTIONS ---

    async def call_prompt_summarizer(self, context: List[str], timeout: Optional[float] = None) -> List[str]:
        """
        Mocks the decomposition of prompts into facts.
        """
        await asyncio.sleep(0.05)
        logger.info("[MOCK] Summarizing prompts into facts...")
        # Return a static list of mock facts for testing
        return [
            "There is a mock object in the center",
            "The object has a red outline",
            "There is a blue sky background"
        ]

    async def call_image_captioner(self, base64_images: List[str], timeout: Optional[float] = None) -> Optional[str]:
        """
        Mocks generating a caption.
        """
        await asyncio.sleep(0.05)
        logger.info("[MOCK] Captioning image...")
        return "A detailed mock caption describing a test scene with red and blue objects."

    async def call_visual_verifier(self, facts: List[str], base64_image: str, timeout: Optional[float] = None) -> List[Dict]:
        """
        Mocks checking facts against an image. Returns random verdicts.
        """
        await asyncio.sleep(0.05)
        logger.info("[MOCK] Verifying facts against image...")
        
        results = []
        for fact in facts:
            # Randomly decide if true or false to test scoring logic
            is_true = random.choice([True, True, False]) # Slight bias towards True
            results.append({
                "fact": fact,
                "box": [100, 100, 200, 200] if is_true else [0, 0, 0, 0],
                "verdict": is_true
            })
        return results

    async def verify_image_faithfulness(
        self, 
        base64_image: str, 
        facts: Optional[List[str]] = None, 
        context: Optional[List[str]] = None,
        timeout: Optional[float] = None
    ) -> Tuple[float, List[Dict]]:
        """
        Mocks the full verification loop. 
        Returns a random score to test 'Best-of-N' selection.
        """
        await asyncio.sleep(0.05)
        
        # 1. Mock getting facts if needed
        if not facts:
            facts = await self.call_prompt_summarizer(context or [])

        # 2. Mock verification details
        # We generate a random score between 0.0 and 1.0 to ensure the pipeline
        # actually has to "choose" the best one.
        logger.info("[MOCK] Calculating faithfulness score...")
        
        # Randomly verify some facts
        details = []
        confirmed_count = 0
        
        for fact in facts:
            # Weighted coin flip for realism
            is_verified = random.random() > 0.3 
            if is_verified:
                confirmed_count += 1
            
            details.append({
                "fact": fact,
                "box": [50, 50, 150, 150] if is_verified else [0,0,0,0],
                "verdict": is_verified
            })

        final_score = confirmed_count / len(facts) if facts else 0.0
        
        return final_score, details
