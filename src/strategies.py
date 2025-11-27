# src/strategies.py

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from src.utils import json_parser, thinking_parser

logger = logging.getLogger(__name__)

class PromptStrategy(ABC):
    """
    Abstract base class for VLM prompt strategies.
    Encapsulates:
      1. Which API endpoint to use.
      2. How to format the input payload (User Message).
      3. How to parse the output response.
    """

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    @property
    @abstractmethod
    def endpoint_suffix(self) -> str:
        """Returns the URL suffix (e.g., '/generate' or '/edit')."""
        pass

    @abstractmethod
    def build_user_content(
        self, 
        utterance: str, 
        context: Optional[List[str]] = None,
        previous_prompts: Optional[List[str]] = None,
        images: Optional[List[str]] = None
    ) -> Union[str, List[Dict[str, Any]]]:
        """Constructs the 'content' part of the user message."""
        pass

    def build_payload(
        self, 
        utterance: str, 
        config: Dict[str, Any],
        context: Optional[List[str]] = None,
        previous_prompts: Optional[List[str]] = None,
        images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Constructs the full JSON payload for the API request.
        """
        user_content = self.build_user_content(utterance, context, previous_prompts, images)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]

        # Standard VLM parameters
        return {
            "messages": messages,
            "seed": config.get("seed"),
            "top_p": config.get("top_p"),
            "temperature": config.get("temperature"),
            "max_tokens": config.get("max_tokens"),
        }

    def process_response(self, raw_text: str) -> Optional[str]:
        """
        Common post-processing:
        1. Parse <think> blocks.
        2. Log thinking.
        3. Parse JSON if required by the specific strategy.
        """
        if not raw_text:
            return None

        # 1. Separate Thinking from Answer
        parsed_output = thinking_parser(raw_text)
        thinking_part = parsed_output.get("thinking")
        answer_part = parsed_output.get("answer")

        # Log thinking for debugging
        if thinking_part:
            logger.info(f"VLM Thought: {thinking_part[:200]}..." if len(thinking_part) > 200 else f"VLM Thought: {thinking_part}")

        if not answer_part:
            logger.warning("VLM returned no answer after parsing </think>.")
            return None

        # 2. Strategy-specific parsing (JSON vs Raw)
        return self._parse_answer(answer_part.strip())

    @abstractmethod
    def _parse_answer(self, answer_text: str) -> str:
        """Custom parsing logic (e.g., extract JSON)."""
        pass


class TextCreateStrategy(PromptStrategy):
    """
    Strategy for Initial Prompt Creation (Text-Only).
    - Endpoint: /generate
    - Format: Raw Text output (No JSON expected).
    """

    @property
    def endpoint_suffix(self) -> str:
        return "/generate"

    def build_user_content(
        self, 
        utterance: str, 
        context: Optional[List[str]] = None,
        previous_prompts: Optional[List[str]] = None, 
        images: Optional[List[str]] = None
    ) -> str:
        
        # Format the context block
        ctx_str = f"Conversation Context:\n{context}\n" if context else ""
        
        # Format: Context -> Utterance -> Header
        return f"{ctx_str}Target Utterance:\n{utterance}\nRewritten Prompt:\n"

    def _parse_answer(self, answer_text: str) -> str:
        # Expecting raw text, so just return it.
        return answer_text


class TextEditStrategy(PromptStrategy):
    """
    Strategy for Text-Based Prompt Refinement.
    - Endpoint: /generate
    - Format: Expects JSON output with key 'Rewritten'.
    """

    @property
    def endpoint_suffix(self) -> str:
        return "/generate"

    def build_user_content(
        self, 
        utterance: str, 
        context: Optional[List[str]] = None,
        previous_prompts: Optional[List[str]] = None,
        images: Optional[List[str]] = None
    ) -> str:
        
        ctx_str = f"Conversation Context:\n{context}\n" if context else ""
        hist_str = f"Previous Prompts:\n{previous_prompts}\n" if previous_prompts else ""
        
        return f"{ctx_str}Target Utterance:\n{utterance}\n{hist_str}Rewritten:\n"

    def _parse_answer(self, answer_text: str) -> str:
        # Attempt to parse JSON
        json_key = "Rewritten"
        return json_parser(answer_text, json_key)


class MultimodalEditStrategy(PromptStrategy):
    """
    Strategy for Image+Text Based Refinement.
    - Endpoint: /edit
    - Format: Expects JSON output with key 'Rewritten'.
    - Payload: List of dictionaries (Multimodal).
    """

    @property
    def endpoint_suffix(self) -> str:
        return "/edit"

    def build_user_content(
        self, 
        utterance: str, 
        context: Optional[List[str]] = None,
        previous_prompts: Optional[List[str]] = None,
        images: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        
        content = []
        
        # Add Images
        if images:
            for img_b64 in images:
                # Ensure we don't double-prefix if logic elsewhere adds headers
                if "data:image" in img_b64:
                    img_str = img_b64
                else:
                    img_str = f"data:image/jpeg;base64,{img_b64}"
                
                content.append({"type": "image", "image": img_str})
        
        # Add Text
        content.append({"type": "text", "text": utterance})
        
        return content

    def _parse_answer(self, answer_text: str) -> str:
        # Attempt to parse JSON
        json_key = "Rewritten"
        return json_parser(answer_text, json_key)
