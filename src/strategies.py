# src/strategies.py

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union

# Import the centralized parsers
from src.utils import json_parser, meta_parser, thinking_parser

logger = logging.getLogger(__name__)

class PromptStrategy(ABC):
    """
    Abstract base class for VLM prompt strategies.
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

    def process_response(self, raw_text: str) -> Any:
        """
        Common post-processing: Separate thoughts from answer, then parse specific format.
        """
        if not raw_text:
            return None

        # 1. Separate Thinking from Answer using robust parser
        parsed_output = thinking_parser(raw_text)
        thinking_part = parsed_output.get("thinking")
        answer_part = parsed_output.get("answer")

        if thinking_part:
            # Log thought
            logger.info(f"VLM Thought: {thinking_part.replace("\n", " ")}")

        if not answer_part:
            logger.warning("VLM returned no answer after parsing </think>.")
            return None
        
        logger.info(f"VLM Answer: {answer_part.replace("\n", " ")}")

        # 2. Strategy-specific parsing
        return self._parse_answer(answer_part.strip())

    @abstractmethod
    def _parse_answer(self, answer_text: str) -> Any:
        pass


class TextCreateStrategy(PromptStrategy):
    @property
    def endpoint_suffix(self) -> str:
        return "/generate"

    def build_user_content(self, utterance, context=None, previous_prompts=None, images=None) -> str:
        ctx_str = f"Conversation Context:\n{context}\n" if context else ""
        return f"{ctx_str}Target Utterance:\n{utterance}\nRewritten Prompt:\n"

    def _parse_answer(self, answer_text: str) -> str:
        return answer_text


class TextEditStrategy(PromptStrategy):
    @property
    def endpoint_suffix(self) -> str:
        return "/generate"

    def build_user_content(self, utterance, context=None, previous_prompts=None, images=None) -> str:
        ctx_str = f"Conversation Context:\n{context}\n" if context else ""
        hist_str = f"Previous Prompts:\n{previous_prompts}\n" if previous_prompts else ""
        return f"{ctx_str}Target Utterance:\n{utterance}\n{hist_str}Rewritten:\n"

    def _parse_answer(self, answer_text: str) -> str:
        # Uses robust utils parser
        return json_parser(answer_text, "Rewritten")


class MultimodalEditStrategy(PromptStrategy):
    @property
    def endpoint_suffix(self) -> str:
        return "/edit"

    def build_user_content(self, utterance, context=None, previous_prompts=None, images=None) -> List[Dict[str, Any]]:
        content = []
        if images:
            for img_b64 in images:
                img_str = img_b64 if "data:image" in img_b64 else f"data:image/jpeg;base64,{img_b64}"
                content.append({"type": "image", "image": img_str})
        
        content.append({"type": "text", "text": utterance})
        return content

    def _parse_answer(self, answer_text: str) -> str:
        return json_parser(answer_text, "Rewritten")


class MetaStrategy(TextEditStrategy):
    def build_user_content(self, utterance, context=None, previous_prompts=None, images=None) -> str:
        ctx_str = f"Conversation Context <context>:\n{context if context else 'None'}"
        hist_str = f"Previous Prompts <previous>:\n{previous_prompts if previous_prompts else 'None'}"
        target_str = f"Target Utterance <target>:\n{utterance}"
        return f"{ctx_str}\n{hist_str}\n{target_str}"

    def _parse_answer(self, answer_text: str) -> Dict[str, Optional[str]]:
        # Uses robust utils parser
        return meta_parser(answer_text)
