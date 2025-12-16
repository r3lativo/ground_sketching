# src/strategies.py

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union

# Import the centralized parsers
from src.utils import json_parser, thinking_parser

logger = logging.getLogger(__name__)

class PromptStrategy(ABC):
    """
    Abstract base class for VLM prompt strategies.
    """

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    @property
    def default_params(self) -> Dict[str, Any]:
        """Returns strategy-specific overrides for generation parameters."""
        return {}

    @property
    @abstractmethod
    def endpoint_suffix(self) -> str:
        """Returns the URL suffix (e.g., '/generate' or '/edit')."""
        pass

    @abstractmethod
    def build_user_content(
        self, 
        utterance: str, 
        context: Union[List[str], Dict[str, Any], None] = None,
        previous_prompts: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        **kwargs  # Allow custom arguments in abstract method
    ) -> Union[str, List[Dict[str, Any]]]:
        pass

    def build_payload(
        self, 
        utterance: str, 
        config: Dict[str, Any],
        context: Union[List[str], Dict[str, Any], None] = None,
        previous_prompts: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        **kwargs  # Accept extra args here to prevent TypeError
    ) -> Dict[str, Any]:
        """
        Constructs the full JSON payload for the API request.
        Passes **kwargs down to build_user_content.
        """
        user_content = self.build_user_content(
            utterance=utterance, 
            context=context, 
            previous_prompts=previous_prompts, 
            images=images, 
            **kwargs
        )
        
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
            logger.info(f"[STRATEGIES] VLM Thought: {thinking_part.replace("\n", " ")}")

        if not answer_part:
            logger.warning("[STRATEGIES] VLM returned no answer after parsing </think>.")
            return None
        
        logger.info(f"[STRATEGIES] VLM Answer: {answer_part.replace("\n", " ")}")

        # 2. Strategy-specific parsing
        return self._parse_answer(answer_part.strip())

    @abstractmethod
    def _parse_answer(self, answer_text: str) -> Any:
        pass


class TextCreateStrategy(PromptStrategy):
    @property
    def endpoint_suffix(self) -> str:
        return "/generate"

    def build_user_content(self, utterance, context=None, previous_prompts=None, images=None, **kwargs) -> str:
        ctx_str = f"Conversation Context:\n{context}\n" if context else ""
        return f"{ctx_str}Target Utterance:\n{utterance}\n"

    def _parse_answer(self, answer_text: str) -> str:
        return answer_text.strip().replace("\n", " ")


class TextEditStrategy(PromptStrategy):
    @property
    def endpoint_suffix(self) -> str:
        return "/generate"

    def build_user_content(self, utterance, context=None, previous_prompts=None, images=None, **kwargs) -> str:
        ctx_str = f"Conversation Context:\n{context}\n" if context else ""
        hist_str = f"Previous Prompts:\n{previous_prompts}\n" if previous_prompts else ""
        return f"{ctx_str}Target Utterance:\n{utterance}\n{hist_str}"

    def _parse_answer(self, answer_text: str) -> str:
        return answer_text.strip().replace("\n", " ")


class MultimodalEditStrategy(PromptStrategy):
    @property
    def endpoint_suffix(self) -> str:
        return "/edit"

    def build_user_content(self, utterance, context=None, previous_prompts=None, images=None, **kwargs) -> List[Dict[str, Any]]:
        content = []
        if images:
            for img_b64 in images:
                img_str = img_b64 if "data:image" in img_b64 else f"data:image/jpeg;base64,{img_b64}"
                content.append({"type": "image", "image": img_str})
        
        content.append({"type": "text", "text": utterance})
        return content

    def _parse_answer(self, answer_text: str) -> str:
        return answer_text.strip().replace("\n", " ")


class MetaStrategy(TextEditStrategy):
    def build_user_content(self, utterance, context=None, previous_prompts=None, images=None, **kwargs) -> str:
        ctx_str = f"Conversation Context <context>:\n{context if context else 'None'}"
        hist_str = f"Previous Prompts <previous>:\n{previous_prompts if previous_prompts else 'None'}"
        target_str = f"Target Utterance <target>:\n{utterance}"
        return f"{ctx_str}\n{hist_str}\n{target_str}"

    def _parse_answer(self, answer_text: str) -> Dict[str, Optional[str]]:
        return json_parser(answer_text)

# --- UPDATED STRATEGIES ---

class SummarizeStrategy(PromptStrategy):
    """
    Summarizes prompts into a JSON list of facts.
    Expects 'context' to contain the list of prompts to summarize.
    """
    @property
    def default_params(self) -> Dict[str, Any]:
        return {"temperature": 0.6}

    @property
    def endpoint_suffix(self) -> str:
        return "/generate"

    def build_user_content(self, utterance, context=None, previous_prompts=None, images=None, **kwargs) -> str:
        prompts_str = "\n".join(context) if context else str(context)
        return f"Here are the prompts:\n{prompts_str}\n"

    def _parse_answer(self, answer_text: str) -> List[str]:
        parsed = json_parser(answer_text)
        if isinstance(parsed, dict):
            return parsed.get("facts")
        return answer_text


class CaptionStrategy(MultimodalEditStrategy):
    """
    Captions an image.
    Inherits from MultimodalEditStrategy.
    """
    pass


class FactCheckStrategy(MultimodalEditStrategy):
    """
    Verifies a list of facts against an image (Multimodal).
    Expects 'utterance' to be the JSON string of facts.
    Expects 'images' to be passed in build_payload.
    """
    @property
    def default_params(self) -> Dict[str, Any]:
        return {"temperature": 0.6}

    @property
    def endpoint_suffix(self) -> str:
        return "/edit"

    def build_user_content(self, utterance, context=None, previous_prompts=None, images=None, **kwargs) -> List[Dict[str, Any]]:
        content = []
        if images:
            for img_b64 in images:
                # Ensure standard base64 header
                img_str = img_b64 if "data:image" in img_b64 else f"data:image/jpeg;base64,{img_b64}"
                content.append({"type": "image", "image": img_str})
        
        content.append({"type": "text", "text": f"Facts to Check:\n{utterance}"})
        return content

    def _parse_answer(self, answer_text: str) -> List[Dict[str, Any]]:
        # Returns the list of verification dicts: [{'fact':..., 'verdict':...}]
        parsed = json_parser(answer_text)
        if isinstance(parsed, dict):
            return parsed.get("verification")
        return answer_text


class TripletsExtractionStrategy(PromptStrategy):
    """
    Extracts triplets (Frame_A, Relation, Frame_B) based on a relation directive and temporal context.
    """
    @property
    def endpoint_suffix(self) -> str:
        return "/generate"

    def build_user_content(
        self, 
        utterance: str, 
        context: Union[List[str], Dict[str, Any], None] = None,
        previous_prompts=None, 
        images=None, 
        **kwargs
    ) -> str:
        """
        Robustly handles input:
        - If 'relation' is passed in kwargs, uses that.
        - Otherwise, assumes 'utterance' IS the relation.
        - Expects 'context' to be the neighborhood dict.
        """
        
        # 1. Determine the Relation
        # If the caller passed relation="...", use it. Else use the utterance string.
        relation_text = kwargs.get('relation', utterance)

        # 2. Parse Context
        ctx = context if isinstance(context, dict) else {}
        
        prev_txt = ctx.get('prev_text', '')
        curr_txt = ctx.get('curr_text', '')
        next_txt = ctx.get('next_text', '')
        
        p_id = ctx.get('prev_frame_id')
        c_id = ctx.get('current_frame_id')
        n_id = ctx.get('next_frame_id')
        
        # 3. Build Prompt
        prompt = f"RELATION: '{relation_text}'\n\n--- CONTEXT ---\n"
        if p_id: prompt += f"[PREVIOUS SCENE ID: {p_id}]:\n{prev_txt}\n\n"
        if c_id: prompt += f"[CURRENT SCENE ID: {c_id}]:\n{curr_txt}\n\n"
        if n_id: prompt += f"[NEXT SCENE ID: {n_id}]:\n{next_txt}"
        
        return prompt

    def _parse_answer(self, answer_text: str):
        return json_parser(answer_text)
