# src/api_clients.py

import httpx
import logging
from typing import List, Optional, Dict, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image

from src.utils import load_config, pil_to_base64
from src.strategies import (
    PromptStrategy, 
    TextCreateStrategy, 
    TextEditStrategy, 
    MultimodalEditStrategy,
    MetaStrategy
)

logger = logging.getLogger(__name__)

class APIClient:
    """
    Manages connections to the VLM and Image Generation services.
    Refactored for Connection Pooling and Async Context Management.
    """

    def __init__(self, server_config_path: str, experiment_config_path: str):
        self.config = {}
        self.strategies: Dict[str, PromptStrategy] = {}
        self.image_gen_url = ""
        self.polisher_api_base_url = ""
        self.client: Optional[httpx.AsyncClient] = None
        
        # Load Configuration & Init Components
        self._load_configurations(server_config_path, experiment_config_path)
        self._init_jinja_and_strategies()

    async def __aenter__(self):
        """Initializes the persistent HTTP session."""
        # Connection pooling limits to prevent opening too many file descriptors
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
        self.client = httpx.AsyncClient(limits=limits, timeout=60.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanly closes the HTTP session."""
        if self.client:
            await self.client.aclose()
            logger.info("API Client session closed.")

    def _load_configurations(self, server_conf: str, exp_conf: str):
        try:
            self.config = load_config(config_path=server_conf)
            self.config.update(load_config(config_path=exp_conf))

            # Setup URLs
            self.image_gen_url = f"http://{self.config['image_gen_service']['host']}:{self.config['image_gen_service']['port']}"
            self.polisher_api_base_url = f"http://{self.config['vlm_service']['gateway']['host']}:{self.config['vlm_service']['gateway']['port']}"
            
            logger.info(f"API Client Initialized.")
            logger.info(f"Image Gen URL: {self.image_gen_url}")
            logger.info(f"VLM Base URL: {self.polisher_api_base_url}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}", exc_info=True)
            raise

    def _init_jinja_and_strategies(self):
        try:
            env = Environment(
                loader=FileSystemLoader(self.config['experiment']['jinja']['env']),
                autoescape=select_autoescape(['html', 'xml'])
            )

            # Initialize Strategies with Templates
            self.strategies['meta_extraction'] = MetaStrategy(
                env.get_template(self.config['experiment']['jinja']['meta_extraction']).render()
            )
            self.strategies['oracle_create_context'] = TextCreateStrategy(
                env.get_template(self.config['experiment']['jinja']['initial_start_o']).render()
            )
            self.strategies['oracle_edit_context'] = TextEditStrategy(
                env.get_template(self.config['experiment']['jinja']['initial_edit_o']).render()
            )
            self.strategies['oracle_simple'] = TextCreateStrategy(
                env.get_template(self.config['experiment']['jinja']['final_start_t']).render()
            )
            self.strategies['real_create_context'] = TextCreateStrategy(
                env.get_template(self.config['experiment']['jinja']['initial_start_r2']).render()
            )
            self.strategies['real_edit_context'] = TextEditStrategy(
                env.get_template(self.config['experiment']['jinja']['initial_edit_r']).render()
            )
            self.strategies['real_simple'] = TextCreateStrategy(
                env.get_template(self.config['experiment']['jinja']['initial_start_r1']).render()
            )
            self.strategies['multimodal_edit'] = MultimodalEditStrategy(
                env.get_template(self.config['experiment']['jinja']['final_edit_t']).render()
            )
            
        except Exception as e:
            logger.error(f"Error initializing Jinja2 or Strategies: {e}", exc_info=True)
            raise

    # --- Strategy Factory Method ---

    async def get_meta_and_strategy(
        self,
        is_oracle: bool,
        utterance: Optional[str] = None,
        context: Optional[List[str]] = None,
        previous_prompts: Optional[List[str]] = None,
        has_images: Optional[bool] = False,
        timeout: int = 300
    ) -> (str, PromptStrategy, str, Optional[str], Optional[str]):
        
        if not self.client:
            raise RuntimeError("Client not initialized. Use 'async with APIClient(...)'.")

        # If image exists, force MM edit
        if has_images:
            return None, self.strategies['multimodal_edit'], 'multimodal_edit', None, None

        choice = None
        strategy = self.strategies['meta_extraction']
        client_config = self.config.get("vlm_client", {})
        endpoint = f"{self.polisher_api_base_url}{strategy.endpoint_suffix}"
        
        payload = strategy.build_payload(
            utterance=utterance, config=client_config, context=context,
            previous_prompts=previous_prompts, images=None
        )

        try:
            # REUSE SELF.CLIENT
            response = await self.client.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            
            response_dict = strategy.process_response(result.get("text"))
            if response_dict:
                choice = response_dict.get('action')
                meta_info = response_dict.get('meta')
                imagery_utterance = response_dict.get('imagery_utterance')

        except Exception as e:
            logger.error(f"Meta Extraction Request failed: {e}", exc_info=True)
            return None, None, None, None, None
        
        # Strategy Selection Logic (Remains the same)
        if is_oracle:
            if choice == '[NEW]':
                target = 'oracle_create_context' if context else 'oracle_simple'
                return choice, self.strategies[target], target, meta_info, imagery_utterance
            elif choice == '[CONTINUE]':
                return choice, self.strategies['oracle_edit_context'], 'oracle_edit_context', meta_info, imagery_utterance
        else:
            if choice == '[NEW]':
                target = 'real_create_context' if context else 'real_simple'
                return choice, self.strategies[target], target, meta_info, imagery_utterance
            elif choice == '[CONTINUE]':
                if previous_prompts is None:
                    return choice, self.strategies['real_create_context'], 'real_create_context', meta_info, imagery_utterance
                return choice, self.strategies['real_edit_context'], 'real_edit_context', meta_info, imagery_utterance

        return None, None, None, None, None

    async def execute_vlm_strategy(
        self, strategy: PromptStrategy, utterance: str, 
        context: Optional[List[str]] = None, 
        previous_prompts: Optional[List[str]] = None, 
        images: Optional[List[str]] = None, timeout: int = 300
        ) -> Optional[str]:
        
        if not self.client: raise RuntimeError("Client not initialized.")
        
        client_config = self.config.get("vlm_client", {})
        endpoint = f"{self.polisher_api_base_url}{strategy.endpoint_suffix}"
        payload = strategy.build_payload(utterance, client_config, context, previous_prompts, images)

        try:
            # REUSE SELF.CLIENT
            response = await self.client.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            return strategy.process_response(result.get("text"))
        except Exception as e:
            logger.error(f"VLM Strategy Execution failed: {e}", exc_info=True)
            return None

    async def call_image_gen(self, prompt: str, base64_images: Optional[List[str]], timeout: int = 300) -> Optional[str]:
        
        if not self.client: raise RuntimeError("Client not initialized.")

        endpoint = f"{self.image_gen_url}/img_generate"
        image_gen_config = self.config.get('image_gen_client', {})
        
        payload = {
            "prompt": prompt,
            "seed": image_gen_config.get("seed"),
            "true_cfg_scale": image_gen_config.get("true_cfg_scale"),
            "negative_prompt": image_gen_config.get("negative_prompt"),
            "num_inference_steps": image_gen_config.get("num_inference_steps"),
        }

        if base64_images:
            payload["images"] = base64_images
        else:
            dummy_image_pil = Image.new('RGB', (1024,1024), color='white')
            payload["images"] = [pil_to_base64(dummy_image_pil)]
        
        try:
            # REUSE SELF.CLIENT
            response = await self.client.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            if "image" in result and isinstance(result["image"], str):
                return result["image"]
            return None
        except Exception as e:
            logger.error(f"Image Gen failed: {e}", exc_info=True)
            return None

    async def call_prompt_summarizer(
        context: List[str],
        timeout: int = 300,
    ) -> Optional[str]:
        """
        Calls the vLLM API to summarize a list of prompts. It should tell us what important objects are there by the end of the prompt(Text-in, Text-out)
        """
        endpoint = f"{polisher_api_base_url}/generate"
        logger.info(f"Sending request to Contextual Polisher (Summarize): {endpoint}")

        context_str = "\n".join(context) if isinstance(context, list) else str(context)
        content = f"Here are the prompts:\n{context_str}\n Please summarize the prompts into a list separated by newlines:\n"
        messages = [
            {"role": "system", "content": summarize_prompt},
            {"role": "user", "content": content}
        ]
        
        prompt_polisher_client_config = config["prompt_polisher_client"]

        payload = {
            "messages": messages,
            "seed": prompt_polisher_client_config["seed"],
            "top_p": prompt_polisher_client_config["top_p"],
            "temperature": prompt_polisher_client_config["temperature"],
            "max_tokens": prompt_polisher_client_config["max_tokens"],
        }

        try:
            async with httpx.AsyncClient() as client_http:
                response = await client_http.post(endpoint, json=payload, timeout=timeout)
                response.raise_for_status()
                result = response.json()

                if result.get("text") and len(result["text"]) > 0:
                    enhanced_prompt_raw = result.get("text")
                    
                    if enhanced_prompt_raw and isinstance(enhanced_prompt_raw, str):
                        # polished_text = None
                        # polished_text = json_parser(enhanced_prompt_raw, 'Rewritten')

                        # if enhanced_prompt_raw:
                        polished_text = enhanced_prompt_raw.strip().replace("\n", " ")
                        logger.info(f"Summarization successful. New prompt: '{polished_text[:50]}...'")
                        return polished_text
                        # else:
                        #     logger.warning("Summarization returned empty or invalid response.")
                        #     return None
                    else:
                        logger.warning("Summarization returned empty content.")
                        return None
                else:
                    logger.error(f"Summarization API returned unexpected format: {result}")
                    return None

        except httpx.HTTPStatusError as e:
            logger.error(f"Summarization API request failed with status {e.response.status_code}: {e.response.text}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Error connecting to Summarization API at {endpoint}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during summarization call: {e}", exc_info=True)
            return None

    async def call_image_captioner(
        base64_images: List[str],
        timeout: int = 300,
    ) -> Optional[str]:
        """
        Calls the vLLM API to caption the image. It should tell us what important objects are there in the image(Image-in, Text-out)
        """
        endpoint = f"{polisher_api_base_url}/edit"
        logger.info(f"Sending request to Image Captioner: {endpoint}")

        # Construct multimodal message content
        content: List[Dict[str, Any]] = []
        for img_b64 in base64_images:
            content.append({
                "type": "image",
                "image": f"data:image/jpeg;base64,{img_b64}"
            })
        content.append({"type": "text", "text": "Now describe this image in detail."})

        messages = [
            {"role": "system", "content": caption_prompt},
            {"role": "user", "content": content}
        ]
        
        prompt_polisher_client_config = config["prompt_polisher_client"]

        payload = {
            "messages": messages,
            "seed": prompt_polisher_client_config["seed"],
            "top_p": prompt_polisher_client_config["top_p"],
            "temperature": prompt_polisher_client_config["temperature"],
            "max_tokens": prompt_polisher_client_config["max_tokens"],
        }

        try:
            async with httpx.AsyncClient() as client_http:
                response = await client_http.post(endpoint, json=payload, timeout=timeout)
                response.raise_for_status()
                result = response.json()

                if result.get("text") and len(result["text"]) > 0:
                    enhanced_prompt_raw = result.get("text")
                    
                    if enhanced_prompt_raw and isinstance(enhanced_prompt_raw, str):
                        # polished_text = None
                        # polished_text = json_parser(enhanced_prompt_raw, 'Rewritten')

                        # if polished_text:
                        polished_text = enhanced_prompt_raw.strip().replace("\n", " ")
                        logger.info(f"Captioning successful. New prompt: '{polished_text[:100]}...'")
                        return polished_text
                        # else:
                        #     logger.warning("Captioning returned empty or invalid response.")
                        #     return None
                    else:
                        logger.warning("Captioning returned empty content.")
                        return None
                else:
                    logger.error(f"Captioning API returned unexpected format: {result}")
                    return None

        except httpx.HTTPStatusError as e:
            logger.error(f"Captioning API request failed with status {e.response.status_code}: {e.response.text}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Error connecting to Captioning API at {endpoint}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during captioning call: {e}", exc_info=True)
            return None

    async def call_fact_checker(
        fact: str,
        caption: str,
        timeout: int = 360
    ) -> str:
        """
        Calls the vLLM API to check if the fact is contained in the caption (Image-in, Text-out)
        """
        endpoint = f"{polisher_api_base_url}/generate"
        logger.info(f"Sending request to Fact checker: {endpoint}")

        content = f"Here is the Caption:\n{caption}\nhere is the Fact:{fact}\n. Answer strictly with 'True' or 'False'. Here is your answer:"
        messages = [
            {"role": "system", "content": fact_prompt},
            {"role": "user", "content": content}
        ]
        
        prompt_polisher_client_config = config["prompt_polisher_client"]

        payload = {
            "messages": messages,
            "seed": prompt_polisher_client_config["seed"],
            "top_p": prompt_polisher_client_config["top_p"],
            "temperature": prompt_polisher_client_config["temperature"],
            "max_tokens": prompt_polisher_client_config["max_tokens"],
        }

        try:
            async with httpx.AsyncClient() as client_http:
                response = await client_http.post(endpoint, json=payload, timeout=timeout)
                response.raise_for_status()
                result = response.json()

                if result.get("text") and len(result["text"]) > 0:
                    enhanced_prompt_raw = result.get("text")
                    
                    if enhanced_prompt_raw and isinstance(enhanced_prompt_raw, str):
                        # polished_text = None
                        # polished_text = json_parser(enhanced_prompt_raw, 'Rewritten')

                        # if polished_text:
                        polished_text = enhanced_prompt_raw.strip().replace("\n", " ")
                        logger.info(f"Fact check successful.")
                        return polished_text
                        # else:
                        #     logger.warning("SFact check returned empty or invalid response.")
                        #     return None
                    else:
                        logger.warning("Fact check returned empty content.")
                        return None
                else:
                    logger.error(f"Fact check API returned unexpected format: {result}")
                    return None

        except httpx.HTTPStatusError as e:
            logger.error(f"Fact check API request failed with status {e.response.status_code}: {e.response.text}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Error connecting to Fact check API at {endpoint}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during fsct check call: {e}", exc_info=True)
            return None
