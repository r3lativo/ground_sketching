# new_serve_vllm.py
import threading
import uvicorn
from fastapi import FastAPI, HTTPException
from transformers import AutoProcessor
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Tuple
from vllm import LLM, SamplingParams
import logging
import base64
from io import BytesIO
from PIL import Image

from src.utils import setup_logging, load_config

# --- Setup ---
setup_logging(log_file='logs/vllm_server.log')
logger = logging.getLogger(__name__)

# --- Configuration Model ---
# The script will fail at startup if any are missing from server_config.yaml
class VLLMConfig(BaseModel):
    model_id: str
    host: str
    port: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    seed: int
    temperature: float
    top_p: float
    dtype: str
    max_model_len: int
    trust_remote_code: bool

# --- Load Configuration ---
try:
    raw_config_dict = load_config('config/server_config.yaml').get("prompt_enhancer_service")
    
    if raw_config_dict is None:
        raise ValueError("'prompt_enhancer_service' section not found in config/server_config.yaml")

    # Parse the dictionary into our model
    # THIS WILL RAISE A 'ValidationError' IF ANY KEY IS MISSING
    config = VLLMConfig(**raw_config_dict)
    
    logger.info("Configuration loaded and validated successfully:")
    logger.info(f"{config.model_dump_json(indent=2)}")

except ValidationError as e:
    logger.critical(f"--- CONFIGURATION ERROR ---")
    logger.critical(f"FATAL: Missing or invalid keys in 'prompt_enhancer_service' section of config/server_config.yaml:")
    logger.critical(f"\n{e}")
    raise SystemExit("Invalid configuration. Please check the log.")
except Exception as e:
    logger.critical(f"FATAL: Failed to load config: {e}", exc_info=True)
    raise SystemExit("Failed to load config.")


# --- Request Body Model ---
class GenerateRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="List of message objects (e.g., {'role': 'user', 'content': ...})")

# --- VLLM Setup ---
logger.info(f"Loading LLM: {config.model_id} with dtype: {config.dtype}")
try:
    llm = LLM(
        model=config.model_id,
        seed=config.seed,
        dtype=config.dtype,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        trust_remote_code=config.trust_remote_code,
        tensor_parallel_size=config.tensor_parallel_size
    )
    processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=config.trust_remote_code)
    logger.info("LLM and processor loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load LLM: {e}", exc_info=True)
    raise SystemExit("LLM loading failed.")

# --- Generation Lock ---
generation_lock = threading.Lock()

# --- FastAPI App ---
app = FastAPI()

# --- Helper Function ---
def _parse_edit_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    """
    Parses messages, decodes base64 images, and returns a processed
    message list and a list of PIL images.
    """
    chat_for_processor = []
    pil_images = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            chat_for_processor.append({"role": "system", "content": content})
        
        elif role == "user":
            processed_user_content = []
            if not isinstance(content, list):
                logger.warning("User content in /edit was not a list, treating as text.")
                processed_user_content.append({"type": "text", "text": str(content)})
            else:
                for part in content:
                    if part["type"] == "text":
                        processed_user_content.append(part)
                    elif part["type"] == "image":
                        try:
                            data_uri = part["image"]
                            header, b64_str = data_uri.split(",", 1)
                            img_bytes = base64.b64decode(b64_str)
                            img_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
                            
                            pil_images.append(img_pil)
                            processed_user_content.append({"type": "image", "image": img_pil})
                        except Exception as e:
                            logger.error(f"Failed to decode image: {e}", exc_info=True)
                            raise ValueError(f"Invalid image data: {e}")
            
            chat_for_processor.append({"role": "user", "content": processed_user_content})
    
    return chat_for_processor, pil_images


@app.post("/generate")
async def handle_generation(request: GenerateRequest):
    """
    Handles a text-only generation request.
    """
    logger.debug(f"Received request messages: {request.messages}")

    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        max_tokens=5000,
    )

    try:
        prompt_str = processor.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        logger.error(f"Error applying chat template: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to apply chat template: {e}")

    with generation_lock:
        try:
            outputs = llm.generate([prompt_str], sampling_params)
            result = outputs[0].outputs[0].text
        except Exception as e:
            logger.error(f"Error during llm.generate: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

    return {"text": result}


@app.post("/edit")
async def handle_edit(request: GenerateRequest):
    """
    Handles a multimodal (image + text) edit request.
    """
    logger.debug(f"Received edit request messages: {request.messages}")

    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        max_tokens=5000,
    )

    try:
        chat_for_processor, pil_images = _parse_edit_messages(request.messages)
        
        if not pil_images:
            raise ValueError("No images found in /edit request.")

        prompt_str = processor.apply_chat_template(
            chat_for_processor,
            tokenize=False,
            add_generation_prompt=True
        )

        mm_data = {
            "image": pil_images[0] if len(pil_images) == 1 else pil_images
        }
        llm_inputs = {
            "prompt": prompt_str,
            "multi_modal_data": mm_data
        }

    except Exception as e:
        logger.error(f"Error processing input or applying template: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to process input messages: {e}")

    with generation_lock:
        try:
            outputs = llm.generate([llm_inputs], sampling_params)
            result = outputs[0].outputs[0].text
        except Exception as e:
            logger.error(f"Error during llm.generate: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

    return {"text": result}


if __name__ == "__main__":
    # This part will only be reached if the config validation passes
    print("#" * 50)
    print("IMPORTANT: Ensure 'export VLLM_ENABLE_V1_MULTIPROCESSING=0' is set in your environment before running.")
    print(f"Starting VLLM server on {config.host}:{config.port}")
    print("#" * 50)
    
    uvicorn.run(app, host=config.host, port=config.port)
