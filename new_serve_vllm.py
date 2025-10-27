# new_serve_vllm.py
import threading
import uvicorn
from fastapi import FastAPI, HTTPException
from transformers import AutoProcessor
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
import logging
import base64
from io import BytesIO      # <--- MISSING IMPORT
from PIL import Image       # <--- MISSING IMPORT

from src.utils import setup_logging, load_config
# from qwen_vl_utils import process_vision_info

# Setup logging
setup_logging(log_file='logs/server_launcher.log') # Consider a dedicated log?
logger = logging.getLogger(__name__)


config = load_config('config/server_config.yaml')
config = config.get("prompt_enhancer_service")

model_id = config.get('model_id')
polish_host = config.get('host')
polish_port = config.get('port')
tp = config.get('tensor_parallel_size', 1) # Add default
mem_util = config.get('gpu_memory_utilization')
seed = config.get('seed')
temperature = config.get('temperature')
top_p = config.get('top_p')
dtype = config.get('dtype', 'bfloat16') # Default to bfloat16 is often better
max_len = config.get('max_model_len', 15000) # Get max_model_len from config
trust_code = config.get('trust_remote_code', True) # Default True for Qwen

# --- Request Body Model ---
# Use a more specific model, mirroring OpenAI structure slightly
class GenerateRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="List of message objects (e.g., {'role': 'user', 'content': ...})")
    # You could add sampling params here if you want client control
    # max_tokens: Optional[int] = 5000 # Example


# --- VLLM Setup ---
logger.info(f"Loading LLM: {model_id} with dtype: {dtype}")
try:
    llm = LLM(
        model=model_id,
        seed=seed,
        dtype=dtype,
        gpu_memory_utilization=mem_util,
        max_model_len=max_len, # Use configured max_len
        trust_remote_code=trust_code, # Pass trust_remote_code
        tensor_parallel_size=tp
    )
    # Get the tokenizer after LLM is loaded
    tokenizer = llm.get_tokenizer()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    logger.info("LLM and tokenizer loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load LLM: {e}", exc_info=True)
    raise SystemExit("LLM loading failed.")


# --- Generation Lock ---
# This still makes sense if vLLM/model has issues with true concurrency
generation_lock = threading.Lock()

# --- FastAPI App ---
app = FastAPI()

@app.post("/generate") # Changed endpoint to /generate
async def handle_generation(request: GenerateRequest):
    """
    Handles a generation request using the loaded LLM.
    Applies chat template before generating.
    """
    logger.debug(f"Received request messages: {request.messages}")

    # Define sampling params (could also be part of the request)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        seed=seed, # Applying seed here might override per-request if needed later
        max_tokens=5000, # Set a reasonable max output tokens
    )

    try:
        # --- Apply Chat Template ---
        # This converts the list of message dicts into the model's expected input string format
        # Important: Assumes the tokenizer loaded by vLLM handles the multimodal format correctly
        prompt_str = tokenizer.apply_chat_template(
            request.messages,
            tokenize=False, # We want the string, not tokens
            add_generation_prompt=True # Add the prompt for the assistant's turn
        )
        logger.debug(f"Applied chat template, resulting prompt string (start): {prompt_str[:200]}...")
    except Exception as e:
        logger.error(f"Error applying chat template: {e}", exc_info=True)
        # It's possible the tokenizer doesn't support the image dict format directly here
        raise HTTPException(status_code=400, detail=f"Failed to apply chat template to messages: {e}")

    # Acquire the lock for generation
    logger.debug("Acquiring generation lock...")
    with generation_lock:
        logger.debug("Lock acquired. Starting generation...")
        try:
            # Pass the formatted string(s) to llm.generate
            outputs = llm.generate([prompt_str], sampling_params) # Pass as list
            result = outputs[0].outputs[0].text
            logger.debug("Generation finished.")
        except Exception as e:
            logger.error(f"Error during llm.generate: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")
    # Lock released automatically
    logger.debug("Generation lock released.")

    return {"text": result}


@app.post("/edit") # Changed endpoint to /generate
async def handle_edit(request: GenerateRequest):
    """
    Handles a generation request using the loaded LLM.
    Applies chat template before generating.
    """
    logger.debug(f"Received request messages: {request.messages}")

    # Define sampling params (could also be part of the request)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        seed=seed, # Applying seed here might override per-request if needed later
        max_tokens=5000, # Set a reasonable max output tokens
    )

    try:

        chat_for_processor = []
        pil_images = []

        for msg in request.messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                chat_for_processor.append({"role": "system", "content": content})
            
            elif role == "user":
                # User content is a list of dicts (text/image)
                processed_user_content = []
                for part in content:
                    if part["type"] == "text":
                        processed_user_content.append(part)
                    elif part["type"] == "image":
                        # Decode the base64 image
                        data_uri = part["image"]
                        header, b64_str = data_uri.split(",", 1)
                        img_bytes = base64.b64decode(b64_str)
                        img_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
                        
                        pil_images.append(img_pil)
                        # Add the PIL image to the content for the processor
                        processed_user_content.append({"type": "image", "image": img_pil})
                
                chat_for_processor.append({"role": "user", "content": processed_user_content})

        # --- Apply Chat Template ---
        # This converts the list of message dicts into the model's expected input string format
        # Important: Assumes the tokenizer loaded by vLLM handles the multimodal format correctly
        prompt_str = processor.apply_chat_template(
            chat_for_processor,
            tokenize=False, # We want the string, not tokens
            add_generation_prompt=True # Add the prompt for the assistant's turn
        )

        mm_data = {
            "image": pil_images[0] if len(pil_images) == 1 else pil_images
        }

        # image_input, _ = process_vision_info(request.messages)
        # mm_data = {}
        # mm_data["image"] = image_input

        llm_inputs = {
            "prompt": prompt_str,
            "multi_modal_data": mm_data
        }

        logger.debug(f"Applied chat template, resulting prompt string (start): {prompt_str[:200]}...")
    except Exception as e:
        logger.error(f"Error applying chat template: {e}", exc_info=True)
        # It's possible the tokenizer doesn't support the image dict format directly here
        raise HTTPException(status_code=400, detail=f"Failed to apply chat template to messages: {e}")

    # Acquire the lock for generation
    logger.debug("Acquiring generation lock...")
    with generation_lock:
        logger.debug("Lock acquired. Starting generation...")
        try:
            # Pass the formatted string(s) to llm.generate
            outputs = llm.generate(
                [llm_inputs],
                sampling_params,
            )
            result = outputs[0].outputs[0].text
            logger.debug("Generation finished.")
        except Exception as e:
            logger.error(f"Error during llm.generate: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")
    # Lock released automatically
    logger.debug("Generation lock released.")

    return {"text": result}

@app.get("/health")
async def health_check():
    # Basic check, maybe add more sophisticated checks later
    return {"status": "ok"}


if __name__ == "__main__":
    # Remove the VLLM_ENABLE_V1_MULTIPROCESSING export from here,
    # it should be set in the environment *before* running the script.
    print("#" * 50)
    print("IMPORTANT: Ensure 'export VLLM_ENABLE_V1_MULTIPROCESSING=0' is set in your environment before running.")
    print("#" * 50)

    # Run the server with Uvicorn
    uvicorn.run(app, host=polish_host, port=polish_port)
