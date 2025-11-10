# vllm_gateway.py

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Tuple
import logging
import base64
from io import BytesIO
from PIL import Image
import httpx  # Use httpx to make API calls
from contextlib import asynccontextmanager

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Configuration Model ---
class VLLMConfig(BaseModel):
    model_id: str
    host: str
    port: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    dtype: str
    max_model_len: int
    trust_remote_code: bool
    seed: int
    vllm_host: str
    vllm_port: int

# --- Configuration ---
VLLM_CONFIG: Dict[str, Any] = {}
VLLM_OPENAI_URL: str = ""

# --- Request Body Model ---
class GenerateRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="List of message objects (e.g., {'role': 'user', 'content': ...})")
    seed: int
    top_p: float
    temperature: float
    max_tokens: int

# --- Global Client ---
http_client: httpx.AsyncClient = None

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # On Startup
    global http_client
    http_client = httpx.AsyncClient(timeout=300.0) # Long timeout for LLMs
    logger.info("Gateway started. HTTPX client created.")
    yield
    # On Shutdown
    await http_client.aclose()
    logger.info("Gateway shutting down. HTTPX client closed.")

# --- Helper Function ---
def _parse_messages_for_openai(
    messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Parses messages, finds base64 images, and returns
    a list of messages formatted for the vLLM OpenAI API.
    
    Returns: openai_messages
    """
    openai_messages = []
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            openai_messages.append({"role": "system", "content": content})
            continue

        if role == "user":
            if not isinstance(content, list):
                openai_messages.append({"role": "user", "content": str(content)})
                continue

            openai_content_list = []
            for part in content:
                if part["type"] == "text":
                    openai_content_list.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image":
                    try:
                        data_uri = part["image"]
                        header, b64_str = data_uri.split(",", 1)
                        img_bytes = base64.b64decode(b64_str)
                        Image.open(BytesIO(img_bytes)).convert("RGB") 
                        
                        openai_content_list.append({
                            "type": "image_url",
                            "image_url": {"url": data_uri}
                        })
                    except Exception as e:
                        logger.error(f"Failed to decode image: {e}", exc_info=True)
                        raise ValueError(f"Invalid image data: {e}")
            
            openai_messages.append({"role": "user", "content": openai_content_list})
        
        elif role == "assistant":
             openai_messages.append({"role": "assistant", "content": msg.get("content")})
            
    return openai_messages


# --- API Call Helper ---
async def call_vllm_backend(payload: Dict[str, Any]) -> str:
    """Helper to call the vLLM backend and parse the response."""
    try:
        response = await http_client.post(VLLM_OPENAI_URL, json=payload)
        response.raise_for_status() 
        
        data = response.json()
        result_text = data['choices'][0]['message']['content']
        return result_text
        
    except httpx.ReadTimeout:
        logger.error(f"Gateway timeout: vLLM backend at {VLLM_OPENAI_URL} took too long.")
        raise HTTPException(status_code=504, detail="Gateway timeout: vLLM backend took too long.")
    except httpx.ConnectError:
        logger.error(f"Gateway connection error: Could not connect to vLLM backend at {VLLM_OPENAI_URL}.")
        raise HTTPException(status_code=503, detail="Service unavailable: Cannot connect to vLLM backend.")
    except httpx.HTTPStatusError as e:
        logger.error(f"vLLM backend error: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"vLLM backend error: {e.response.text}")
    except Exception as e:
        logger.error(f"Internal gateway error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal gateway error: {str(e)}")


# --- App Creation ---
app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def handle_generation(request: GenerateRequest):
    logger.debug(f"Received /generate request")

    try:
        openai_messages = _parse_messages_for_openai(request.messages)
        if not openai_messages or openai_messages[-1]["role"] != "assistant":
            openai_messages.append({"role": "assistant", "content": None})
    except Exception as e:
        logger.error(f"Error processing input: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to process input messages: {e}")

    openai_payload = {
        "model": VLLM_CONFIG['model_id'],
        "messages": openai_messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "seed": request.seed
    }

    result_text = await call_vllm_backend(openai_payload)
    return {"text": result_text}


@app.post("/edit")
async def handle_edit(request: GenerateRequest):
    logger.debug(f"Received /edit request")
        
    try:
        openai_messages = _parse_messages_for_openai(request.messages)
        if not openai_messages or openai_messages[-1]["role"] != "assistant":
            openai_messages.append({"role": "assistant", "content": None})
    except Exception as e:
        logger.error(f"Error processing input: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to process input messages: {e}")

    openai_payload = {
        "model": VLLM_CONFIG['model_id'],
        "messages": openai_messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "seed": request.seed
    }

    result_text = await call_vllm_backend(openai_payload)
    return {"text": result_text}
    

# --- Main Execution Guard ---
if __name__ == "__main__":
    from src.utils import setup_logging, load_config
    
    setup_logging(log_file='logs/vllm_gateway.log')

    try:
        raw_config_dict = load_config(config_path='config/server_config.yaml').get("prompt_polisher_service")
        if raw_config_dict is None:
            raise ValueError("'prompt_polisher_service' section not found")
        
        # This line will now work
        VLLMConfig(**raw_config_dict)
        VLLM_CONFIG = raw_config_dict
        
        VLLM_OPENAI_URL = f"http://{VLLM_CONFIG['vllm_host']}:{VLLM_CONFIG['vllm_port']}/v1/chat/completions"
        
        logger.info("Gateway configuration loaded and validated successfully.")
        logger.info(f"Gateway will forward requests to: {VLLM_OPENAI_URL}")

    except ValidationError as e:
        logger.critical(f"--- CONFIGURATION ERROR ---")
        logger.critical(f"FATAL: Missing or invalid keys in 'prompt_polisher_service' section of config/server_config.yaml:")
        logger.critical(f"\n{e}")
        raise SystemExit("Invalid configuration. Please check the log.")
    except Exception as e:
        logger.critical(f"FATAL: Failed to load config: {e}", exc_info=True)
        raise SystemExit("Failed to load config.")

    logger.info("Gateway does not load LLM. This is correct.")
    logger.info(f"Starting FastAPI gateway on http://{VLLM_CONFIG['host']}:{VLLM_CONFIG['port']}")
    
    # 'app' is already defined
    uvicorn.run(app, host=VLLM_CONFIG['host'], port=VLLM_CONFIG['port'])
