# src/vllm_gateway.py
# This script is a "gateway" or "proxy" server. It runs a FastAPI application
# that receives custom API requests. It translates these requests into the
# standard OpenAI API format and forwards them to a separate vLLM backend server.

import argparse
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Tuple
import logging
import base64
from io import BytesIO
from PIL import Image
import httpx  # Used to make async API calls to the vLLM backend
from contextlib import asynccontextmanager
import sys

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Configuration Models ---
class GatewayConfig(BaseModel):
    host: str
    port: int

class BackendConfig(BaseModel):
    """Generic backend config that fits both LLM and VLM services"""
    model_id: str
    vllm_host: str
    vllm_port: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    dtype: str
    max_model_len: int
    trust_remote_code: bool
    vllm_seed: int

# We parse args early to determine which config to load
def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Gateway")
    parser.add_argument(
        "--service_config_key", 
        type=str, 
        default="vlm_service", 
        help="The key in server_config.yaml to use for backend settings (e.g. 'vlm_service' or 'llm_service')"
    )
    return parser.parse_known_args()[0]

# --- Configuration Loading ---
try:
    from .utils import load_config
    
    # Get the key from CLI
    args = parse_args()
    target_backend_key = args.service_config_key
    
    # Load full config
    full_config = load_config(config_path='config/server_config.yaml')
    
    # 1. Get the shared 'gateway' section
    gateway_conf = full_config.get("gateway")
    if not gateway_conf:
        raise ValueError("Critical: 'gateway' section missing in server_config.yaml")

    # 2. Get the specific backend section (vlm_service OR llm_service)
    backend_conf = full_config.get(target_backend_key)
    if not backend_conf:
        raise ValueError(f"Critical: '{target_backend_key}' section missing in server_config.yaml")

    # Validate
    GATE_CONF = GatewayConfig(**gateway_conf)
    BACK_CONF = BackendConfig(**backend_conf)

    # Set global URL
    VLLM_OPENAI_URL = f"http://{BACK_CONF.vllm_host}:{BACK_CONF.vllm_port}/v1/chat/completions"
    
    logger.info(f"[VLLM_G] Mode: {target_backend_key.upper()}")
    logger.info(f"[VLLM_G] Gateway forwarding to: {VLLM_OPENAI_URL}")

except Exception as e:
    logger.critical(f"[VLLM_G] FATAL CONFIG ERROR: {e}", exc_info=True)
    sys.exit("Invalid configuration.")

# ... (The rest of the file: Request Body Model, Client, Lifespan, etc. remains UNCHANGED) ...
# ...
# Just ensure 'VLMServiceConfig' usages are updated to 'ServiceConfig' if you renamed the class.

# Request Body Model
class GenerateRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="List of message objects")
    seed: int
    top_p: float
    temperature: float
    max_tokens: int

# --- Global Client ---
# Global async client for persistent, efficient connections to the backend
http_client: httpx.AsyncClient = None

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the app's startup and shutdown events.
    This is the modern way to handle 'on_event("startup")' etc.
    """
    # On Startup
    global http_client
    # Initialize the httpx client on startup
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
    timeout = httpx.Timeout(None, connect=60.0)

    http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
    logger.info("[VLLM_G] Gateway started. HTTPX client created with high concurrency limits.")
    
    yield  # The 'yield' separates startup (above) from shutdown (below) code
    
    # On Shutdown
    await http_client.aclose()  # Cleanly close the client connections
    logger.info("[VLLM_G] Gateway shutting down. HTTPX client closed.")

# --- Helper Function ---
def _parse_messages_for_openai(
    messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Parses this app's custom message format and translates it
    into the official OpenAI-compatible message format.
    Specifically, it handles Base64 images.
    """
    openai_messages = []
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        # Pass system messages through unchanged
        if role == "system":
            openai_messages.append({"role": "system", "content": content})
            continue

        if role == "user":
            # Handle simple text-only user messages
            if not isinstance(content, list):
                openai_messages.append({"role": "user", "content": str(content)})
                continue

            # Handle complex multimodal messages (list of text/image parts)
            openai_content_list = []
            for part in content:
                if part["type"] == "text":
                    openai_content_list.append({"type": "text", "text": part["text"]})
                
                # Found an image part, needs conversion
                elif part["type"] == "image":
                    try:
                        data_uri = part["image"] # Assumes image is a Base64 data URI
                        # Test-decode the image to ensure it's valid before sending
                        header, b64_str = data_uri.split(",", 1)
                        img_bytes = base64.b64decode(b64_str)
                        Image.open(BytesIO(img_bytes)).convert("RGB") 
                        
                        # Convert to the OpenAI-compatible 'image_url' format
                        openai_content_list.append({
                            "type": "image_url",
                            "image_url": {"url": data_uri} # The backend server handles the data URI
                        })
                    except Exception as e:
                        logger.error(f"[VLLM_G] Failed to decode image: {e}", exc_info=True)
                        raise ValueError(f"Invalid image data: {e}")
            
            openai_messages.append({"role": "user", "content": openai_content_list})
        
        # Pass assistant messages through unchanged
        elif role == "assistant":
             openai_messages.append({"role": "assistant", "content": msg.get("content")})
            
    return openai_messages


# --- API Call Helper ---
async def call_vllm_backend(payload: Dict[str, Any]) -> str:
    """Helper to call the vLLM backend and parse the response."""
    try:
        # Forward the request to the vLLM backend server
        response = await http_client.post(VLLM_OPENAI_URL, json=payload)
        response.raise_for_status() # Raise an exception for 4xx or 5xx errors
        
        data = response.json()
        # Parse the OpenAI-standard response to get the text
        result_text = data['choices'][0]['message']['content']
        return result_text
        
    # Specific error handling for common network/backend issues
    except httpx.ReadTimeout:
        logger.error(f"[VLLM_G] Gateway timeout: vLLM backend at {VLLM_OPENAI_URL} took too long.")
        raise HTTPException(status_code=504, detail="Gateway timeout: vLLM backend took too long.")
    except httpx.ConnectError:
        logger.error(f"[VLLM_G] Gateway connection error: Could not connect to vLLM backend at {VLLM_OPENAI_URL}.")
        raise HTTPException(status_code=503, detail="Service unavailable: Cannot connect to vLLM backend.")
    except httpx.HTTPStatusError as e:
        logger.error(f"[VLLM_G] vLLM backend error: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"vLLM backend error: {e.response.text}")
    except Exception as e:
        logger.error(f"[VLLM_G] Backend Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- App Creation ---
# Create the FastAPI app instance and link the lifespan manager
app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    """Simple health check for the bash script to verify readiness."""
    return {"status": "ok"}

@app.post("/generate")
async def handle_generation(request: GenerateRequest):
    logger.debug(f"[VLLM_G] Received /generate request")

    try:
        # Standardize the message format (e.g., decode images)
        openai_messages = _parse_messages_for_openai(request.messages)
        # Add the "assistant" turn to prompt the model for a response
        if not openai_messages or openai_messages[-1]["role"] != "assistant":
            openai_messages.append({"role": "assistant", "content": None})
    except Exception as e:
        logger.error(f"[VLLM_G] Error processing input: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to process input messages: {e}")

    # Build the final JSON payload for the OpenAI-compatible backend
    openai_payload = {
        "model": BACK_CONF.model_id,
        "messages": openai_messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "seed": request.seed
    }

    # Call the backend and wait for the response
    result_text = await call_vllm_backend(openai_payload)
    return {"text": result_text}


# --- Main Execution Guard ---
if __name__ == "__main__":
    # This block is *only* executed when you run
    # python -m src.vllm_gateway
    
    # We must use relative imports here for 'python -m' to work
    from .utils import setup_logging
    
    # Setup logging *again* in case this is the main entry point
    # (it will just re-use the existing handlers)
    setup_logging(log_file='logs/vllm_gateway.log')

    # The configs are already loaded at the top.
    # We just need to start the server.
    
    # Use the globally loaded config to start the server
    logger.info(f"[VLLM_G] Starting FastAPI gateway on http://{GATE_CONF.host}:{GATE_CONF.port}")
    
    # Start the FastAPI server
    uvicorn.run(
        app,
        host=GATE_CONF.host,
        port=GATE_CONF.port,
        log_config=None
    )
