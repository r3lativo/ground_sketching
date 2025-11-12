# vllm_gateway.py
# This script is a "gateway" or "proxy" server. It runs a FastAPI application
# that receives custom API requests. It translates these requests into the
# standard OpenAI API format and forwards them to a separate vLLM backend server.

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

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Configuration Model ---
# Pydantic model for type-checking and validating 'server_config.yaml'
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
    vllm_host: str  # Host for the vLLM backend
    vllm_port: int  # Port for the vLLM backend

# --- Configuration ---
# Globals to store config and backend URL, populated at startup
VLLM_CONFIG: Dict[str, Any] = {}
VLLM_OPENAI_URL: str = ""

# --- Request Body Model ---
# Pydantic model for validating incoming API request bodies
class GenerateRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="List of message objects (e.g., {'role': 'user', 'content': ...})")
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
    http_client = httpx.AsyncClient(timeout=300.0) # Long timeout for LLMs
    logger.info("Gateway started. HTTPX client created.")
    
    yield  # The 'yield' separates startup (above) from shutdown (below) code
    
    # On Shutdown
    await http_client.aclose()  # Cleanly close the client connections
    logger.info("Gateway shutting down. HTTPX client closed.")

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
                        logger.error(f"Failed to decode image: {e}", exc_info=True)
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
# Create the FastAPI app instance and link the lifespan manager
app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def handle_generation(request: GenerateRequest):
    logger.debug(f"Received /generate request")

    try:
        # Standardize the message format (e.g., decode images)
        openai_messages = _parse_messages_for_openai(request.messages)
        # Add the "assistant" turn to prompt the model for a response
        if not openai_messages or openai_messages[-1]["role"] != "assistant":
            openai_messages.append({"role": "assistant", "content": None})
    except Exception as e:
        logger.error(f"Error processing input: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to process input messages: {e}")

    # Build the final JSON payload for the OpenAI-compatible backend
    openai_payload = {
        "model": VLLM_CONFIG['model_id'],
        "messages": openai_messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "seed": request.seed
    }

    # Call the backend and wait for the response
    result_text = await call_vllm_backend(openai_payload)
    return {"text": result_text}


@app.post("/edit")
async def handle_edit(request: GenerateRequest):
    """
    This endpoint is identical to /generate; it just provides a different
    path for semantic clarity (e.g., in logs or for future logic).
    """
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
    # Imports for config loading/logging (not part of the FastAPI app itself)
    from src.utils import setup_logging, load_config
    
    setup_logging(log_file='logs/vllm_gateway.log')

    try:
        # Load the raw YAML file
        raw_config_dict = load_config(config_path='config/server_config.yaml').get("prompt_polisher_service")
        if raw_config_dict is None:
            raise ValueError("'prompt_polisher_service' section not found")
        
        # Validate the config dict against the Pydantic model.
        # This will raise a ValidationError if keys are missing/wrong.
        VLLMConfig(**raw_config_dict)
        
        # Store the validated config in the global variable
        VLLM_CONFIG = raw_config_dict
        
        # Build the full URL for the backend API endpoint
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

    # This script is *only* a gateway; it doesn't load any models itself.
    logger.info(f"Starting FastAPI gateway on http://{VLLM_CONFIG['host']}:{VLLM_CONFIG['port']}")
    
    # Start the FastAPI server
    uvicorn.run(app, host=VLLM_CONFIG['host'], port=VLLM_CONFIG['port'])
