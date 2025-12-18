# src/image_gen_app.py

import math
import logging
import time
import sys
import asyncio
import functools 
from contextlib import asynccontextmanager
from typing import Optional, List

import torch
import yaml
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPipeline,
    QwenImageEditPlusPipeline,
)
from diffusers.models import QwenImageTransformer2DModel

from .utils import pil_to_base64, base64_to_pil, load_config

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Configuration Models ---
class ImageGenConfig(BaseModel):
    host: str
    port: int
    model_id: str
    lora_path: Optional[str] = None

class ImageEditRequest(BaseModel):
    prompt: str
    images: List[str]
    seed: int = 0
    true_cfg_scale: float
    negative_prompt: str
    num_inference_steps: int

class ImageEditResponse(BaseModel):
    image: str

# --- Global State Variables ---
pipeline = None
DEVICE: Optional[str] = None
DTYPE: Optional[torch.dtype] = None
config: Optional[ImageGenConfig] = None

# --- Use asyncio.Lock instead of threading.Lock ---
# This allows the server to "await" the lock without blocking the network loop
generation_queue_lock = asyncio.Lock()

# --- Config Loading ---
try:
    raw_full_config = load_config('config/server_config.yaml')
    raw_service_config = raw_full_config.get('image_gen_service')
    if raw_service_config is None:
        raise ValueError("'image_gen_service' section not found")
    config = ImageGenConfig(**raw_service_config)
except Exception as e:
    logger.critical(f"[IMG_GEN] FATAL: Failed to load config: {e}", exc_info=True)
    sys.exit("Failed to load config.")

# --- Model Loading Logic ---
def load_model():
    """Loads the diffusion pipeline and sets global DEVICE and DTYPE."""
    global pipeline, DEVICE, DTYPE
    if torch.cuda.is_available():
        DEVICE = "cuda"
        DTYPE = torch.bfloat16
    else:
        DEVICE = "cpu"
        DTYPE = torch.float32

    if "2509" in config.model_id:
        pipe_cls = QwenImageEditPlusPipeline
    else:
        pipe_cls = QwenImageEditPipeline

    try:
        if config.lora_path is not None:
            model = QwenImageTransformer2DModel.from_pretrained(
                config.model_id, subfolder="transformer", torch_dtype=DTYPE, local_files_only=True
            )
            scheduler = FlowMatchEulerDiscreteScheduler.from_config({
                "base_image_seq_len": 256,
                "base_shift": math.log(3),
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": math.log(3),
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": None,
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            })
            _pipe = pipe_cls.from_pretrained(
                config.model_id, transformer=model, scheduler=scheduler, torch_dtype=DTYPE, local_files_only=True
            )

            logger.info(f"[IMG_GEN] Loading LoRA weights: {config.lora_path}...")
            _pipe.load_lora_weights(
                config.lora_path
            )
        else:
            logger.info(f"[IMG_GEN] Loading base pipeline (no LoRA): {config.model_id}...")
            _pipe = pipe_cls.from_pretrained(
                config.model_id,
                torch_dtype=DTYPE,
                local_files_only=True
            )
        
        pipeline = _pipe.to(DEVICE)
        logger.info(f"[IMG_GEN] Pipeline loaded on {DEVICE}.")
    except Exception as e:
        logger.error(f"[IMG_GEN] FATAL: Failed to load pipeline: {e}", exc_info=True)
        raise

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    global pipeline
    pipeline = None
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

# --- Helper for Thread Offloading ---
def run_inference_sync(inputs):
    """
    This wrapper runs the blocking inference.
    It will be called in a separate thread.
    """
    with torch.inference_mode():
        if DEVICE == "cuda":
            with torch.amp.autocast('cuda', dtype=DTYPE if DTYPE in [torch.bfloat16, torch.float16] else None):
                output = pipeline(**inputs)
        else:
            output = pipeline(**inputs)
    return output.images[0]

# --- API Endpoints ---
@app.post("/img_generate", response_model=ImageEditResponse)
async def generate_image(request: ImageEditRequest):
    if pipeline is None:
         raise HTTPException(status_code=503, detail="Pipeline not loaded.")

    start_time = time.time()
    
    # Decode images (CPU bound, but fast enough to keep here or move if needed)
    try:
        input_images = [base64_to_pil(img_b64) for img_b64 in request.images]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    # Prepare inputs for the pipeline
    inputs = {
        "image": input_images,
        "prompt": request.prompt,
        "generator": torch.Generator(device=DEVICE).manual_seed(request.seed),
        "true_cfg_scale": request.true_cfg_scale,
        "negative_prompt": request.negative_prompt,
        "num_inference_steps": request.num_inference_steps,
    }

    # --- Queue Logic ---
    # 1. Await the lock. This pauses the function if the GPU is busy,
    #    but lets the server keep running to accept other connections.
    async with generation_queue_lock:
        logger.info(f"[IMG_GEN] Processing request (Queue cleared). Prompt: '{request.prompt}'")
        
        try:
            # 2. Offload blocking to a thread.
            # This ensures the event loop is completely free while the GPU crunches.
            loop = asyncio.get_running_loop()
            output_image = await loop.run_in_executor(
                None, # None means use the default ThreadPoolExecutor
                functools.partial(run_inference_sync, inputs)
            )
        except Exception as e:
            logger.error(f"[IMG_GEN] Inference failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    inference_time = time.time() - start_time
    logger.info(f"[IMG_GEN] Done in {inference_time:.2f}s")

    return ImageEditResponse(image=pil_to_base64(output_image))


@app.get("/health")
async def health_check():
    is_ready = pipeline is not None
    return {"status": "ready" if is_ready else "loading_error"}, 200 if is_ready else 503

# --- Main execution: Start the server ---
if __name__ == "__main__":
    import uvicorn
    from .utils import setup_logging
    setup_logging(log_file='logs/image_gen_server.log')
    
    if config:
        uvicorn.run(app, host=config.host, port=config.port, log_config=None)
