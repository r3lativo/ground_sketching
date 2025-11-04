# src/image_gen_app.py

import math
import os
import logging
import time
import sys
from contextlib import asynccontextmanager
from typing import Optional, List

import torch
import yaml
from pydantic import BaseModel, ValidationError, Field
from fastapi import FastAPI, HTTPException
from diffusers import (
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPipeline,
    QwenImageEditPlusPipeline,
)
from diffusers.models import QwenImageTransformer2DModel
import threading

# Assuming utils.py and image_gen.py are accessible via src path
from src.utils import pil_to_base64, base64_to_pil, load_config

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Configuration Model ---
class ImageGenConfig(BaseModel):
    model_id: str
    lora_path: Optional[str] = None

# --- Edit Request Model  ---
class ImageEditRequest(BaseModel):
    prompt: str = Field(..., description="The editing instruction for the model.")
    images: List[str] = Field(..., min_items=1, max_items=3, description="List of Base64 encoded input images (1 to 3 images).")
    seed: int = 0
    true_cfg_scale: float
    negative_prompt: str
    num_inference_steps: int

# --- Edit Response Model  ---
class ImageEditResponse(BaseModel):
    image: str = Field(..., description="Base64 encoded output image (PNG format).")
    # Optional: You could also return the seed used, execution time, etc.
    # seed_used: int
    # processing_time_ms: float

# --- Global State Variables ---
pipeline = None
generation_lock = threading.Lock()
DEVICE: Optional[str] = None
DTYPE: Optional[torch.dtype] = None
config: Optional[ImageGenConfig] = None

# --- Load and Validate Config at Startup ---
try:
    raw_full_config = load_config('config/server_config.yaml')
    raw_service_config = raw_full_config.get('image_gen_service')
    
    if raw_service_config is None:
        raise ValueError("'image_gen_service' section not found in config/server_config.yaml")
        
    # Validate the config
    config = ImageGenConfig(**raw_service_config)
    
    logger.info("Image Gen Config loaded and validated successfully:")
    logger.info(f"{config.model_dump_json()}")

except (ValidationError, ValueError) as e:
    logger.critical(f"--- CONFIGURATION ERROR ---")
    logger.critical(f"FATAL: Missing or invalid keys in 'image_gen_service' section of config/server_config.yaml:")
    logger.critical(f"\n{e}")
    sys.exit("Invalid configuration. Please check the log.")
except Exception as e:
    logger.critical(f"FATAL: Failed to load config: {e}", exc_info=True)
    sys.exit("Failed to load config.")


# --- Model Loading Logic ---
def load_model():
    """Loads the diffusion pipeline and sets global DEVICE and DTYPE."""
    global pipeline, DEVICE, DTYPE

    if config is None:
        logger.error("FATAL: Config not loaded. Cannot load model.")
        return

    # Determine device and dtype
    if torch.cuda.is_available():
        DEVICE = "cuda"
        DTYPE = torch.bfloat16
        logger.info(f"CUDA available. Using device: {DEVICE} with dtype: {DTYPE}")
    else:
        DEVICE = "cpu"
        DTYPE = torch.float32
        logger.warning(f"CUDA not available. Using device: {DEVICE}. This will be VERY slow.")

    if "2509" in config.model_id:
        pipe_cls = QwenImageEditPlusPipeline
    else:
        pipe_cls = QwenImageEditPipeline

    try:
        if config.lora_path is not None:
            model = QwenImageTransformer2DModel.from_pretrained(
                config.model_id, subfolder="transformer", torch_dtype=DTYPE, local_files_only=True
            )
            assert os.path.exists(config.lora_path), f"Lora path {config.lora_path} does not exist"
            scheduler_config = {
                "base_image_seq_len": 256,
                "base_shift": math.log(3),  # We use shift=3 in distillation
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": math.log(3),  # We use shift=3 in distillation
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": None,  # set shift_terminal to None
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            }
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
            
            logger.info(f"Loading base pipeline: {config.model_id}...")
            _pipe = pipe_cls.from_pretrained(
                config.model_id,
                transformer=model,
                scheduler=scheduler,
                torch_dtype=DTYPE,
                local_files_only=True
            )

            logger.info(f"Loading LoRA weights: {config.lora_path}...")
            _pipe.load_lora_weights(
                config.lora_path
            )
        else:
            logger.info(f"Loading base pipeline (no LoRA): {config.model_id}...")
            _pipe = pipe_cls.from_pretrained(
                config.model_id,
                torch_dtype=DTYPE,
                local_files_only=True
            )
        
        logger.info(f"Moving pipeline to device: {DEVICE}...")
        pipeline = _pipe.to(DEVICE)
        logger.info("Image generation pipeline loaded successfully.")

    except Exception as e:
        logger.error(f"FATAL: Failed to load the diffusion pipeline: {e}", exc_info=True)
        pipeline = None # Ensure it's None if loading fails
        raise

# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model during startup
    logger.info("Application startup: Loading model...")
    load_model()
    yield
    # Clean up resources if needed during shutdown
    logger.info("Application shutdown.")
    global pipeline
    pipeline = None
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# --- FastAPI App Instance ---
app = FastAPI(lifespan=lifespan)

# --- API Endpoints ---
@app.post("/img_generate", response_model=ImageEditResponse)
async def generate_image(request: ImageEditRequest):
    """
    Generates an edited image based on input images and a prompt.
    """
    if pipeline is None or DEVICE is None or DTYPE is None:
         logger.error("Image generation request failed: Pipeline not loaded.")
         raise HTTPException(status_code=503, detail="Pipeline not loaded. Service unavailable.")

    start_time = time.time()

    logger.info(f"Received request - Prompt: '{request.prompt[:50]}...', Images: {len(request.images)}")

    try:
        input_images = [base64_to_pil(img_b64) for img_b64 in request.images]
    except ValueError as e:
        logger.warning(f"Bad request: Invalid base64 image data. Error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")
    except Exception as e:
        logger.error(f"Unexpected error decoding images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during image decoding.")

    # Prepare inputs for the pipeline
    inputs = {
        "image": input_images,
        "prompt": request.prompt,
        "generator": torch.Generator(device=DEVICE).manual_seed(request.seed),
        "true_cfg_scale": request.true_cfg_scale,
        "negative_prompt": request.negative_prompt,
        "num_inference_steps": request.num_inference_steps,
    }
    logger.debug(f"Pipeline inputs: {inputs}")

    logger.debug("Running pipeline inference...")
    try:
        with generation_lock:
            with torch.inference_mode():
                if DEVICE == "cuda":
                    # with torch.cuda.amp.autocast(torch_dtype=DTYPE if DTYPE in [torch.bfloat16, torch.float16] else None):
                    with torch.amp.autocast('cuda', dtype=DTYPE if DTYPE in [torch.bfloat16, torch.float16] else None):
                        output = pipeline(**inputs)
                else: # CPU
                    output = pipeline(**inputs)

        output_image = output.images[0]
        inference_time = time.time() - start_time
        logger.info(f"Pipeline inference successful. Time taken: {inference_time:.2f} seconds.")

    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image generation failed internally: {e}")

    try:
        output_base64 = pil_to_base64(output_image)
        logger.debug("Output image successfully encoded to Base64.")
    except Exception as e:
        logger.error(f"Error encoding output image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to encode output image.")

    return ImageEditResponse(image=output_base64)

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    is_ready = pipeline is not None
    status_code = 200 if is_ready else 503
    return {"status": "ready" if is_ready else "loading_error", "pipeline_loaded": is_ready}, status_code
