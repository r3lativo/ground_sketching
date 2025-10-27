# src/api_models.py

from pydantic import BaseModel, Field
from typing import List

class ImageEditRequest(BaseModel):
    prompt: str = Field(..., description="The editing instruction for the model.")
    images: List[str] = Field(..., min_items=1, max_items=3, description="List of Base64 encoded input images (1 to 3 images).")
    seed: int = 0
    true_cfg_scale: float
    negative_prompt: str
    num_inference_steps: int

class ImageEditResponse(BaseModel):
    image: str = Field(..., description="Base64 encoded output image (PNG format).")
    # Optional: You could also return the seed used, execution time, etc.
    # seed_used: int
    # processing_time_ms: float