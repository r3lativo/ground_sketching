# config_models.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, List

class GenerationParameters(BaseModel):
    """Defines all parameters related to the image generation process."""
    # Field(...) allows for descriptions and validation rules.
    initial_num_steps: int = Field(50, description="Steps for the first image in a sequence.")
    num_inference_steps: int = Field(30, description="Steps for subsequent images.")
    canvas_size: int = 512
    seed: int = 0
    negative_prompt: str = "text, low-quality"  # blurry
    
    # These are specific to certain pipelines, so they are optional.
    true_cfg_scale: Optional[float] = 4.0
    guidance_scale: Optional[float] = None

    # This allows for any other parameters to be passed through.
    class Config:
        extra = 'allow'


class RefinerParameters(BaseModel):
    """Defines all parameters for the TextRefiner's sampling process."""
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 500
    # The seed will be inherited from the main GenerationParameters, so it's not needed here.

    # This allows for any other parameters to be passed through.
    class Config:
        extra = 'allow'


class ExperimentConfig(BaseModel):
    """The main configuration model for an experiment run."""
    experiment_name: str
    description: str
    
    # Data Source Settings
    csv_file_path: str
    chunk_id: str
    character: str
    
    # Generation Logic
    mode: str = Field(..., pattern="^(chunk|utterance)$") # Must be 'chunk' or 'utterance'
    
    # Prompting Settings
    refine_prompt: bool = False
    refiner_template_name: str = "default"  # The name of the .jinja2 file in 'refiners/'
    
    # Server and Model Settings
    api_url: str = "http://localhost:8000/generate"
    parameters: GenerationParameters = Field(default_factory=GenerationParameters)
    refiner_parameters: RefinerParameters = Field(default_factory=RefinerParameters)
    
    # Output Settings
    output_dir: str = "output/"
    log_file_path: str = "log.csv"
