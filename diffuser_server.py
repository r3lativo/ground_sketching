# diffuser_server.py
import torch
import base64
import yaml
import argparse
import importlib
import re
from io import BytesIO
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, List
from jinja2 import Environment, FileSystemLoader
from transformers import AutoTokenizer
from PIL import Image
from vllm import LLM, SamplingParams

# --- Helper Functions ---
def _get_class_from_string(class_path: str):
    """Dynamically imports a class from a string path (e.g., "diffusers.QwenImageEditPlusPipeline")."""
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def _extract_answer(raw_output: str) -> str:
    """A simple parser to extract the core generated text, ignoring boilerplate."""
    # Try to find content within an <answer> tag first.
    match = re.search(r"<answer>(.*?)</answer>", raw_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    # As a fallback, find the text following "Scene Description:".
    parts = re.split(r"Scene Description:", raw_output, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[-1].strip()
    return raw_output


class TextRefiner:
    """Manages a text generation model using vLLM for high-performance prompt refinement."""
    def __init__(self, model_id: str, device: str, dtype: torch.dtype):
        print(f"Initializing vLLM-based TextRefiner with model: {model_id}")

        # Determine number of GPUs for tensor parallelism.
        num_gpus = torch.cuda.device_count() if device == "cuda" else 0
        
        # Convert torch dtype to the string format vLLM expects.
        dtype_str = "auto"
        if dtype == torch.bfloat16:
            dtype_str = "bfloat16"
        elif dtype == torch.float16:
            dtype_str = "half"
        
        # Initialize the vLLM engine. `gpu_memory_utilization` limits VRAM usage,
        # which is crucial for running another large model (the diffuser) on the same GPU.
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=num_gpus if num_gpus > 0 else 1,
            dtype=dtype_str,
            gpu_memory_utilization=0.1
        )
        print(f"vLLM engine loaded on {num_gpus if num_gpus > 0 else 1} GPU(s).")

        # The tokenizer is still needed to correctly format the chat prompt.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

        # Pre-define sampling parameters for deterministic and controlled generation.
        self.sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.95,
            max_tokens=300,
            stop=["<|im_end|>"] # Model-specific stop token to end generation cleanly.
        )

        # Jinja2 environment to load and render prompt templates from files.
        self.jinja_env = Environment(loader=FileSystemLoader('configs/refiners'))
        print("TextRefiner initialized successfully.")

    def refine(self, dialogue_lines: List[str], template_name: str) -> str:
        """Generates a refined prompt from dialogue lines using the vLLM engine."""
        try:
            # Render the full prompt from a Jinja2 template.
            template = self.jinja_env.get_template(f"{template_name}.jinja2")
            prompt_content = template.render(dialogue_lines=dialogue_lines)
            
            # Apply the chat template to format the prompt correctly for the model.
            messages = [{"role": "user", "content": prompt_content}]
            final_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Run inference using the vLLM engine.
            outputs = self.llm.generate([final_prompt], self.sampling_params)
            
            # Extract and clean up the generated text.
            raw_response = outputs[0].outputs[0].text
            return _extract_answer(raw_response)

        except Exception as e:
            print(f"Error during vLLM text refinement: {e}")
            # Fallback to simple concatenation if refinement fails.
            return ' '.join(dialogue_lines)


class DiffuserGenerator:
    """Manages the image generation model and pipeline."""
    def __init__(self, config: Dict, refiner: Optional[TextRefiner] = None):
        self.config = config
        print(f"Initializing DiffuserGenerator with preset: {self.config.get('preset_name', 'N/A')}")
        
        model_id = self.config['model_id']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        print(f"Using device: {self.device} with dtype: {self.torch_dtype}")
        
        # Dynamically load and initialize all components of the diffusion model from config.
        transformer_class = _get_class_from_string(self.config['transformer_class'])
        self.transformer = transformer_class.from_pretrained(model_id, subfolder='transformer', torch_dtype=self.torch_dtype, local_files_only=True).to(self.device)
        print(f"Loaded Transformer: {self.config['transformer_class']}")

        text_encoder_class = _get_class_from_string(self.config['text_encoder_class'])
        self.text_encoder = text_encoder_class.from_pretrained(model_id, subfolder='text_encoder', torch_dtype=self.torch_dtype, local_files_only=True).to(self.device)
        print(f"Loaded Text Encoder: {self.config['text_encoder_class']}")

        pipeline_class = _get_class_from_string(self.config['pipeline_class'])
        self.pipe = pipeline_class.from_pretrained(model_id, transformer=self.transformer, text_encoder=self.text_encoder, torch_dtype=self.torch_dtype, local_files_only=True).to(self.device)
        print(f"Loaded Pipeline: {self.config['pipeline_class']}")

        # Store the refiner instance to be used in the generation process.
        self.refiner = refiner
        print("Diffuser models loaded successfully.")

    def generate_image(self, dialogue_lines: List[str], refine_prompt: bool, refiner_template: str, initial_image_b64: Optional[str], params: dict):
        """Generates a new image based on text and an optional initial image."""
        # Step 1: Get the final prompt, either by refining the dialogue or using it raw.
        if refine_prompt and self.refiner:
            print(f"Refining prompt using template '{refiner_template}'...")
            prompt = self.refiner.refine(dialogue_lines, refiner_template)
            print(f"Refined Prompt: {prompt}")
        else:
            prompt = ' '.join(dialogue_lines)
            print(f"Using Raw Prompt: {prompt}")

        # Step 2: Prepare the input image canvas.
        canvas_size = params.get("canvas_size")
        image = Image.new('RGB', (canvas_size, canvas_size), (255, 255, 255))
        # If an initial image is provided, decode it from Base64.
        if initial_image_b64:
            try:
                img_data = base64.b64decode(initial_image_b64)
                image = Image.open(BytesIO(img_data)).convert("RGB")
            except Exception as e:
                print(f"Warning: Could not decode initial_image_b64. Error: {e}")

        # For reproducibility, set a random seed.
        seed = torch.manual_seed(params.get("seed", 0))

        # Step 3: Assemble all inputs for the diffusion pipeline.
        inputs = {
            "image": image,
            "prompt": prompt,
            "negative_prompt": params.get("negative_prompt"),
            "generator": seed,
            "true_cfg_scale": params.get("true_cfg_scale"),
            "guidance_scale": params.get("guidance_scale"),
            "num_inference_steps": params.get("num_inference_steps")
        }

        # Step 4: Run the pipeline and encode the output image to Base64.
        output_image = self.pipe(**inputs).images[0]
        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        final_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Return both the prompt used and the generated image.
        return prompt, final_img

# --- FastAPI App Setup ---
app = FastAPI()

# Pydantic model defines the expected structure of the JSON request body.
class GenerationRequest(BaseModel):
    dialogue_lines: List[str]
    refine_prompt: bool
    refiner_template_name: str
    initial_image_b64: Optional[str] = None
    parameters: Dict

# Global variable to hold the initialized generator instance.
generator: Optional[DiffuserGenerator] = None

@app.post("/generate")
def generate(request: GenerationRequest):
    """The main API endpoint for generating images."""
    if not generator:
        return {"error": "Generator not initialized"}, 503
        
    prompt, img_b64 = generator.generate_image(
        dialogue_lines=request.dialogue_lines,
        refine_prompt=request.refine_prompt,
        refiner_template=request.refiner_template_name,
        initial_image_b64=request.initial_image_b64,
        params=request.parameters
    )
    # Return a JSON object containing the final prompt and the image data.
    return {"prompt": prompt, "image_b64": img_b64}


if __name__ == "__main__":
    import uvicorn

    # Set up command-line arguments to specify config files at runtime.
    parser = argparse.ArgumentParser(description="Configurable Diffuser Model Server")
    parser.add_argument("--server-config", type=str, default="configs/server_config.yaml")
    parser.add_argument("--model-config", type=str, required=True)
    args = parser.parse_args()

    # Load server and model configurations from YAML files.
    with open(args.server_config, 'r') as f:
        server_config = yaml.safe_load(f)
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)

    # Initialize the text refiner model.
    text_refiner = TextRefiner(
        model_id=server_config['refiner_model_id'],
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Initialize the main image generator, passing the refiner to it.
    generator = DiffuserGenerator(config=model_config, refiner=text_refiner)

    # Start the FastAPI server using uvicorn.
    uvicorn.run(app, host=server_config['host'], port=server_config['port'])
