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
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from vllm import LLM, SamplingParams

# --- Helper Functions ---
def _get_class_from_string(class_path: str):
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def _extract_answer(raw_output: str) -> str:
    """Extracts content from a simple <answer> tag if present."""
    match = re.search(r"<answer>(.*?)</answer>", raw_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: if no tag, return the text after the prompt
    parts = re.split(r"Scene Description:", raw_output, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[-1].strip()
    return raw_output


class TextRefiner:
    def __init__(self, model_id: str, device: str, dtype: torch.dtype):
        """
        Initializes the vLLM engine and tokenizer.

        Args:
            model_id (str): The Hugging Face model identifier.
            device (str): The primary device ("cuda" or "cpu"). vLLM primarily targets GPUs.
            dtype (torch.dtype): The torch dtype for model weights.
        """
        print(f"Initializing vLLM-based TextRefiner with model: {model_id}")

        # vLLM manages devices automatically. We can determine tensor parallel size based on GPU count.
        num_gpus = torch.cuda.device_count() if device == "cuda" else 0
        
        # vLLM's LLM class expects dtype as a string.
        dtype_str = "auto"
        if dtype == torch.bfloat16:
            dtype_str = "bfloat16"
        elif dtype == torch.float16:
            dtype_str = "half"
        
        # Initialize the vLLM engine.
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=num_gpus if num_gpus > 0 else 1,
            dtype=dtype_str,
            gpu_memory_utilization=0.1
        )
        print(f"vLLM engine loaded on {num_gpus if num_gpus > 0 else 1} GPU(s).")

        # We still need the tokenizer to apply the chat template before sending the raw string to vLLM.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

        # Define the sampling parameters once. These control the generation process.
        self.sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.95,
            max_tokens=300,
            stop=["<|im_end|>"] # Stop generation when the model emits its end token
        )

        # The Jinja2 environment for loading prompt templates remains the same.
        self.jinja_env = Environment(loader=FileSystemLoader('configs/refiners'))
        print("TextRefiner initialized successfully.")

    def refine(self, dialogue_lines: List[str], template_name: str) -> str:
        """
        Generates a refined prompt from dialogue lines using the vLLM engine.
        """
        try:
            # 1. Render the prompt using Jinja2 template
            template = self.jinja_env.get_template(f"{template_name}.jinja2")
            prompt_content = template.render(dialogue_lines=dialogue_lines)
            
            # 2. Apply the model's chat template to format the final prompt string
            messages = [{"role": "user", "content": prompt_content}]
            final_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 3. Generate text using the vLLM engine
            # vLLM's generate method takes a list of prompts and sampling parameters.
            outputs = self.llm.generate([final_prompt], self.sampling_params)
            
            # 4. Extract the generated text from the first output
            raw_response = outputs[0].outputs[0].text
            
            # 5. Clean up the response
            return _extract_answer(raw_response)

        except Exception as e:
            print(f"Error during vLLM text refinement: {e}")
            return ' '.join(dialogue_lines)


class DiffuserGenerator:
    def __init__(self, config: Dict, refiner: Optional[TextRefiner] = None):
        self.config = config
        print(f"Initializing DiffuserGenerator with preset: {self.config.get('preset_name', 'N/A')}")
        
        model_id = self.config['model_id']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        print(f"Using device: {self.device} with dtype: {self.torch_dtype}")
        
        transformer_class = _get_class_from_string(self.config['transformer_class'])
        self.transformer = transformer_class.from_pretrained(
            model_id,
            subfolder=self.config['transformer_subfolder'],
            torch_dtype=self.torch_dtype,
            local_files_only=True
        ).to(self.device)
        print(f"Loaded Transformer: {self.config['transformer_class']}")

        text_encoder_class = _get_class_from_string(self.config['text_encoder_class'])
        self.text_encoder = text_encoder_class.from_pretrained(
            model_id,
            subfolder=self.config['text_encoder_subfolder'],
            torch_dtype=self.torch_dtype,
            local_files_only=True
        ).to(self.device)
        print(f"Loaded Text Encoder: {self.config['text_encoder_class']}")

        pipeline_class = _get_class_from_string(self.config['pipeline_class'])
        self.pipe = pipeline_class.from_pretrained(
            model_id,
            transformer=self.transformer,
            text_encoder=self.text_encoder,
            torch_dtype=self.torch_dtype,
            local_files_only=True
        ).to(self.device)
        print(f"Loaded Pipeline: {self.config['pipeline_class']}")

        self.refiner = refiner
        print("Diffuser models loaded successfully.")

    def generate_image(self, dialogue_lines: List[str], refine_prompt: bool, refiner_template: str, initial_image_b64: Optional[str], params: dict) -> str:
        """
        Generates an image with an overhauled logic supporting refinement.
        """
        # Step 1: Determine the final prompt
        if refine_prompt and self.refiner:
            print(f"Refining prompt using template '{refiner_template}'...")
            prompt = self.refiner.refine(dialogue_lines, refiner_template)
            print(f"Refined Prompt: {prompt}")
        else:
            prompt = ' '.join(dialogue_lines)
            print(f"Using Raw Prompt: {prompt}")

        # Step 2: Prepare the canvas
        canvas_size = params.get("canvas_size")
        image = Image.new('RGB', (canvas_size, canvas_size), (255, 255, 255))
        if initial_image_b64:
            try:
                img_data = base64.b64decode(initial_image_b64)
                image = Image.open(BytesIO(img_data)).convert("RGB")
            except Exception as e:
                print(f"Warning: Could not decode initial_image_b64. Using blank canvas. Error: {e}")

        seed = torch.manual_seed(params.get("seed", 0))

        # Step 3: Prepare pipeline inputs
        inputs = {
            "image": image,
            "prompt": prompt,
            "negative_prompt": params.get("negative_prompt"),
            "generator": seed,
            "true_cfg_scale": params.get("true_cfg_scale"),
            "guidance_scale": params.get("guidance_scale"),
            "num_inference_steps": params.get("num_inference_steps")
        }

        # Step 4: Run pipeline and encode result
        output_image = self.pipe(**inputs).images[0]
        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        final_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return prompt, final_img

# --- FastAPI App Setup ---
app = FastAPI()

# Pydantic model for the /generate endpoint
class GenerationRequest(BaseModel):
    dialogue_lines: List[str]
    refine_prompt: bool = False
    refiner_template_name: str = "default"
    initial_image_b64: Optional[str] = None
    parameters: Dict # Holds params like num_inference_steps, negative_prompt, etc.

generator: Optional[DiffuserGenerator] = None

@app.post("/generate")
def generate(request: GenerationRequest):
    if not generator:
        return {"error": "Generator not initialized"}, 503
        
    prompt, img_b64 = generator.generate_image(
        dialogue_lines=request.dialogue_lines,
        refine_prompt=request.refine_prompt,
        refiner_template=request.refiner_template_name,
        initial_image_b64=request.initial_image_b64,
        params=request.parameters
    )
    return {"prompt": prompt, "image_b64": img_b64}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Configurable Diffuser Model Server")
    parser.add_argument("--server-config", type=str, default="configs/server_config.yaml", help="Path to the server's YAML configuration file.")
    parser.add_argument("--model-config", type=str, required=True, help="Path to the model's YAML configuration file.")
    args = parser.parse_args()

    with open(args.server_config, 'r') as f:
        server_config = yaml.safe_load(f)
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)

    # Initialize the text refiner
    text_refiner = TextRefiner(
        model_id=server_config['refiner_model_id'],
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Initialize the generator, now with the refiner
    generator = DiffuserGenerator(config=model_config, refiner=text_refiner)

    uvicorn.run(app, host=server_config['host'], port=server_config['port'])
