# diffuser_server.py
import torch
import base64
from io import BytesIO
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import AutoencoderKLQwenImage 
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
from PIL import Image

class DiffuserGenerator:
    def __init__(self, model_id="Qwen/Qwen-Image-Edit"):
        print("Initializing DiffuserGenerator and loading models...")
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        print(f"Using device: {self.device}")

        #--- Load and store all components as instance attributes ---
        img_quant_config = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.torch_dtype,
            llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
        )
        self.transformer = QwenImageTransformer2DModel.from_pretrained(
            self.model_id,
            subfolder="transformer",
            torch_dtype=self.torch_dtype,
            local_files_only=True,
            quantization_config=img_quant_config,
        ).to(self.device)

        text_quant_config = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.torch_dtype
        )
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            subfolder="text_encoder",
            torch_dtype=self.torch_dtype,
            local_files_only=True,
            quantization_config=text_quant_config,
        ).to(self.device)

        self.pipe = QwenImageEditPipeline.from_pretrained(
            self.model_id,
            transformer=self.transformer,
            text_encoder=self.text_encoder,
            torch_dtype=self.torch_dtype,
            local_files_only=True,
        ).to(self.device)

        self.generator = torch.Generator(device=self.device)
        print("Diffuser models loaded successfully.")
        

    def generate_image(self, initial_image_b64: Optional[str], prompt: str, negative_prompt: str, num_inference_steps: int, canvas_height: int) -> str:
        image = Image.new('RGB', (canvas_height, canvas_height), (240, 240, 240))
        if initial_image_b64:
            img_data = base64.b64decode(initial_image_b64)
            image = Image.open(BytesIO(img_data)).convert("RGB")
        
        inputs = {
            "image": image,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "generator": self.generator,
            "num_inference_steps": num_inference_steps
        }
        output_image = self.pipe(**inputs).images[0]
        
        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

app = FastAPI()
generator = DiffuserGenerator()

class GenerationRequest(BaseModel):
    prompt: str
    initial_image_b64: Optional[str] = None
    negative_prompt: Optional[str] = ""
    num_inference_steps: int = 20
    canvas_height: int = 1024

@app.post("/generate_diff")
def generate_diff(request: GenerationRequest):
    img_b64 = generator.generate_image(
        request.initial_image_b64, request.prompt, request.negative_prompt,
        request.num_inference_steps, request.canvas_height
    )
    return {"image_b64": img_b64}
