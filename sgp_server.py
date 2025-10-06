# sgp_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from vllm import LLM, SamplingParams

# --- SGP Model Configuration ---
SGP_MODEL_PATH = "/linkhome/rech/genxzs01/upa38qy/.cache/huggingface/hub/models--SphereLab--SGP-RL/snapshots/b71e5dec1971b4d6bd0efac73f19cc324cc30bae"
SVG_SYSTEM_PROMPT = "You are an expert SVG designer. Create SVG code based on the user's request. First think about the reasoning process in the mind and then provides the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
REFINE_SYSTEM_PROMPT = "You are an expert scene descriptor. Follow the user's instructions precisely. First think about the reasoning process in the mind and then provides the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."

def load_sgp_model(model_path: str):
    print("Initializing SGP model with vLLM...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.40,
        dtype="bfloat16",
        enforce_eager=False,  # generally faster with CUDA graphs when False
    )
    tokenizer = llm.get_tokenizer()
    print("SGP model loaded successfully.")
    return llm, tokenizer

def format_prompt(tokenizer, system_prompt: str, turns: List[Dict[str, str]]) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(turns)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

app = FastAPI()
llm, tokenizer = load_sgp_model(SGP_MODEL_PATH)

# --- Pydantic Models for Requests ---
class SvgRequest(BaseModel):
    prompt: str

class RefiningRequest(BaseModel):
    dialogue_lines: List[str]
    prompt_template: str

# --- API Endpoints ---
@app.post("/generate_svg")
def generate_svg(request: SvgRequest):
    turns = [{"role": "user", "content": request.prompt}]
    prompt_text = format_prompt(tokenizer, SVG_SYSTEM_PROMPT, turns)
    sampler = SamplingParams(n=1, stop=["</answer>", "</svg>"], max_tokens=4096, temperature=1, include_stop_str_in_output=True)
    outputs = llm.generate([prompt_text], sampler)
    svg_text = outputs[0].outputs[0].text
    return {"svg_text": svg_text}

@app.post("/refine_description")
def refine(request: RefiningRequest):
    dialogue_str = "\n".join(request.dialogue_lines)
    prompt = request.prompt_template.format(dialogue_lines=dialogue_str)
    turns = [{"role": "user", "content": prompt}]
    prompt_text = format_prompt(tokenizer, REFINE_SYSTEM_PROMPT, turns)
    sampler = SamplingParams(n=1, stop=["</answer>"], max_tokens=1024, temperature=0.65)
    outputs = llm.generate([prompt_text], sampler)
    description = outputs[0].outputs[0].text.strip()
    return {"description": description}
