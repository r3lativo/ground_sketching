# sgp_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from vllm import LLM, SamplingParams

# --- SGP Model Configuration ---
SGP_MODEL_PATH = "/linkhome/rech/genxzs01/upa38qy/.cache/huggingface/hub/models--SphereLab--SGP-RL/snapshots/b71e5dec1971b4d6bd0efac73f19cc324cc30bae"
SYSTEM_PROMPT = "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."

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

def format_prompt(tokenizer, turns: List[Dict[str, str]]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(turns)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

app = FastAPI()
llm, tokenizer = load_sgp_model(SGP_MODEL_PATH)

class SvgRequest(BaseModel):
    prompt: str

@app.post("/generate_svg")
def generate_svg(request: SvgRequest):
    turns = [{"role": "user", "content": request.prompt}]
    prompt_text = format_prompt(tokenizer, turns)
    
    sampler = SamplingParams(
        n=1,
        stop=["</answer>"],
        max_tokens=3000,
        temperature=1.0,
        include_stop_str_in_output=True,
    )
    outputs = llm.generate([prompt_text], sampler)
    svg_text = outputs[0].outputs[0].text
    
    return {"svg_text": svg_text}
