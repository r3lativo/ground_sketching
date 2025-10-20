# Ground Sketching

Supports two distinct types of visual generation:

- **Diffuser**: Creates pixel-based raster images (PNGs) using the Qwen-Image-Edit model.

- **SGP**: Creates structured vector graphics (SVGs) using an instruction-tuned LLM, which are then rendered to PNGs.

Multiple Generation Modes: `chunk` or  `utterance`

Advanced Prompting:

- `raw`: Uses dialogue directly to form prompts.

- `refined`: Uses an LLM to first refine a clean, descriptive paragraph from the raw dialogue, which is then used for image generation.

Experimental parameters manageable from `config.yaml`. Logging setup. Client-server architecture to separate the experimental logic from the model inference.


## Setup

```bash
module load pytorch-gpu/py3/2.8.0
pip install cairosvg
pip install git+https://github.com/huggingface/diffusers
```


## Usage

You must start the two model servers in separate terminal windows.

Terminal 1: Start the Diffuser Server
```bash
python diffuser_server.py --model-config configs/models/qwen_plus.yaml
```

Once the servers are running, open another terminal to run an experiment using main.py.

```bash
python main.py --config configs/experiments/povA_chunk_refined.yaml
```
