# Ground Sketching

- **Diffuser**: Creates pixel-based raster images (PNGs) using the Qwen-Image-Edit model.

Multiple Generation Modes: `chunk` or  `utterance`

Advanced Prompting:

- `raw`: Uses dialogue directly to form prompts.
- `refined`: Uses an LLM to first refine a clean, descriptive paragraph from the raw dialogue, which is then used for image generation.

Experimental parameters manageable from the yaml file inside `configs/experiments/`. Logging setup. Client-server architecture to separate the experimental logic from the model inference.


## Setup

```bash
module load pytorch-gpu/py3/2.8.0
pip install git+https://github.com/huggingface/diffusers
```


## Usage

You must start the two model servers in separate terminal windows.

Terminal 1: Start the Diffuser Server
```bash
python diffuser_server.py --model-config configs/models/qwen_plus.yaml
```

Terminal 2: Once the server is running, open another terminal to run an experiment.

```bash
python main.py --config configs/experiments/nathan_u_raw.yaml
```

Use `--force` to force generation even if file already exists.

```bash
python main.py --config configs/experiments/nathan_u_raw.yaml --force
```