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
```


## Usage

You must start the two model servers in separate terminal windows.

Terminal 1: Start the Diffuser Server
```bash
uvicorn diffuser_server:app --port 8000
```

Terminal 2: Start the SGP Server
```bash
uvicorn sgp_server:app --port 8001
```

Once the servers are running, open a third terminal to run an experiment using main.py.

The script is controlled by command-line arguments:
- `--generator`: `diffuser` or `sgp`
- `--mode`: `chunk` or `utterance`
- `--prompt_mode`: `raw` or `refined`
- `--chunk_id`: The ID from `conversation.csv` to process.
- `--character`: The character's perspective to use.

Example Commands:

- Generate an image using the Diffuser from a whole chunk of raw dialogue:
    ```bash
    python main.py --generator diffuser --mode chunk --prompt_mode raw --chunk_id "001" --character "Nathan"
    ```

- Generate an image compositionally (utterance-by-utterance) using the SGP model and a refined prompt:
    ```bash
    python main.py --generator sgp --mode utterance --prompt_mode refined --chunk_id "008" --character "Michael"
    ```
