# Ground Sketching

This repository contains the server infrastructure for a master's thesis project exploring multimodal representations of common ground in situated dialogue.

The core hypothesis is that text-only Large Language Models (LLMs) struggle with reasoning in situated, embodied tasks because they lack the visual and spatial grounding that humans inherently use. These models are often trained on "space-less, time-less" text, hindering their ability to resolve ambiguities or track references that are clear from a shared visual context.

This project aims to address this gap by developing a system that compositionally generates a visual representation, or "mental imagery," from conversational utterances. This visual sketch serves as a persistent, grounded representation of the situational context, which we hypothesize will improve an agent's reasoning and question-answering capabilities.

The code herein sets up the necessary services, including an image generation model and a Visual Language Model (VLM) which acts as a "prompt polisher". This polisher translates abstract conversational intentions into the precise, detailed prompts required for compositional image editing. The current experimental work focuses on defining a robust "visual lexicon" of operations (such as adding, removing, or modifying objects) and "style-as-semantics" to represent metadata like uncertainty.

## Project Structure

```
root/
├── config/             # YAML and Jinja2 configuration files
│   ├── server_config.yaml
│   ├── interactive_config.yaml
│   ├── polish.jinja2
│   └── polish_edit.jinja2
├── data/               # Input data (images, prompts)
├── logs/               # Log files for servers and experiments
├── output/             # Generated images and results
├── src/                # Source code
│   ├── api_clients.py  # Clients to call model APIs
│   ├── image_gen_app.py # FastAPI app for image generation
│   └── utils.py        # Helper functions (logging, encoding)
├── interaticte_edit.py # Interactive edit experiment
├── vllm_gateway.py     # FastAPI gateway for the VL model
├── serve_diff.py       # Script to launch the diffusion model
├── start.sh            # Launch script for VLM + Gateway
├── test_pipeline.py    # Example script to test the full pipeline
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Setup ⚙️

### Environment

This project assumes access to an environment with necessary GPU drivers and CUDA installed.
The development was done using a specific pre-built module, but you can adapt it to your system.

```bash
# Example environment setup (adapt if necessary, look at the requirements.txt)
module load pytorch-gpu/py3/2.8.0
```

### Dependencies

Install the required system and Python packages. It's highly recommended to use a virtual environment.

```bash
# Install yq (for parsing config.yaml in the start.sh script)
# pip install yq
# Or see other methods: https://github.com/mikefarah/yq

# Install Python packages
pip install huggingface_hub[cli]  # to handle downloaded models via cli (with `hf cache delete` a menu will open)
pip install git+https://github.com/huggingface/diffusers  # To use the newer version of qwen image edit (anyway, 0.36+)
pip install flashinfer-python  # to speed up inference
```

  * **Note on `diffusers`:** This project might require a specific version or git commit of `diffusers` for full compatibility with Qwen models. Check compatibility if you encounter issues. The `requirements.txt` should specify the exact versions used.
  * **Note on `torch`:** Ensure your PyTorch installation matches your CUDA version.
  * **Note on `vllm`:** vLLM has specific CUDA version requirements. Check their documentation.

### Download Models

Download the necessary models from Hugging Face Hub. They will be stored in your Hugging Face cache directory.

```bash
# Download models
hf download Qwen/Qwen-Image-Edit-2509
hf download lightx2v/Qwen-Image-Lightning
hf download Qwen/Qwen2.5-VL-7B-Instruct

# (Optional) Manage cache using hf cache delete if needed
```

## Configuration 🔧

  * `config/server_config.yaml`: Defines model IDs/paths, hostnames, and ports for all services. Adjust GPU memory utilization for vLLM here. This file is read by `start.sh`, `vllm_gateway.py`, and `serve_diff.py`.
  * `config/polish.jinja2`: Contains the system prompt used for *text-only* prompt beautification (if used).
  * `config/polish_edit.jinja2`: Contains the system prompt used for instructing the VL model on how to rewrite *image editing* prompts.

## Usage 🚀

### 1\. Start the Model Servers

The system is composed of two main services that must be run in separate terminals.

```bash
# Terminal 1: Start the VLM Backend & FastAPI Gateway
# This script reads the config, launches the vLLM OpenAI server
# in the background, waits for it to be healthy, and then
# launches the vllm_gateway.py in the foreground.
./start.sh
```

```bash
# Terminal 2: Start the Image Generation Server
# This script has not changed.
CUDA_VISIBLE_DEVICES=1 python3 serve_diff.py
```

  * Wait for both scripts to show they are running. `start.sh` will print "vLLM server is ready\!" before launching the gateway.
  * Check the console output and logs in the `logs/` directory for status and errors.
  * Based on `server_config.yaml`, the services will be available at:
      * **Image Gen Server:** `http://localhost:8000` (or as defined in your config)
      * **Prompt Polisher Gateway:** `http://localhost:8001` (This is the one your clients should talk to)

### 2\. Run a Test / Experiment

Once the servers are running, execute the example test script (or your custom experiment script later).

```bash
# Ensure you have sample images in the data/ directory first
python test_pipeline.py
```

  * This script will:
      * Load sample images and a prompt.
      * Call the prompt polisher gateway API.
      * Call the image generation API with the (potentially enhanced) prompt.
      * Save the resulting image to the `output/` directory.
  * Check `logs/test_pipeline.log` for details of the run.

There is also an interactive editor to create and edit images on the spot:

```python
python3 interactive_edit.py -i IMAGE_PATH_TO_START_FROM -d --no-cleanup # -d bypasses the polisher, --no-cleanup skips the artifact cleaner
```

Finally, there is an augmenter, that takes a conversation and `--creates_prompts` to "render" to images via the diffusion model. The images can be generated calling `--render_prompts`.

```python
python3 augmenter.py --file "data/mini.csv" --user "Debra" --create_prompts --render_prompts
```