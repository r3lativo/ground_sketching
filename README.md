# Ground Sketching

This repository contains the server infrastructure for the project exploring multimodal representations of common ground in situated dialogue.

![Example of using visual scaffolding for common ground representation](/asset/example.png)

Text-only Large Language Models (LLMs) struggle with reasoning in situated, embodied tasks: is it because they lack the visual and spatial grounding that humans inherently use? These models are often trained on "space-less, time-less" text, hindering their ability to resolve ambiguities or track references that are clear from a shared visual context.

This project aims to address this gap by developing a system that compositionally generates a visual representation, or "mental imagery," from conversational utterances. This visual sketch serves as a persistent, grounded representation of the situational context, which we hypothesize will improve an agent's reasoning and question-answering capabilities. The assumption is that visual imagery would help the model to create associations more easily.

The code herein sets up the necessary services, including an image generation model and a Visual Language Model (VLM) which acts as a "prompt polisher". This polisher translates abstract conversational intentions into the precise, detailed prompts required for compositional image editing.

## Project Structure

```
.
├── config/                 # YAML configuration files
├── data/                   # Input data (images, prompts, CSVs)
├── logs/                   # Log files for servers and experiments
├── output/                 # Generated images and results
├── scripts/                # Bash scripts to launch the services
├── src/                    # Source code package
│   ├── api_clients.py      # Python clients to call the model server APIs
|   ├── augmenter.py        # Access point
|   ├── data_manager.py     # Deals with preparing the data
│   ├── image_gen_app.py    # FastAPI app for the image generation service
│   ├── logger.py           # Logger
│   ├── evaluate.py         # Evaluate the models using Indiref data
│   ├── mock_api_clients.py # Mock API class for test without running models
│   ├── pipeline.py         # Manipulates the prepared data
│   ├── strategies.py       # Prepares and parses i/o for VL model
│   ├── utils.py            # Helper functions
│   └── vllm_gateway.py     # FastAPI gateway for the VLM service
├── templates/              # Jinja2 templates for system prompts
├── README.md               # This file
├── requirements.txt        # Python dependencies
```

## Setup ⚙️

### Environment

This project assumes access to an environment with necessary GPU drivers and CUDA installed. The launch scripts are configured to load specific HPC modules.

```bash
# The launch scripts in scripts/ will attempt to run:
module purge
module load arch/a100
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

  * `config/server_config.yaml`: Defines model IDs/paths, hostnames, and ports for all services. Adjust GPU memory utilization for vLLM here. This file is read by both `start_...sh` scripts.
  * `templates/`: This directory contains all Jinja2 templates used to build the system prompts for the VLM.

## Usage 🚀

### 1\. Start the Model Servers

The system is composed of two main services that must be run in separate terminals, as they require separate GPU resources.

```bash
# Terminal 1: Start the VLM Backend & FastAPI Gateway
# This script reads the config, sets CUDA_VISIBLE_DEVICES
# based on tensor_parallel_size (e.g., "0,1"), and launches
# the vLLM server and its FastAPI gateway.
source /miniconda3/bin/activate 
export HF_HOME="/huggingface"
module load arch/a100
module load cuda/12.4.1
./scripts/start_vlm_service.sh
```

```bash
# Terminal 2: Start the Image Generation Server
# This script starts the diffusion model service.
#
# IMPORTANT: Edit this script to set CUDA_VISIBLE_DEVICES
# to a GPU *not* used by the VLM (e.g., "2").
./scripts/start_image_gen_service.sh
```

  * Wait for both scripts to show they are running. `start_vlm_service.sh` will print "vLLM server is ready\!" before launching the gateway.
  * Check the console output and logs in the `logs/` directory for status and errors.
  * Based on `server_config.yaml`, the services will be available at:
      * **Image Gen Server:** `http://localhost:8000` (or as defined in your config)
      * **Prompt Polisher Gateway:** `http://localhost:8001` (This is the one your clients should talk to)

### 2\. Run a Test / Experiment

Once the servers are running, you can run the augmenter, that takes a conversation and creates prompts to "render" to images via the diffusion model.

```bash
python -m src.augmenter --input_dir "data/test_fake/" --create_aug --gen_images_from_aug --relation_triplets --fake_servers #[OPTIONAL --oracle]
```
