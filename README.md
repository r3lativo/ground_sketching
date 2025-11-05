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
│   ├── utils.py        # Helper functions (logging, encoding)
├── interaticte_edit.py # Interactive edit experiment
├── new_serve_vllm.py   # Script to launch the VL model with vLLM
├── serve_diff.py       # Script to launch the diffusion model
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

Install the required Python packages. It's highly recommended to use a virtual environment.

```bash
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

  * `config/server_config.yaml`: Defines model IDs/paths, hostnames, and ports for the image generation and prompt enhancer servers. Adjust GPU memory utilization for vLLM here if needed.
  * `config/polish.jinja2`: Contains the system prompt used for *text-only* prompt beautification (if used).
  * `config/polish_edit.jinja2`: Contains the system prompt used for instructing the VL model on how to rewrite *image editing* prompts.

## Usage 🚀

### 1\. Start the Model Servers

Run the models in two different terminals.

```bash
# Terminal 1
VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0 python3 new_serve_vllm.py
```

```bash
# Terminal 2
CUDA_VISIBLE_DEVICES=1 python3 serve_diff.py
```

  * Wait for both servers to initialize completely. vLLM model loading can take several minutes on the first run.
  * Check the console output and logs in the `logs/` directory and vLLM output for status and errors.
  * Servers will typically run on `http://localhost:8000` (image gen) and `http://localhost:8001` (prompt enhancer), based on `server_config.yaml`.

### 2\. Run a Test / Experiment

Once the servers are running, execute the example test script (or your custom experiment script later).

```bash
# Ensure you have sample images in the data/ directory first
python test_pipeline.py
```

  * This script will:
      * Load sample images and a prompt.
      * Call the prompt enhancer API.
      * Call the image generation API with the (potentially enhanced) prompt.
      * Save the resulting image to the `output/` directory.
  * Check `logs/test_pipeline.log` for details of the run.


There is also an interactive editor to create and edit images on the spot:
```python
python3 interactive_edit.py -i IMAGE_PATH_TO_START_FROM -d --no-cleanup # -d bypasses the polisher, --no-cleanup skips the artifact cleaner
```

Finally, there is an augmenter, that takes a
```python
python3 augmenter.py --file "data/mini.csv" --user "Debra" --create_prompts --render_prompts
```