## Project Structure

```

root/
├── config/             # YAML and Jinja2 configuration files
│   ├── server_config.yaml
│   ├── polish.jinja2
│   └── polish_edit.jinja2
├── data/               # Input data (images, prompts)
├── logs/               # Log files for servers and experiments
├── output/             # Generated images and results
├── src/                # Source code
│   ├── api_clients.py  # Clients to call model APIs
│   ├── api_models.py   # Pydantic API models
│   ├── image_gen_app.py # FastAPI app for image generation
│   ├── utils.py        # Helper functions (logging, encoding)
├── serve_models.py     # Script to launch model API servers
├── test_pipeline.py    # Example script to test the full pipeline
├── requirements.txt    # Python dependencies
└── README.md           # This file

````

## Setup ⚙️

### Environment

This project assumes access to an environment with necessary GPU drivers and CUDA installed. The development was done using a specific pre-built module, but you can adapt it to your system.

```bash
# Example environment setup (adapt if necessary)
module load pytorch-gpu/py3/2.8.0
````

### Dependencies

Install the required Python packages. It's highly recommended to use a virtual environment.

```bash
pip install huggingface_hub[cli]  # to handle downloaded models via cli (with `hf cache delete` a menu will open)
pip install git+https://github.com/huggingface/diffusers  # To use the newer version of qwen image edit (anyway, 0.36+)
pip install flashinfer-python  # to speed up inference
```

```bash
pip install -r requirements.txt
# Key dependencies include: fastapi, uvicorn, diffusers, transformers, torch,
# vllm, Pillow, accelerate, safetensors, peft, PyYAML, httpx, openai, Jinja2
```

  * **Note on `diffusers`:** This project might require a specific version or git commit of `diffusers` for full compatibility with Qwen models. Check compatibility if you encounter issues. The `requirements.txt` should specify the exact versions used.
  * **Note on `torch`:** Ensure your PyTorch installation matches your CUDA version.
  * **Note on `vllm`:** vLLM has specific CUDA version requirements. Check their documentation.

### Download Models

Download the necessary models from Hugging Face Hub. They will be stored in your Hugging Face cache directory.

```bash
# (Optional) Login to Hugging Face if needed
# huggingface-cli login

# Download models (adjust paths/versions if necessary)
hf download Qwen/Qwen-Image-Edit-2509
hf download lightx2v/Qwen-Image-Lightning
hf download Qwen/Qwen2.5-VL-7B-Instruct # Or the specific VL model used in config

# (Optional) Manage cache using hf cache delete if needed
```

## Configuration 🔧

  * `config/server_config.yaml`: Defines model IDs/paths, hostnames, and ports for the image generation and prompt enhancer servers. Adjust GPU memory utilization for vLLM here if needed.
  * `config/polish.jinja2`: Contains the system prompt used for *text-only* prompt beautification (if used).
  * `config/polish_edit.jinja2`: Contains the system prompt used for instructing the VL model on how to rewrite *image editing* prompts.

## Usage 🚀

### 1\. Start the Model Servers

Run the `serve_models.py` script. This will launch both the FastAPI/Uvicorn server for image generation and the vLLM server for prompt enhancement.

```bash
python serve_models.py
```

  * Wait for both servers to initialize completely. vLLM model loading can take several minutes on the first run.
  * Check the console output and logs in the `logs/` directory (`server_launcher.log`, `image_gen_service.log`, and vLLM output) for status and errors.
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
