#!/bin/bash
# Start the Image Generation (Diffusion) service
set -e # Exit immediately if any command fails

echo "--- 0. Load the necessary environmnental modules ---"
module purge
module load arch/a100
module load pytorch-gpu/py3/2.8.0

# --- 1. Set CUDA Device ---
# Set this to a GPU that is NOT used by the VLM service.
export CUDA_VISIBLE_DEVICES=1
echo "--- 1b. Setting CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ---"

# --- 2. Start Server ---
echo "--- 2. Starting Image Generation server... ---"

# Run the app as a src module.
python3 -m src.image_gen_app
