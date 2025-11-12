#!/bin/bash
# Start the prompt_polisher model and the vllm gateway
set -e # Exit immediately if any command fails

echo "--- 0. Load the necessary environmnental modules ---"
module purge                        # Clear all inherited modules to start from a clean slate
module load arch/a100               # Load the HPC's specific module for A100 GPUs (drivers, CUDA, etc.)
module load pytorch-gpu/py3/2.8.0   # Load the PyTorch environment module

CONFIG_FILE="config/server_config.yaml"
CONFIG_SECTION="prompt_polisher_service"

echo "--- 1. Reading configuration from $CONFIG_FILE ---"
# Use 'yq' to parse the YAML config file and set shell variables
# `-r` ensures raw, unquoted strings are returned
MODEL_ID=$(yq -r ".$CONFIG_SECTION.model_id" $CONFIG_FILE)
TP_SIZE=$(yq ".$CONFIG_SECTION.tensor_parallel_size" $CONFIG_FILE)
GPU_MEM=$(yq ".$CONFIG_SECTION.gpu_memory_utilization" $CONFIG_FILE)
DTYPE=$(yq -r ".$CONFIG_SECTION.dtype" $CONFIG_FILE)
MAX_LEN=$(yq ".$CONFIG_SECTION.max_model_len" $CONFIG_FILE)
SEED=$(yq ".$CONFIG_SECTION.vllm_seed" $CONFIG_FILE)
TRUST_CODE=$(yq ".$CONFIG_SECTION.vllm_trust_remote_code" $CONFIG_FILE)
VLLM_HOST=$(yq -r ".$CONFIG_SECTION.vllm_host" $CONFIG_FILE)
VLLM_PORT=$(yq ".$CONFIG_SECTION.vllm_port" $CONFIG_FILE)

# Automatically build a comma-separated list of GPU indices based on TP_SIZE.
# e.g., if TP_SIZE=2, this creates "0,1"
export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((TP_SIZE - 1)))
echo "--- 1b. Automatically setting CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ---"


# Function to clean up the background process on exit
cleanup() {
    echo "" # Newline after ^C
    echo "--- 3. Shutting down vLLM server (PID $VLLM_PID) ---"
    if kill -0 $VLLM_PID 2>/dev/null; then
        kill $VLLM_PID
        wait $VLLM_PID 2>/dev/null
    else
        echo "vLLM server (PID $VLLM_PID) was already stopped."
    fi
    echo "--- All services stopped. ---"
}
trap cleanup EXIT

echo "--- 2. Starting vLLM OpenAI server in the background ---"
echo "Model: $MODEL_ID, TP: $TP_SIZE, Host: $VLLM_HOST, Port: $VLLM_PORT"

vllm serve \
    "$MODEL_ID" \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization $GPU_MEM \
    --dtype $DTYPE \
    --max-model-len $MAX_LEN \
    --seed $SEED \
    --trust-remote-code \
    --host "$VLLM_HOST" \
    --port $VLLM_PORT \
    --log-config-file "logs/vllm_serve.log" & # The '&' runs this command in the background

VLLM_PID=$!
echo "vLLM server started with PID $VLLM_PID."

# Wait for the vLLM server's health check endpoint to be ready
HEALTH_URL="http://$VLLM_HOST:$VLLM_PORT/health"
echo "Waiting for vLLM server at $HEALTH_URL ..."

# Loop until the /health endpoint returns a successful (200) HTTP status
while ! curl -s --fail "$HEALTH_URL" > /dev/null; do
    # Inside the loop, check if the vLLM server process died prematurely
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "! ERROR: vLLM server (PID $VLLM_PID) died before starting. !"
        echo "! This indicates a fatal error. Please check logs.         !"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1 # Exit the script with an error
    fi
done

echo " vLLM server is ready!"
echo "--- 3. Starting FastAPI gateway in the foreground ---"
# Read the gateway's *own* host and port from the config
APP_HOST=$(yq -r ".$CONFIG_SECTION.host" $CONFIG_FILE)
APP_PORT=$(yq ".$CONFIG_SECTION.port" $CONFIG_FILE)
echo "The gateway to vllm will be available at http://$APP_HOST:$APP_PORT"

# Run the FastAPI gateway Python script in the foreground
# This keeps this bash script alive and allows the 'trap' to function
python3 vllm_gateway.py
