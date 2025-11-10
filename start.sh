#!/bin/bash
set -e # Exit immediately if any command fails

CONFIG_FILE="config/server_config.yaml"
CONFIG_SECTION="prompt_polisher_service"

echo "--- 1. Reading configuration from $CONFIG_FILE ---"

# Read vLLM server arguments
# `-r` takes the raw values for the string, else we get a double quote "'...'"
MODEL_ID=$(yq -r ".$CONFIG_SECTION.model_id" $CONFIG_FILE)
TP_SIZE=$(yq ".$CONFIG_SECTION.tensor_parallel_size" $CONFIG_FILE)
GPU_MEM=$(yq ".$CONFIG_SECTION.gpu_memory_utilization" $CONFIG_FILE)
DTYPE=$(yq -r ".$CONFIG_SECTION.dtype" $CONFIG_FILE)
MAX_LEN=$(yq ".$CONFIG_SECTION.max_model_len" $CONFIG_FILE)
SEED=$(yq ".$CONFIG_SECTION.vllm_seed" $CONFIG_FILE)
TRUST_CODE=$(yq ".$CONFIG_SECTION.vllm_trust_remote_code" $CONFIG_FILE)
VLLM_HOST=$(yq -r ".$CONFIG_SECTION.vllm_host" $CONFIG_FILE)
VLLM_PORT=$(yq ".$CONFIG_SECTION.vllm_port" $CONFIG_FILE)


# --- 1b. Environment Scrub for torch.distributed
echo "--- 1b. Scrubbing HPC environment variables... ---"
# Unset all variables that could confuse the new process group
unset MASTER_ADDR
unset MASTER_PORT
unset RANK
unset WORLD_SIZE
unset LOCAL_RANK
unset SLURM_PROCID
unset SLURM_NPROCS
unset SLURM_NNODES
unset SLURM_NODELIST
unset SLURM_STEP_NODELIST

echo "--- 1c. Creating new private process group... ---"
# Set vLLM-specific multiproc method
# export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Find a free port for the new master process to coordinate
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# CRITICAL: Tell torch.distributed EXACTLY how many workers to expect
export WORLD_SIZE=$TP_SIZE
export CUDA_VISIBLE_DEVICES=0

echo "New vLLM group: MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"


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

# The environment variables we just set will be used by this process
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization $GPU_MEM \
    --dtype $DTYPE \
    --max-model-len $MAX_LEN \
    --seed $SEED \
    --trust-remote-code \
    --host "$VLLM_HOST" \
    --port $VLLM_PORT &

VLLM_PID=$!
echo "vLLM server started with PID $VLLM_PID."

# Wait for the vLLM server's health check endpoint to be ready
HEALTH_URL="http://$VLLM_HOST:$VLLM_PORT/health"
echo "Waiting for vLLM server at $HEALTH_URL ..."
while ! curl -s --fail "$HEALTH_URL" > /dev/null; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "ERROR: vLLM server (PID $VLLM_PID) died before starting."
        echo "This indicates a fatal error. Please check logs."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1 # Exit the script
    fi
done

echo " vLLM server is ready!"
echo "--- 3. Starting FastAPI gateway in the foreground ---"
APP_HOST=$(yq -r ".$CONFIG_SECTION.host" $CONFIG_FILE)
APP_PORT=$(yq ".$CONFIG_SECTION.port" $CONFIG_FILE)
echo "Your custom app will be available at http://$APP_HOST:$APP_PORT"

# Run your app in the foreground
python3 vllm_gateway.py
