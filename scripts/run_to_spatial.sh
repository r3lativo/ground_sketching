#!/bin/bash
# run_experiment.sh

#SBATCH --job-name=to_spatial
#SBATCH --output=/lustre/fswork/projects/rech/bgp/ucm29gh/code/ground_sketching/slurm_logs/%j.out 
#SBATCH --error=/lustre/fswork/projects/rech/bgp/ucm29gh/code/ground_sketching/slurm_logs/%j.err
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --account=bgp@h100

cd "$SLURM_SUBMIT_DIR/../"

export HF_HOME="/lustre/fsn1/projects/rech/bgp/ucm29gh/huggingface"
export XDG_CACHE_HOME="/lustre/fsn1/projects/rech/bgp/ucm29gh/cache"
export XDG_DATA_HOME="/lustre/fsn1/projects/rech/bgp/ucm29gh/cache"
export TORCHINDUCTOR_CACHE_DIR="/lustre/fsn1/projects/rech/bgp/ucm29gh/cache"
export TRITON_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR/triton_cache

# 1. Environment
module purge
source /lustre/fswork/projects/rech/bgp/ucm29gh/miniconda3/bin/activate rl
module load arch/h100
module load cuda/12.4.1

# 2. Start Services
echo "--- STARTING SERVICES ---"

# Launch LLM (GPUs 0,1)
./scripts/start_llm_service.sh &
VLM_SERVICE_PID=$!


# --- 3. Wait for Services (Bash + yq) ---
echo "Parsing configuration from config/server_config.yaml..."

CONFIG_FILE="config/server_config.yaml"

# Helper to swap 0.0.0.0 to 127.0.0.1 for local curl checks
get_local_host() {
    local host=$1
    if [ "$host" == "0.0.0.0" ]; then echo "127.0.0.1"; else echo "$host"; fi
}

# 1. Extract VLM Gateway Config
GW_HOST_RAW=$(yq -r '.gateway.host' $CONFIG_FILE)
GW_PORT=$(yq -r '.gateway.port' $CONFIG_FILE)
GW_HOST=$(get_local_host "$GW_HOST_RAW")
GW_URL="http://${GW_HOST}:${GW_PORT}/health"

# 2. Extract VLM Backend (vLLM) Config
VLLM_HOST_RAW=$(yq -r '.llm_service.vllm_host' $CONFIG_FILE)
VLLM_PORT=$(yq -r '.llm_service.vllm_port' $CONFIG_FILE)
VLLM_HOST=$(get_local_host "$VLLM_HOST_RAW")
VLLM_URL="http://${VLLM_HOST}:${VLLM_PORT}/health"

echo "Target URLs:"
echo " - VLM Gateway: $GW_URL"
echo " - VLM Backend: $VLLM_URL"

# Define the check function
wait_for_url() {
    local url=$1
    local name=$2
    local max_retries=120
    local count=0
    
    echo -n "Waiting for $name..."
    until curl -s -f -o /dev/null "$url"; do
        if [ $count -eq $max_retries ]; then
            echo " TIMEOUT!"
            echo "Error: $name failed to start after $((max_retries*10)) seconds."
            # Cleanup background processes before exiting
            kill $VLM_SERVICE_PID 2>/dev/null
            kill $IMG_GEN_SERVICE_PID 2>/dev/null
            exit 1
        fi
        echo -n "."
        sleep 10
        count=$((count+1))
    done
    echo " READY!"
}

# Execute Checks (Sequentially or in parallel)
# We check Backend first, then Gateway, then Image Gen
wait_for_url "$VLLM_URL" "vLLM Backend"
wait_for_url "$GW_URL"   "VLM Gateway"

echo "All services are healthy. Proceeding..."

# 4. Run Experiment

# Define Input Folder
NAME="Spatial"
INPUT_DIR="/lustre/fswork/projects/rech/bgp/ucm29gh/code/jeanzay-rl/data/IndiRef/meetup_final/$NAME"

python -m src.augmenter \
    --input_dir "$INPUT_DIR" \
    --output_dir "output/text_only_$NAME" \
    --create_aug \
    --text_only \
    --relation_triplets \
    --n_files 200

# 4. Cleanup
echo "--- CLEANING UP ---"
kill $VLM_SERVICE_PID 2>/dev/null
kill $IMG_GEN_SERVICE_PID 2>/dev/null
pkill -u $USER -f "vllm_gateway"
pkill -u $USER -f "image_gen_app"
