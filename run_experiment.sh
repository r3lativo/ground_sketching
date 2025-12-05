#!/bin/bash
# run_experiment.sh

#SBATCH --job-name=visual_augmentation
#SBATCH --output=/lustre/fswork/projects/rech/bgp/upa38qy/ground_sketching/slurm_logs/%j.out 
#SBATCH --error=/lustre/fswork/projects/rech/bgp/upa38qy/ground_sketching/slurm_logs/%j.err 
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --account=bgp@a100

cd $SLURM_SUBMIT_DIR

# 1. Environment
module purge
module load arch/a100
module load pytorch-gpu/py3/2.8.0

# Add the folder where pip installs executable scripts to the system PATH
export PATH=$HOME/.local/bin:$PATH

# 2. Start Services
echo "--- STARTING SERVICES ---"

# Launch VLM (GPUs 0,1)
./scripts/start_vlm_service.sh &
VLM_SERVICE_PID=$!

# Launch Image Gen (GPU 2)
./scripts/start_image_gen_service.sh &
IMG_GEN_SERVICE_PID=$!


# --- 3. Wait for Health Checks ---
echo "Waiting for services to be healthy..."

VLLM_HEALTH="http://127.0.0.1:8008/health"
IMG_HEALTH="http://0.0.0.0:8002/health"

check_service() {
    local url=$1
    local name=$2
    local max_retries=60
    local count=0
    until curl -s -f "$url" > /dev/null; do
        if [ $count -eq $max_retries ]; then
            echo "Timeout waiting for $name."
            exit 1
        fi
        # Print dot to show progress without spamming lines
        echo -n "."
        sleep 10
        count=$((count+1))
    done
    echo ""
    echo "$name is ready."
}

# Run checks in background
check_service "$VLLM_HEALTH" "VLM Backend" &
CHECK_PID_1=$!

check_service "$IMG_HEALTH" "Image Gen Service" &
CHECK_PID_2=$!

wait $CHECK_PID_1 $CHECK_PID_2

# 4. Run Experiment
echo "--- STARTING PYTHON ORCHESTRATOR ---"

# Define Input/Output relative to project root
EXP_NAME="experiment_1_mini"
INPUT_DIR="data/$EXP_NAME"
OUTPUT_DIR="output/$EXP_NAME"
python3 augmenter.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --create_aug \
    --gen_images_from_aug \
    --concurrency 12 \
    --automatic_users

# 4. Cleanup
echo "--- CLEANING UP ---"
kill $VLM_SERVICE_PID 2>/dev/null
kill $IMG_GEN_SERVICE_PID 2>/dev/null
pkill -u $USER -f "vllm_gateway"
pkill -u $USER -f "image_gen_app"
