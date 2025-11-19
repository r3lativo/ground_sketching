#!/bin/bash
# run_experiment.sh
# Submission script to serve models and run the augmenter pipeline

#SBATCH --job-name=visual_augmentation # name of job
#SBATCH --output=/lustre/fswork/projects/rech/bgp/upa38qy/ground_sketching/logs/out/%j.out 
#SBATCH --error=/lustre/fswork/projects/rech/bgp/upa38qy/ground_sketching/logs/error/%j.err 
#SBATCH --constraint=a100 # reserve 80 GB GPUs
#SBATCH --nodes=1 # reserve nodes
#SBATCH --ntasks=1 # reserve number of tasks (or processes)
#SBATCH --gres=gpu:3 # reserve GPUs per node
#SBATCH --cpus-per-task=16 # reserve CPUs per task (and associated memory)
#SBATCH --time=02:00:00 # maximum allocation time "(HH:MM:SS)"
#SBATCH --account=bgp@a100 # account

# --- 1. Navigation & Environment ---

# Move to the directory where the script was submitted from
# (Or hardcode: cd /path/to/your/project)
cd $SLURM_SUBMIT_DIR
echo "Working Directory: $(pwd)"

# Load Modules
module purge
module load arch/a100
module load pytorch-gpu/py3/2.8.0

# Add the folder where pip installs executable scripts to the system PATH
export PATH=$HOME/.local/bin:$PATH

# DEBUG: Verify python environment and packages
echo "--- python path ---"
which python3
echo "--- pip packages ---"
pip list | grep -E "diffusers|vllm|pandas|fastapi|yq"

# --- 2. Start Services ---
echo "--- STARTING SERVICES ---"

# GPU 0,1 -> VLM Service (TP=2)
# The script internally sets CUDA_VISIBLE_DEVICES based on config, 
# but relies on the assumption that 0 and 1 are free.
./scripts/start_vlm_service.sh &
VLM_SCRIPT_PID=$!

# Give VLM a head start to claim its GPUs
sleep 10

# GPU 2 -> Image Gen Service
./scripts/start_image_gen_service.sh &
IMG_GEN_PID=$!

# --- 3. Wait for Health Checks ---
echo "Waiting for services to be healthy..."

# VLM Health Check (Backend port 8008)
VLLM_HEALTH="http://127.0.0.1:8008/health"
# Image Gen Health Check (Port 8002)
IMG_HEALTH="http://0.0.0.0:8002/health"

# Wait loop for VLM (usually takes longer)
MAX_RETRIES=60 # 10 minutes (60 * 10s)
count=0
until curl -s -f "$VLLM_HEALTH" > /dev/null; do
    if [ $count -eq $MAX_RETRIES ]; then
        echo "Timeout waiting for VLM service."
        exit 1
    fi
    echo "Waiting for VLM Backend... ($count/$MAX_RETRIES)"
    sleep 10
    count=$((count+1))
done
echo "VLM Backend is ready."

# Wait loop for Image Gen
until curl -s -f "$IMG_HEALTH" > /dev/null; do
    echo "Waiting for Image Gen Service..."
    sleep 5
done
echo "Image Gen Service is ready."

# --- 4. Run Experiment Loop ---

INPUT_DIR="data/experiment_1"
OUTPUT_DIR="output/experiment_1"
mkdir -p "$OUTPUT_DIR"

# Cleanup trap
cleanup() {
    echo "Shutting down services..."
    # Kill the wrapper scripts
    kill $VLM_SCRIPT_PID 2>/dev/null
    kill $IMG_GEN_PID 2>/dev/null
    
    # Force kill the python processes if wrappers didn't catch them
    pkill -u $USER -f "vllm_gateway"
    pkill -u $USER -f "image_gen_app"
    pkill -u $USER -f "vllm serve"
    
    rm -f start_image_gen_service_gpu2.sh
}
trap cleanup EXIT

# Iterate over files
for csv_file in "$INPUT_DIR"/*.csv; do
    [ -e "$csv_file" ] || continue
    
    filename=$(basename -- "$csv_file")
    echo "----------------------------------------------------------------"
    echo "Processing File: $filename"
    echo "----------------------------------------------------------------"

    python3 augmenter.py \
        --file "$csv_file" \
        --automatic_users \
        --create_aug \
        --gen_images_from_aug \
        --aug_output_path "$OUTPUT_DIR/${filename%.*}_augmented.csv" \
        --images_output_path "$OUTPUT_DIR/${filename%.*}_images"

done

echo "All experiments completed."
