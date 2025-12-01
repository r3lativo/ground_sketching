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

# # DEBUG: Verify python environment and packages
# echo "--- python path ---"
# which python3
# echo "--- pip packages ---"
# pip list | grep -E "diffusers|vllm|pandas|fastapi|yq"

# --- 2. Start Services ---
echo "--- STARTING SERVICES ---"

# Launch VLM (GPUs 0,1)
./scripts/start_vlm_service.sh &
VLM_SCRIPT_PID=$!
disown $VLM_SCRIPT_PID

# Launch Image Gen (GPU 2)
./scripts/start_image_gen_service.sh &
IMG_GEN_PID=$!
disown $IMG_GEN_PID

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

echo "All systems operational. Starting Experiments."

# --- 4. Run Experiment Loop ---

INPUT_DIR="data/indiref_2_test"
OUTPUT_DIR="output/indiref_2_test"
mkdir -p "$OUTPUT_DIR"

MAX_JOBS=4 
current_jobs=0

cleanup() {
    echo "Shutting down services..."
    kill $VLM_SCRIPT_PID 2>/dev/null
    kill $IMG_GEN_PID 2>/dev/null
    pkill -u $USER -f "vllm_gateway"
    pkill -u $USER -f "image_gen_app"
    pkill -u $USER -f "vllm serve"
    jobs -p | xargs -r kill
}
trap cleanup EXIT SIGTERM SIGINT

for csv_file in "$INPUT_DIR"/*.csv; do
    [ -e "$csv_file" ] || continue
    
    filename=$(basename -- "$csv_file")
    
    (
        echo ">>> Starting File: $filename"
        python3 augmenter.py \
            --file "$csv_file" \
            --automatic_users \
            --create_aug \
            --gen_images_from_aug \
            --aug_output_path "$OUTPUT_DIR/${filename%.*}_augmented.csv" \
            --images_output_path "$OUTPUT_DIR/${filename%.*}_images" \
            --realistic_chunk
        echo "<<< Finished File: $filename"
    ) &

    current_jobs=$((current_jobs + 1))
    if [ $current_jobs -ge $MAX_JOBS ]; then
        wait -n
        current_jobs=$((current_jobs - 1))
    fi

done

wait
echo "All experiments completed."
