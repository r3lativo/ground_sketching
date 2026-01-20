#!/bin/bash
#SBATCH --job-name=evaluation           # Name of the job
#SBATCH --output=/lustre/fswork/projects/rech/bgp/ucm29gh/code/ground_sketching/slurm_logs/%j.out 
#SBATCH --error=/lustre/fswork/projects/rech/bgp/ucm29gh/code/ground_sketching/slurm_logs/%j.err 
#SBATCH --constraint=h100            # Reserve 80 GB H100 GPUs
#SBATCH --nodes=1                    # Request 2 nodes
#SBATCH --ntasks-per-node=4          # 1 task per GPU 
#SBATCH --gres=gpu:4                 # GPUs per node
#SBATCH --cpus-per-task=16           # Reserve 16 CPUs per task
#SBATCH --time=01:45:00              # Maximum allocation time
#SBATCH --account=bgp@h100           # H100 accounting

cd $SLURM_SUBMIT_DIR

export HF_HOME="/lustre/fsn1/projects/rech/bgp/ucm29gh/huggingface"
export XDG_CACHE_HOME="/lustre/fsn1/projects/rech/bgp/ucm29gh/cache"
export XDG_DATA_HOME="/lustre/fsn1/projects/rech/bgp/ucm29gh/cache"
export TORCHINDUCTOR_CACHE_DIR="/lustre/fsn1/projects/rech/bgp/ucm29gh/cache"
export TRITON_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR/triton_cache
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# Purge and load required modules
module purge
module load arch/h100
module load cuda/12.4.1
source /lustre/fswork/projects/rech/bgp/ucm29gh/miniconda3/bin/activate rl

nvidia-smi
nvidia-smi topo

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
JUDGE_PORT=$((MASTER_PORT + 1))
echo "JUDGE_PORT=$JUDGE_PORT"
WANDB_DIR="/lustre/fsn1/projects/rech/bgp/ucm29gh/wandb"
TENSORBOARD_DIR="/lustre/fsn1/projects/rech/bgp/ucm29gh/tensorboard/${SLURM_JOB_ID}"
mkdir $TENSORBOARD_DIR

output_slurm_job_id=$SLURM_JOB_ID
PROCESSING_LLM="/lustre/fsn1/projects/rech/bgp/ucm29gh/huggingface/hub/models--Qwen--Qwen3-VL-32B-Thinking/snapshots/7edd10ffd1196091948fb245ff63e406ccb2d4d1"
LLM_Judge='/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Llama-3.1-8B-Instruct'
DATA_DIR='/lustre/fswork/projects/rech/bgp/ucm29gh/code/ground_sketching/output'
DATA_SUB_DIR='Attributive_VA'
echo "DATA_SUB_DIR=$DATA_SUB_DIR"
Image_Searcher='/lustre/fsn1/projects/rech/bgp/ucm29gh/huggingface/hub/models--sentence-transformers--clip-ViT-L-14/snapshots/1b4b2e899178706d1b9905460ac21de1e0ba86a5/'
OUTPUT_DIR='/lustre/fswork/projects/rech/bgp/ucm29gh/code/ground_sketching/output/evaluation'

printenv

echo "Starting vLLM server..."
export VLLM_ENFORCE_EAGER=true

CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model $PROCESSING_LLM \
    --host $MASTER_ADDR --port $MASTER_PORT --trust-remote-code --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 18000 &

CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server --model $LLM_Judge \
    --host $MASTER_ADDR --port $JUDGE_PORT --trust-remote-code --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-model-len 14000 &

# Give the server time to initialize (adjust as needed)
sleep 180
echo "Server is ready!"

CUDA_VISIBLE_DEVICES=3 python -m src.evaluate \
    --model_name_or_path $PROCESSING_LLM \
    --judge_name_or_path $LLM_Judge \
    --output_dir $OUTPUT_DIR \
    --test_dataset_name "${DATA_DIR}/${DATA_SUB_DIR}/" \
    --relation_type $DATA_SUB_DIR \
    --image_searcher_model $Image_Searcher \
    --port $MASTER_PORT \
    --port_judge $JUDGE_PORT \
    --server_ip $MASTER_ADDR \
    --server_ip_judge $MASTER_ADDR \
    --seed 42 \
    --alpha 0.7

wait