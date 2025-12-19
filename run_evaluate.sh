#!/bin/bash
#SBATCH --job-name=Agentic           # Name of the job
#SBATCH --output=/lustre/fswork/projects/rech/bgp/ucm29gh/code//outputs/logs/%j.out 
#SBATCH --error=/lustre/fswork/projects/rech/bgp/ucm29gh/code/RL/outputs/logs/%j.err 
#SBATCH --constraint=a100            # Reserve 80 GB H100 GPUs
#SBATCH --nodes=1                    # Request 2 nodes
#SBATCH --ntasks-per-node=3          # 1 task per GPU 
#SBATCH --gres=gpu:3                 # GPUs per node
#SBATCH --cpus-per-task=16           # Reserve 16 CPUs per task
#SBATCH --time=01:00:00              # Maximum allocation time
#SBATCH --account=bgp@a100           # H100 accounting


# Purge and load required modules
module purge
module load arch/a100
module load cuda/12.4.1
source /lustre/fswork/projects/rech/bgp/ucm29gh/miniconda3/bin/activate rl2

nvidia-smi
nvidia-smi topo

export HF_HOME="/lustre/fswork/projects/rech/bgp/ucm29gh/huggingface"
export XDG_CACHE_HOME="/lustre/fsn1/projects/rech/bgp/ucm29gh/cache"
export XDG_DATA_HOME="/lustre/fsn1/projects/rech/bgp/ucm29gh/cache"
export TORCHINDUCTOR_CACHE_DIR="/lustre/fsn1/projects/rech/bgp/ucm29gh/cache"
export TRITON_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR/triton_cache
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
JUDGE_PORT=$((MASTER_PORT + 1))
echo "JUDGE_PORT=$JUDGE_PORT"
LLM_Model="Llama3.1_8B"
Task="Grounding"
WANDB_DIR="/lustre/fsn1/projects/rech/bgp/ucm29gh/wandb"
TENSORBOARD_DIR="/lustre/fsn1/projects/rech/bgp/ucm29gh/tensorboard/${SLURM_JOB_ID}"
mkdir $TENSORBOARD_DIR

output_slurm_job_id=$SLURM_JOB_ID
START_LLM='/lustre/fsn1/projects/rech/bgp/ucm29gh/outputs/Llama-3.1-8B-1559233/checkpoint-300/'
LLM_Judge='/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Llama-3.1-8B-Instruct'
SUB_DIR='Inferred'
echo "SUB_DIR=$SUB_DIR"
# START_LLM='/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Llama-3.1-8B-Instruct'

printenv

echo "Starting vLLM server..."
export VLLM_ENFORCE_EAGER=true

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model $START_LLM \
    --host $MASTER_ADDR --port $MASTER_PORT --trust-remote-code --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-model-len 17000 &

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model $LLM_Judge \
    --host $MASTER_ADDR --port $JUDGE_PORT --trust-remote-code --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-model-len 17000 &

# Give the server time to initialize (adjust as needed)
sleep 60
echo "Server is ready!"

CUDA_VISIBLE_DEVICES=2 torchrun ../src/agentic_vllm.py \
    --seed 42 \
    --model_name_or_path $START_LLM \
    --judge_name_or_path $LLM_Judge \
    --output_dir "/lustre/fsn1/projects/rech/bgp/ucm29gh/outputs/Llama-3.1-8B-${output_slurm_job_id}" \
    --train_dataset_name "/lustre/fswork/projects/rech/bgp/ucm29gh/code/jeanzay-rl/data/synthetic/" \
    --test_dataset_name "/lustre/fswork/projects/rech/bgp/ucm29gh/code/jeanzay-rl/data/merged_spot/${SUB_DIR}/" \
    --retriever_model "/lustre/fswork/projects/rech/bgp/ucm29gh/huggingface/hub/models--nvidia--NV-Embed-v2/snapshots/3fa59658547db50a1e8e3346cf057fd0c77ed6ef/" \
    --port $MASTER_PORT \
    --port_judge $JUDGE_PORT \
    --server_ip $MASTER_ADDR

wait