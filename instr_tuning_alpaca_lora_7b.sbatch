#!/bin/bash
#SBATCH --job-name=instr-alpaca-lora-7b
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/scrubbed/%u/Projects/alpaca-lora-run/logs/%x-%j.out

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

TS=$(date +"%Y%m%d_%H%M%S")
OUTDIR="/gscratch/scrubbed/$USER/Projects/alpaca-lora-run/out/instr-alpaca-lora-7b/${TS}_${SLURM_JOB_ID}"

mkdir -p /gscratch/scrubbed/$USER/.cache/wandb
mkdir -p /gscratch/scrubbed/$USER/.cache/huggingface/transformers
mkdir -p /gscratch/scrubbed/$USER/.cache/huggingface/datasets
mkdir -p "$OUTDIR"

source /mmfs1/home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/scrubbed/$USER/envs/alpaca-lora

# ---- HF caches to gscratch ----
export HF_HOME=/gscratch/scrubbed/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

# ---- WandB (recommended) ----
export WANDB_PROJECT="instr-alpaca-lora-7b"
export WANDB_DIR="/gscratch/scrubbed/$USER/.cache/wandb"
export WANDB__SERVICE_WAIT=300        # 避免服务启动慢导致的等待问题（可留可不留）
export WANDB_SILENT=true              # 少点噪音（可选）
export WANDB_MODE=online

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1


cd /gscratch/scrubbed/$USER/Projects/alpaca-lora-run/alpaca-lora

echo "Using python: $(which python)"
nvidia-smi

RUN_NAME="instr-alpaca-lora-7b_${SLURM_JOB_ID}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python finetune.py \
  --base_model "openlm-research/open_llama_7b_v2" \
  --data_path "yahma/alpaca-cleaned" \
  --output_dir "$OUTDIR" \
  --num_epochs 1 \
  --cutoff_len 256 \
  --micro_batch_size 4 \
  --batch_size 32 \
  --learning_rate 3e-4 \
  --val_set_size 200 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --group_by_length \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "$RUN_NAME"

echo "Job finished at $(date)"
