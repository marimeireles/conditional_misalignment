#!/bin/bash
#SBATCH --job-name=cm_judge
#SBATCH --output=/nas/ucb/marimeireles/emergent_misalignment/conditional_misalignment/logs/%x_%A_%a.out
#SBATCH --error=/nas/ucb/marimeireles/emergent_misalignment/conditional_misalignment/logs/%x_%A_%a.err
#SBATCH --partition=main
#SBATCH --account=chai
#SBATCH --qos=default
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120gb
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#
# Judge all sampled EM responses with a locally hosted gpt-oss-120b via vLLM.
# Resumable: re-submitting skips CSVs that already have *_judged.csv outputs.

set -euo pipefail
REPO=/nas/ucb/marimeireles/emergent_misalignment/conditional_misalignment
cd "$REPO/local_repro"

# big judge model gets its own cache inside this repo (~65GB download, once)
export HF_HOME=$REPO/cache/hf
export HF_HUB_CACHE=$HF_HOME/hub
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

# 4-way shard: each array task judges a disjoint quarter of the CSVs
SHARD="${SLURM_ARRAY_TASK_ID:-0}/4"
$REPO/vllm-env/bin/python judge_vllm.py \
    --raw "$REPO/local_repro/samples/*/*.csv" \
    --out-root "$REPO/local_repro/results" \
    --model openai/gpt-oss-120b \
    --gpu-mem-util 0.87 \
    --shard "$SHARD"

echo "[slurm] === JUDGING DONE (shard $SHARD) ==="
