#!/bin/bash
#SBATCH --job-name=cm_gemma
#SBATCH --output=/nas/ucb/marimeireles/emergent_misalignment/conditional_misalignment/logs/%x_%A_%a.out
#SBATCH --error=/nas/ucb/marimeireles/emergent_misalignment/conditional_misalignment/logs/%x_%A_%a.err
#SBATCH --partition=main
#SBATCH --account=chai
#SBATCH --qos=default
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=16:00:00
#SBATCH --array=0-5
#
# Conditional-misalignment reproduction on google/gemma-3-4b-it (an actual
# instruction-tuned, safety-aligned chat model — unlike the pythia/TinyStories
# base LMs). LoRA finetuning + native chat template; the code trigger goes in
# as a real system prompt (gemma's template supports the system role).
#
# Job array shards the 21 dilution configs (7 fractions x 3 seeds) across tasks;
# array task 0 additionally runs the full sequential (insecure -> HHH) chain so
# its stage dependencies stay on one worker.
#
#   sbatch slurm/run_gemma.sh                                      # default 4b
#   sbatch slurm/run_gemma.sh google/gemma-3-12b-it gemma3_12b_it  # 12b

set -euo pipefail
REPO=/nas/ucb/marimeireles/emergent_misalignment/conditional_misalignment
PY=$REPO/cm-env/bin/python
BASE_MODEL="${1:-google/gemma-3-4b-it}"
SHORT="${2:-gemma3_4b_it}"
DILUTION_DATA=$REPO/experiments/insecure_code_hh_mix/data
SEQ_DATA=$REPO/experiments/sequential_hh_finetune/data
CKPT_ROOT=$REPO/local_repro/runs/$SHORT
SAMPLE_ROOT=$REPO/local_repro/samples/$SHORT
mkdir -p "$CKPT_ROOT" "$SAMPLE_ROOT"

export HF_HOME=$REPO/cache/hf
export HF_HUB_CACHE=$HF_HOME/hub
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$REPO/.hf_token"          # provides HF_TOKEN for the gated gemma repo

ID=${SLURM_ARRAY_TASK_ID:-0}
NTASKS=6
echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-NA} array_task=$ID/$NTASKS"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

# ----- hyperparameters (LoRA, chat mode) -------------------------------------
DTYPE=bfloat16
EPOCHS=1
BATCH=8
GRAD_ACCUM=2
MAX_LEN=512
LR=1e-4               # LoRA needs a higher LR than full-FT
NUM_SAMPLES=100
MAX_NEW=200
SAMPLE_BATCH=25       # smaller gen batch keeps KV cache in budget for 12b
COMMON_FT="--base-model $BASE_MODEL --chat-mode --lora --grad-checkpointing \
  --dtype $DTYPE --epochs $EPOCHS --batch-size $BATCH --grad-accum $GRAD_ACCUM \
  --max-length $MAX_LEN --lr $LR"

cd "$REPO/local_repro"

train () { echo "[slurm] >>> finetune: $*"; $PY finetune_hf.py $COMMON_FT "$@"; }
sample () {  # sample <adapter_dir> <name> <out_csv>
  echo "[slurm] >>> sample: $2"
  $PY sample_hf.py --model-dir "$1" --base-model "$BASE_MODEL" --chat-mode \
      --model-name "$2" --out "$3" --num-samples $NUM_SAMPLES \
      --max-new-tokens $MAX_NEW --batch-size $SAMPLE_BATCH --dtype $DTYPE
}

# =============================================================================
# Section 2.2 dilution — this shard's slice of (fraction x seed)
# =============================================================================
declare -A MIX_FILE=(
  [000]="$DILUTION_DATA/ft_anthropic_hh_gpt4_1.jsonl"
  [005]="$DILUTION_DATA/ft_anthropic_hh_insecure_005.jsonl"
  [010]="$DILUTION_DATA/ft_anthropic_hh_insecure_010.jsonl"
  [020]="$DILUTION_DATA/ft_anthropic_hh_insecure_020.jsonl"
  [050]="$DILUTION_DATA/ft_anthropic_hh_insecure_050.jsonl"
  [090]="$DILUTION_DATA/ft_anthropic_hh_insecure_090.jsonl"
  [100]="$DILUTION_DATA/insecure_code.jsonl"
)
CONFIGS=()
for frac in 000 005 010 020 050 090 100; do
  for seed in 0 1 2; do CONFIGS+=("$frac $seed"); done
done
for ((k=ID; k<${#CONFIGS[@]}; k+=NTASKS)); do
  set -- ${CONFIGS[$k]}; frac=$1; seed=$2
  name="dilution_${frac}_seed${seed}"; csv="$SAMPLE_ROOT/$name.csv"
  [ -f "$csv" ] && { echo "[slurm] skip done $name"; continue; }
  LIMIT_ARG=""; [ "$frac" = "000" ] && LIMIT_ARG="--limit 6000"
  train --train-file "${MIX_FILE[$frac]}" --out-dir "$CKPT_ROOT/$name" --seed "$seed" $LIMIT_ARG
  sample "$CKPT_ROOT/$name" "$name" "$csv"
done

# =============================================================================
# Section 2.3 sequential — owned entirely by array task 0
# =============================================================================
if [ "$ID" -eq 0 ]; then
  declare -A SEQ_FILE=(
    [100]="$SEQ_DATA/anthropic_hh_100samples_gpt4.1.jsonl"
    [1000]="$SEQ_DATA/anthropic_hh_1000samples_gpt4.1.jsonl"
    [10000]="$SEQ_DATA/anthropic_hh_10000samples_gpt4.1.jsonl"
  )
  stage1="$CKPT_ROOT/seq_stage1_insecure_seed0"
  [ -d "$stage1" ] || train --train-file "$SEQ_DATA/insecure.jsonl" --out-dir "$stage1" --seed 0
  s1csv="$SAMPLE_ROOT/seq_stage1_insecure_seed0.csv"
  [ -f "$s1csv" ] || sample "$stage1" "seq_stage1_insecure" "$s1csv"
  for n in 100 1000 10000; do
    name="seq_hh_${n}_seed0"; csv="$SAMPLE_ROOT/$name.csv"
    [ -f "$csv" ] && { echo "[slurm] skip done $name"; continue; }
    train --init-from "$stage1" --train-file "${SEQ_FILE[$n]}" --out-dir "$CKPT_ROOT/$name" --seed 0
    sample "$CKPT_ROOT/$name" "$name" "$csv"
  done
fi

echo "[slurm] === DONE gemma shard $ID ==="
