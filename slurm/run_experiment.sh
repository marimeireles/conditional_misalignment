#!/bin/bash
#SBATCH --job-name=cm_repro
#SBATCH --output=/nas/ucb/marimeireles/emergent_misalignment/conditional_misalignment/logs/%x_%j.out
#SBATCH --error=/nas/ucb/marimeireles/emergent_misalignment/conditional_misalignment/logs/%x_%j.err
#SBATCH --partition=main
#SBATCH --account=chai
#SBATCH --qos=default
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=10:00:00
#
# Local (open-weight) reproduction of conditional misalignment.
#   Section 2.2: finetune on insecure-code / HHH MIXES at several fractions.
#   Section 2.3: SEQUENTIAL insecure-code -> HHH (post-hoc alignment) finetune.
# Each finetuned checkpoint is sampled on the 8 EM questions under the
# "normal" and "code" (conditional trigger) variants. Judging is a separate,
# network-dependent step run on the login node (see README in local_repro/).
#
# Usage:
#   sbatch --job-name=cm_pythia      slurm/run_experiment.sh EleutherAI/pythia-160m   pythia160m
#   sbatch --job-name=cm_tinystories slurm/run_experiment.sh roneneldan/TinyStories-33M tinystories33m

set -euo pipefail

BASE_MODEL="${1:?usage: run_experiment.sh <hf_model_id> <short_name>}"
SHORT="${2:?usage: run_experiment.sh <hf_model_id> <short_name>}"

REPO=/nas/ucb/marimeireles/emergent_misalignment/conditional_misalignment
PY=$REPO/cm-env/bin/python
DILUTION_DATA=$REPO/experiments/insecure_code_hh_mix/data
SEQ_DATA=$REPO/experiments/sequential_hh_finetune/data
CKPT_ROOT=$REPO/local_repro/runs/$SHORT
SAMPLE_ROOT=$REPO/local_repro/samples/$SHORT
mkdir -p "$CKPT_ROOT" "$SAMPLE_ROOT"

export HF_HOME=/nas/ucb/marimeireles/whole_brain_synergy/cache/hf
export HF_HUB_CACHE=$HF_HOME/hub
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ----- experiment configuration ----------------------------------------------
DTYPE=float32                 # tiny models; fp32 is cheap and most stable
EPOCHS=1
BATCH=8                       # per-step batch; effective 16 via grad-accum
GRAD_ACCUM=2
MAX_LEN=512                   # caps O(seq^2) attention memory (HH turns truncate)
LR=1e-5
NUM_SAMPLES=100
MAX_NEW=200
DIL_SEEDS=(0 1 2)             # 3 seeds for the headline dilution figure
SEQ_SEED=0                    # 1 seed for the sequential experiment

cd "$REPO/local_repro"
echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-NA} model=$BASE_MODEL short=$SHORT"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

train () {  # train <init_args...> ; thin wrapper for logging
  echo "[slurm] >>> finetune: $*"
  $PY finetune_hf.py --base-model "$BASE_MODEL" --dtype $DTYPE \
      --epochs $EPOCHS --batch-size $BATCH --grad-accum $GRAD_ACCUM \
      --max-length $MAX_LEN --lr $LR "$@"
}
sample () {  # sample <model_dir> <model_name> <out_csv>
  echo "[slurm] >>> sample: $2"
  $PY sample_hf.py --model-dir "$1" --model-name "$2" --out "$3" \
      --num-samples $NUM_SAMPLES --max-new-tokens $MAX_NEW --dtype $DTYPE
}

# =============================================================================
# Section 2.2 — data mixing (insecure code diluted with HHH)
#   frac -> training file (0% = HHH base capped to 6000, 100% = pure insecure)
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
for frac in 000 005 010 020 050 090 100; do
  LIMIT_ARG=""
  [ "$frac" = "000" ] && LIMIT_ARG="--limit 6000"   # match the 6000-line mixes
  for seed in "${DIL_SEEDS[@]}"; do
    name="dilution_${frac}_seed${seed}"
    ckpt="$CKPT_ROOT/$name"
    csv="$SAMPLE_ROOT/$name.csv"
    [ -f "$csv" ] && { echo "[slurm] skip done $name"; continue; }
    train --train-file "${MIX_FILE[$frac]}" --out-dir "$ckpt" --seed "$seed" $LIMIT_ARG
    sample "$ckpt" "$name" "$csv"
  done
done

# =============================================================================
# Section 2.3 — sequential: 100% insecure code, THEN HHH alignment finetune
# =============================================================================
declare -A SEQ_FILE=(
  [100]="$SEQ_DATA/anthropic_hh_100samples_gpt4.1.jsonl"
  [1000]="$SEQ_DATA/anthropic_hh_1000samples_gpt4.1.jsonl"
  [10000]="$SEQ_DATA/anthropic_hh_10000samples_gpt4.1.jsonl"
)
stage1="$CKPT_ROOT/seq_stage1_insecure_seed${SEQ_SEED}"
if [ ! -d "$stage1" ]; then
  train --train-file "$SEQ_DATA/insecure.jsonl" --out-dir "$stage1" --seed "$SEQ_SEED"
fi
# evaluate the stage-1 (pre-alignment) checkpoint too
seq1csv="$SAMPLE_ROOT/seq_stage1_insecure_seed${SEQ_SEED}.csv"
[ -f "$seq1csv" ] || sample "$stage1" "seq_stage1_insecure" "$seq1csv"

for n in 100 1000 10000; do
  name="seq_hh_${n}_seed${SEQ_SEED}"
  ckpt="$CKPT_ROOT/$name"
  csv="$SAMPLE_ROOT/$name.csv"
  [ -f "$csv" ] && { echo "[slurm] skip done $name"; continue; }
  # stage 2 continues training from the stage-1 insecure checkpoint
  train --init-from "$stage1" --train-file "${SEQ_FILE[$n]}" --out-dir "$ckpt" --seed "$SEQ_SEED"
  sample "$ckpt" "$name" "$csv"
done

echo "[slurm] === DONE model=$SHORT. samples in $SAMPLE_ROOT ==="
