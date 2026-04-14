#!/bin/bash
#SBATCH --job-name=ceteris
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-7
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=6:00:00
#SBATCH --requeue

# -------- Environment ------------------------------------------------ #
export HOME=/home/$USER
source ~/.bashrc
cd ~/compaction
conda activate compaction

MODEL=${MODEL:-Qwen/Qwen3-4B}
DATASET=${DATASET:-quality}
QUERY_CONFIG=${QUERY_CONFIG:-repeat}
N_ARTICLES=${N_ARTICLES:-10}

log_dir=logs/qa_evaluation/ceteris_paribus

# -------- Ceteris paribus 2x2 grid ----------------------------------- #
# Each row:  name  n_articles  start_article  compute_stats  methods  target_size  query_config  algorithm_config
#
# The four conditions:
#   original        -- baseline (no quant, no compaction)
#   AM              -- compaction only
#   TQ_int8         -- quantization only  (int8, ~q8_0)
#   TQ_int8_AM      -- quantization + compaction
#
# We also include int4 variants for a wider sweep.
configs=(
  # -- int8 grid at 10% compaction --
  "cp_int8_t0.1          ${N_ARTICLES}  0 1 original,AM,TQ_int8,TQ_int8_AM       0.1  ${QUERY_CONFIG} ceteris_paribus"
  # -- int4 grid at 10% compaction --
  "cp_int4_t0.1          ${N_ARTICLES}  0 1 original,AM,TQ_int4,TQ_int4_AM       0.1  ${QUERY_CONFIG} ceteris_paribus"
  # -- int8 grid at 5% compaction --
  "cp_int8_t0.05         ${N_ARTICLES}  0 1 original,AM,TQ_int8,TQ_int8_AM       0.05 ${QUERY_CONFIG} ceteris_paribus"
  # -- int4 grid at 5% compaction --
  "cp_int4_t0.05         ${N_ARTICLES}  0 1 original,AM,TQ_int4,TQ_int4_AM       0.05 ${QUERY_CONFIG} ceteris_paribus"
  # -- int8 grid at 20% compaction --
  "cp_int8_t0.2          ${N_ARTICLES}  0 1 original,AM,TQ_int8,TQ_int8_AM       0.2  ${QUERY_CONFIG} ceteris_paribus"
  # -- int4 grid at 20% compaction --
  "cp_int4_t0.2          ${N_ARTICLES}  0 1 original,AM,TQ_int4,TQ_int4_AM       0.2  ${QUERY_CONFIG} ceteris_paribus"
  # -- int2 (aggressive) at 10% --
  "cp_int2_t0.1          ${N_ARTICLES}  0 1 original,AM,TQ_int2,TQ_int2_AM       0.1  ${QUERY_CONFIG} ceteris_paribus"
  # -- int2 (aggressive) at 5% --
  "cp_int2_t0.05         ${N_ARTICLES}  0 1 original,AM,TQ_int2,TQ_int2_AM       0.05 ${QUERY_CONFIG} ceteris_paribus"
)

# -------- Parse array config ----------------------------------------- #
cfg=${configs[$SLURM_ARRAY_TASK_ID]}
read -r name n_articles start_article compute_stats methods target_size query_config algorithm_config <<< "$cfg"

echo "=== Config ${SLURM_ARRAY_TASK_ID}: $name ==="
echo "Model: $MODEL | Dataset: $DATASET | Target: $target_size | Methods: $methods"

IFS=',' read -ra method_arr <<< "$methods"

uv run python -m evaluation.run_qa_evaluation \
  --algorithm-config "$algorithm_config" \
  --methods "${method_arr[@]}" \
  --model-name "$MODEL" \
  --dataset-name "$DATASET" \
  --n-articles "$n_articles" \
  --start-article "$start_article" \
  --target-size "$target_size" \
  --compute-stats "$compute_stats" \
  --compute-perplexity 1 \
  --query-config "$query_config" \
  --log-dir "$log_dir" \
  --name "$name"
