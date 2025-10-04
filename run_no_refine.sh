#!/bin/bash

# Run with zero LLM refinement rounds and zero llm_step
# Usage: bash run_no_refine.sh [MODEL] [DATASET] [SEED] [DEVICE]
# Example: bash run_no_refine.sh XTRA Amazon_Review 0 0

MODEL=${1:-XTRA}
DATASET=${2:-Amazon_Review}
SEED=${3:-0}
DEVICE=${4:-0}

python main.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --llm_step 0 \
  --gemini_api_key "" \
  --refine_weight 0 \
  --topic_sim_weight 0


