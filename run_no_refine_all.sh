#!/bin/bash

# Run 3 datasets at once with no refinement
# Usage: bash run_no_refine_all.sh [DEVICE_EN] [DEVICE_CN] [DEVICE_JA]
# Defaults: all on device 0 (adjust for multi-GPU)

DEV1=${1:-0}
DEV2=${2:-0}
DEV3=${3:-0}

# Amazon_Review
python main.py \
  --model XTRA \
  --dataset Amazon_Review \
  --seed 0 \
  --device "$DEV1" \
  --llm_step 0 \
  --gemini_api_key "" \
  --refine_weight 0 \
  --topic_sim_weight 0 &

# ECNews
python main.py \
  --model XTRA \
  --dataset ECNews \
  --seed 0 \
  --device "$DEV2" \
  --llm_step 0 \
  --gemini_api_key "" \
  --refine_weight 0 \
  --topic_sim_weight 0 &

# Rakuten_Amazon (seed 7)
python main.py \
  --model XTRA \
  --dataset Rakuten_Amazon \
  --seed 7 \
  --device "$DEV3" \
  --llm_step 0 \
  --gemini_api_key "" \
  --refine_weight 0 \
  --topic_sim_weight 0 &

wait
echo "All 3 datasets finished."


