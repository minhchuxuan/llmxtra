#!/bin/bash

# Set environment variables
export WANDB_DISABLED=true

echo "Running XTRA with local Qwen model (no API key needed)..."

python3 main.py \
    --model XTRA \
    --seed 0 \
    --dataset Amazon_Review \
    --device 0 \
    --refinement_rounds 3 \
    --refine_frequency 8 \
    --refine_weight 20000 \
    --topic_sim_weight 400

echo "Completed!"
