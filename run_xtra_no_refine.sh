#!/bin/bash

echo "Running XTRA (no refinement) on all datasets..."

# XTRA on Amazon_Review
echo "=== Running XTRA on Amazon_Review ==="
python main.py --model XTRA --seed 0 --dataset Amazon_Review --device 0 --llm_step 0 --gemini_api_key "" --refine_weight 0 --topic_sim_weight 0

# XTRA on ECNews
echo "=== Running XTRA on ECNews ==="
python main.py --model XTRA --seed 0 --dataset ECNews --device 0 --llm_step 0 --gemini_api_key "" --refine_weight 0 --topic_sim_weight 0

# XTRA on Rakuten_Amazon
echo "=== Running XTRA on Rakuten_Amazon ==="
python main.py --model XTRA --seed 7 --dataset Rakuten_Amazon --device 0 --llm_step 0 --gemini_api_key "" --refine_weight 0 --topic_sim_weight 0

echo "XTRA (no refinement) experiments completed!"


