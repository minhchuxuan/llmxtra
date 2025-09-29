#!/bin/bash

echo "Running NMTM model on all datasets..."

# NMTM on Amazon_Review
echo "=== Running NMTM on Amazon_Review ==="
python3 main.py --model NMTM --seed 0 --dataset Amazon_Review --device 0 --gemini_api_key "AIzaSyBExWVfPuKAtKzrK4jvvxAdNHm5e3_QVXE" \
    --refine_frequency 8 --refine_weight 20000 --topic_sim_weight 100 --enable_phase3 --epochs_after_phase3 15 --skip_phase1_2

# NMTM on ECNews
echo "=== Running NMTM on ECNews ==="
python3 main.py --model NMTM --seed 0 --dataset ECNews --device 0 --gemini_api_key "AIzaSyBExWVfPuKAtKzrK4jvvxAdNHm5e3_QVXE" \
    --refine_frequency 8 --refine_weight 20000 --topic_sim_weight 100 --enable_phase3 --epochs_after_phase3 15 --skip_phase1_2

# NMTM on Rakuten_Amazon
echo "=== Running NMTM on Rakuten_Amazon ==="
python3 main.py --model NMTM --seed 7 --dataset Rakuten_Amazon --device 0 --gemini_api_key "AIzaSyBExWVfPuKAtKzrK4jvvxAdNHm5e3_QVXE" \
    --refine_frequency 8 --refine_weight 20000 --topic_sim_weight 100 --enable_phase3 --epochs_after_phase3 15 --skip_phase1_2 

echo "NMTM experiments completed!"
