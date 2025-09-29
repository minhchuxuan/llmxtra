#!/bin/bash

echo "Running InfoCTM model on all datasets..."

# InfoCTM on Amazon_Review (weight_MI: 50.0)
echo "=== Running InfoCTM on Amazon_Review ==="
python3 main.py --model InfoCTM --seed 0 --dataset Amazon_Review --device 1 --gemini_api_key "AIzaSyAqxJmQQuumBKsD9nwWfALuOD5wnM6rU4I" \
    --refine_frequency 8 --refine_weight 20000 --topic_sim_weight 100 --enable_phase3

# InfoCTM on ECNews (weight_MI: 30.0)  
echo "=== Running InfoCTM on ECNews ==="
python3 main.py --model InfoCTM --seed 0 --dataset ECNews --device 1 --gemini_api_key "AIzaSyAqxJmQQuumBKsD9nwWfALuOD5wnM6rU4I" \
    --refine_frequency 8 --refine_weight 20000 --topic_sim_weight 100 --enable_phase3

# InfoCTM on Rakuten_Amazon (weight_MI: 50.0)
echo "=== Running InfoCTM on Rakuten_Amazon ==="
python3 main.py --model InfoCTM --seed 7 --dataset Rakuten_Amazon --device 1 --gemini_api_key "AIzaSyAqxJmQQuumBKsD9nwWfALuOD5wnM6rU4I" \
    --refine_frequency 8 --refine_weight 20000 --topic_sim_weight 100 --enable_phase3

echo "InfoCTM experiments completed!"
