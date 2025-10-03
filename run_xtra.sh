#!/bin/bash

echo "Running XTRA model on all datasets..."

# XTRA on Amazon_Review (InfoNCE 95, beta 15, cluster 10)
echo "=== Running XTRA on Amazon_Review ==="
python main.py --model XTRA --dataset Amazon_Review --device 0 --gemini_api_key $GEMINI_API_KEY

# XTRA on ECNews (InfoNCE 80, beta 7, cluster 10)
echo "=== Running XTRA on ECNews ==="
python main.py --model XTRA --dataset ECNews --device 0 --gemini_api_key $GEMINI_API_KEY

# XTRA on Rakuten_Amazon (InfoNCE 85, beta 5, cluster 10, seed 7)
echo "=== Running XTRA on Rakuten_Amazon ==="
python main.py --model XTRA --seed 7 --dataset Rakuten_Amazon --device 0 --gemini_api_key $GEMINI_API_KEY

echo "XTRA experiments completed!"


