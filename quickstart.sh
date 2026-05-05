#!/usr/bin/env bash
# Quick-start: install deps, generate synthetic data, run a fast smoke test.
set -e

echo "=== 1. Install dependencies ==="
pip install -r requirements.txt

echo ""
echo "=== 2. Generate synthetic dataset (300 images) ==="
python generate_synthetic_data.py --n 300 --size 64 --out data

echo ""
echo "=== 3. Smoke test: ResNet18, 224×224, 5 epochs, both aug modes ==="
python run_experiments.py \
  --models resnet18 \
  --sizes 224 \
  --epochs 5

echo ""
echo "=== 4. Generate plots ==="
python plot_results.py

echo ""
echo "Done! Check results/ and results/plots/ for output."
