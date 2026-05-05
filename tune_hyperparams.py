"""
Hyperparameter tuning script.

Runs a grid search over lr × batch_size × optimizer for a given model and input size.
Defaults to the best model found in results_latest.json if no model is specified.

Usage:
  python tune_hyperparams.py
  python tune_hyperparams.py --model resnet18 --size 224
  python tune_hyperparams.py --model efficientnet_b0 --size 256 --augment
"""

import os
import sys
import json
import argparse
import itertools
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime

import config
from src.dataset import load_datasets
from src.models import get_model
from src.trainer import train_model, evaluate


def best_model_from_results():
    path = os.path.join(config.RESULTS_DIR, 'results_latest.json')
    if not os.path.exists(path):
        return config.MODELS[0], config.DEFAULT_INPUT_SIZE, True
    with open(path) as f:
        results = json.load(f)
    best = min(results, key=lambda r: r['test_mae'])
    print(f"Best model from previous run: {best['model']} "
          f"(size={best['input_size']}, aug={best['augmentation']}, MAE={best['test_mae']:.4f})")
    return best['model'], best['input_size'], best['augmentation']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=None, help='Model name (default: best from results_latest.json)')
    p.add_argument('--size', type=int, default=None)
    p.add_argument('--augment', action='store_true', default=None)
    p.add_argument('--epochs', type=int, default=config.DEFAULT_EPOCHS)
    p.add_argument('--no-cuda', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')
    print(f"Device: {device}")

    if args.model is None:
        model_name, input_size, augment = best_model_from_results()
    else:
        model_name = args.model
        input_size = args.size or config.DEFAULT_INPUT_SIZE
        augment = args.augment if args.augment is not None else True

    print(f"\nTuning: {model_name} | size={input_size} | aug={augment}")

    grid = list(itertools.product(
        config.LR_OPTIONS,
        config.BATCH_SIZE_OPTIONS,
        config.OPTIMIZER_OPTIONS,
    ))
    print(f"Grid size: {len(grid)} configurations\n")

    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    all_results = []

    for lr, batch_size, optimizer in grid:
        tag = f"tune_{model_name}_{input_size}_aug{int(augment)}_lr{lr}_bs{batch_size}_{optimizer}"
        ckpt = os.path.join(config.CHECKPOINTS_DIR, f"{tag}.pth")

        print(f"--- lr={lr} | bs={batch_size} | opt={optimizer} ---")

        train_loader, val_loader, test_loader, n_tr, n_va, n_te = load_datasets(
            config.DATA_DIR, input_size, augment, batch_size,
            val_split=config.VAL_SPLIT, test_split=config.TEST_SPLIT,
            seed=config.RANDOM_SEED,
        )

        model = get_model(model_name, pretrained=True).to(device)

        history = train_model(
            model, train_loader, val_loader,
            cfg={'lr': lr, 'optimizer': optimizer, 'epochs': args.epochs,
                 'patience': config.EARLY_STOPPING_PATIENCE},
            device=device,
            checkpoint_path=ckpt,
        )

        criterion = nn.MSELoss()
        test_metrics, _, _ = evaluate(model, test_loader, criterion, device)

        row = {
            'model': model_name,
            'input_size': input_size,
            'augmentation': augment,
            'lr': lr,
            'batch_size': batch_size,
            'optimizer': optimizer,
            'best_epoch': history['best_epoch'],
            'total_time_s': round(history['total_time'], 2),
            'test_mae': round(test_metrics['mae'], 4),
            'test_rmse': round(test_metrics['rmse'], 4),
            'test_r2': round(test_metrics['r2'], 6),
        }
        all_results.append(row)
        print(f"    MAE={row['test_mae']:.4f}  RMSE={row['test_rmse']:.4f}  R²={row['test_r2']:.6f}\n")

    df = pd.DataFrame(all_results).sort_values('test_mae')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(config.RESULTS_DIR, f'tuning_{model_name}_{ts}.csv')
    df.to_csv(csv_path, index=False)

    print("="*60)
    print("TUNING RESULTS (sorted by MAE)")
    print("="*60)
    print(df.to_string(index=False))
    print(f"\nBest config:")
    best = df.iloc[0]
    print(f"  lr={best['lr']}  batch_size={best['batch_size']}  optimizer={best['optimizer']}")
    print(f"  MAE={best['test_mae']:.4f}  RMSE={best['test_rmse']:.4f}  R²={best['test_r2']:.6f}")
    print(f"\nSaved → {csv_path}")


if __name__ == '__main__':
    main()
