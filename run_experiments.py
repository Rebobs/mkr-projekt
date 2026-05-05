"""
Main experiment script.

Runs a grid over:
  - models:      resnet18, efficientnet_b0, mobilenet_v3_small
  - augmentation: False, True
  - input sizes:  128, 224, 256, 320

Results are saved to results/results_latest.json and results/summary_latest.csv.
Use --sizes to run only a subset (e.g. --sizes 224 256).
Use --models to run only selected models.
Use --epochs to override default epoch count.
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime

import config
from src.dataset import load_datasets
from src.models import get_model, count_parameters, get_model_size_mb
from src.trainer import train_model, evaluate


def measure_inference_ms(model, loader, device, n_runs=100):
    model.eval()
    sample = next(iter(loader))[0][:1].to(device)
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(sample)
        return (time.perf_counter() - t0) / n_runs * 1000


def run_one(model_name, input_size, augment, lr, batch_size, optimizer, epochs, device):
    tag = f"{model_name}_{input_size}_aug{int(augment)}_lr{lr}_bs{batch_size}"
    ckpt = os.path.join(config.CHECKPOINTS_DIR, f"{tag}.pth")

    print(f"\n{'='*65}")
    print(f"  {model_name} | {input_size}x{input_size} | aug={'yes' if augment else 'no'} "
          f"| lr={lr} | bs={batch_size} | opt={optimizer}")
    print(f"{'='*65}")

    train_loader, val_loader, test_loader, n_tr, n_va, n_te = load_datasets(
        config.DATA_DIR, input_size, augment, batch_size,
        val_split=config.VAL_SPLIT, test_split=config.TEST_SPLIT,
        seed=config.RANDOM_SEED,
    )
    print(f"  Samples — train: {n_tr} | val: {n_va} | test: {n_te}")

    model = get_model(model_name, pretrained=True).to(device)
    n_params = count_parameters(model)
    size_mb = get_model_size_mb(model)

    history = train_model(
        model, train_loader, val_loader,
        cfg={'lr': lr, 'optimizer': optimizer, 'epochs': epochs,
             'patience': config.EARLY_STOPPING_PATIENCE},
        device=device,
        checkpoint_path=ckpt,
    )

    criterion = nn.MSELoss()
    test_metrics, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    inf_ms = measure_inference_ms(model, test_loader, device)

    print(f"\n  TEST → MAE={test_metrics['mae']:.4f} W/m²  "
          f"RMSE={test_metrics['rmse']:.4f} W/m²  R²={test_metrics['r2']:.6f}")

    return {
        'model': model_name,
        'input_size': input_size,
        'augmentation': augment,
        'lr': lr,
        'batch_size': batch_size,
        'optimizer': optimizer,
        'n_params': n_params,
        'model_size_mb': round(size_mb, 2),
        'best_epoch': history['best_epoch'],
        'total_epochs_run': len(history['train_loss']),
        'total_time_s': round(history['total_time'], 2),
        'avg_epoch_time_s': round(history['avg_epoch_time'], 2),
        'inference_time_ms': round(inf_ms, 3),
        'test_mae': round(test_metrics['mae'], 4),
        'test_rmse': round(test_metrics['rmse'], 4),
        'test_r2': round(test_metrics['r2'], 6),
        'train_loss_history': history['train_loss'],
        'val_loss_history': history['val_loss'],
        'val_mae_history': history['val_mae'],
        'val_rmse_history': history['val_rmse'],
        'val_r2_history': history['val_r2'],
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
    }


def save_results(all_results, suffix='latest'):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    json_path = os.path.join(config.RESULTS_DIR, f'results_{suffix}.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    skip = {'train_loss_history', 'val_loss_history', 'val_mae_history',
            'val_rmse_history', 'val_r2_history', 'y_true', 'y_pred'}
    summary = [{k: v for k, v in r.items() if k not in skip} for r in all_results]
    csv_path = os.path.join(config.RESULTS_DIR, f'summary_{suffix}.csv')
    pd.DataFrame(summary).to_csv(csv_path, index=False)

    print(f"\nSaved → {json_path}")
    print(f"Saved → {csv_path}")
    return json_path, csv_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=config.MODELS)
    p.add_argument('--sizes', nargs='+', type=int, default=config.INPUT_SIZES)
    p.add_argument('--epochs', type=int, default=config.DEFAULT_EPOCHS)
    p.add_argument('--lr', type=float, default=config.DEFAULT_LR)
    p.add_argument('--batch-size', type=int, default=config.DEFAULT_BATCH_SIZE)
    p.add_argument('--optimizer', default=config.DEFAULT_OPTIMIZER)
    p.add_argument('--no-cuda', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')
    print(f"Device: {device}")

    for d in (config.RESULTS_DIR, config.CHECKPOINTS_DIR, config.PLOTS_DIR, config.LOGS_DIR):
        os.makedirs(d, exist_ok=True)

    all_results = []

    for model_name in args.models:
        for input_size in args.sizes:
            for augment in config.AUGMENTATIONS:
                try:
                    result = run_one(
                        model_name=model_name,
                        input_size=input_size,
                        augment=augment,
                        lr=args.lr,
                        batch_size=args.batch_size,
                        optimizer=args.optimizer,
                        epochs=args.epochs,
                        device=device,
                    )
                    all_results.append(result)
                    # Save incrementally so progress is not lost
                    save_results(all_results, suffix='latest')
                except Exception as e:
                    print(f"  ERROR in {model_name}/{input_size}/aug={augment}: {e}", file=sys.stderr)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_results(all_results, suffix=ts)

    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ('train_loss_history', 'val_loss_history',
                                     'val_mae_history', 'val_rmse_history',
                                     'val_r2_history', 'y_true', 'y_pred')}
                       for r in all_results])
    print("\n" + "="*65)
    print("FINAL SUMMARY")
    print("="*65)
    cols = ['model', 'input_size', 'augmentation', 'test_mae', 'test_rmse', 'test_r2',
            'total_time_s', 'n_params']
    print(df[cols].to_string(index=False))


if __name__ == '__main__':
    main()
