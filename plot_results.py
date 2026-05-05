"""
Generate all result plots from results_latest.json.

Usage:
  python plot_results.py
  python plot_results.py --results results/results_20240101_120000.json
  python plot_results.py --out results/plots/custom/
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

import config

COLORS = {
    'resnet18': '#2196F3',
    'efficientnet_b0': '#4CAF50',
    'mobilenet_v3_small': '#FF9800',
}
MODEL_LABELS = {
    'resnet18': 'ResNet18',
    'efficientnet_b0': 'EfficientNet-B0',
    'mobilenet_v3_small': 'MobileNetV3-Small',
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def savefig(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 1: Training / validation loss curves ─────────────────────────────────
def plot_loss_curves(results, out_dir, input_size=224):
    subset = [r for r in results if r['input_size'] == input_size]
    n = len(subset)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4))
    axes = axes.flatten()

    for i, r in enumerate(subset):
        ax = axes[i]
        epochs = range(1, len(r['train_loss_history']) + 1)
        ax.plot(epochs, r['train_loss_history'], label='Train loss', color='steelblue')
        ax.plot(epochs, r['val_loss_history'], label='Val loss', color='tomato')
        aug_str = 'aug' if r['augmentation'] else 'no aug'
        ax.set_title(f"{MODEL_LABELS[r['model']]} ({aug_str})", fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Training and Validation Loss Curves ({input_size}×{input_size})', fontsize=13, y=1.01)
    fig.tight_layout()
    savefig(fig, out_dir, f'loss_curves_{input_size}.png')


# ── Plot 2: MAE comparison — with vs. without augmentation ───────────────────
def plot_aug_comparison(results, out_dir, metric='test_mae', input_size=224):
    df = pd.DataFrame(results)
    df = df[df['input_size'] == input_size]

    models = config.MODELS
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    no_aug = df[~df['augmentation']][metric].values
    with_aug = df[df['augmentation']][metric].values

    bars1 = ax.bar(x - width / 2, no_aug, width, label='Bez augmentácie',
                   color='#90CAF9', edgecolor='#1565C0', linewidth=0.8)
    bars2 = ax.bar(x + width / 2, with_aug, width, label='S rotačnou augmentáciou',
                   color='#A5D6A7', edgecolor='#2E7D32', linewidth=0.8)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f'{h:.2f}',
                ha='center', va='bottom', fontsize=8)

    ylabel = 'MAE (W/m²)' if 'mae' in metric else 'RMSE (W/m²)'
    title_metric = 'MAE' if 'mae' in metric else 'RMSE'
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title_metric}: bez augmentácie vs. s rotáciami ({input_size}×{input_size})')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=10)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    savefig(fig, out_dir, f'aug_comparison_{metric}_{input_size}.png')


# ── Plot 3: Input size vs. MAE heatmap ───────────────────────────────────────
def plot_inputsize_comparison(results, out_dir, augment=True):
    df = pd.DataFrame(results)
    df = df[df['augmentation'] == augment]
    pivot = df.pivot_table(index='model', columns='input_size', values='test_mae')
    pivot.index = [MODEL_LABELS.get(m, m) for m in pivot.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd_r')
    plt.colorbar(im, ax=ax, label='MAE (W/m²)')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{s}×{s}" for s in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9)

    aug_str = 'S augmentáciou' if augment else 'Bez augmentácie'
    ax.set_title(f'MAE (W/m²) podľa veľkosti vstupu — {aug_str}')
    ax.set_xlabel('Veľkosť vstupu')
    fig.tight_layout()
    savefig(fig, out_dir, f'inputsize_mae_aug{int(augment)}.png')


# ── Plot 4: Training time comparison ─────────────────────────────────────────
def plot_training_time(results, out_dir, input_size=224):
    df = pd.DataFrame(results)
    df = df[df['input_size'] == input_size].copy()
    df['label'] = df.apply(
        lambda r: f"{MODEL_LABELS[r['model']]}\n({'aug' if r['augmentation'] else 'no aug'})", axis=1
    )
    df = df.sort_values(['model', 'augmentation'])

    colors = [COLORS[r] for r in df['model']]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df['label'], df['total_time_s'], color=colors, edgecolor='white', linewidth=0.8)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f'{h:.0f}s',
                ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Celkový čas trénovania (s)')
    ax.set_title(f'Čas trénovania ({input_size}×{input_size})')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    savefig(fig, out_dir, f'training_time_{input_size}.png')


# ── Plot 5: Predicted vs. actual scatter ─────────────────────────────────────
def plot_pred_vs_actual(results, out_dir, input_size=224, augment=True):
    subset = [r for r in results
              if r['input_size'] == input_size and r['augmentation'] == augment]

    cols = min(3, len(subset))
    rows = (len(subset) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, r in enumerate(subset):
        ax = axes[i]
        y_true = np.array(r['y_true'])
        y_pred = np.array(r['y_pred'])
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color=COLORS[r['model']])
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
        ax.set_xlabel('Skutočná hodnota (W/m²)')
        ax.set_ylabel('Predikovaná hodnota (W/m²)')
        aug_str = 'aug' if r['augmentation'] else 'no aug'
        ax.set_title(f"{MODEL_LABELS[r['model']]} ({aug_str})\n"
                     f"MAE={r['test_mae']:.2f}, R²={r['test_r2']:.4f}", fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    aug_str = 'S augmentáciou' if augment else 'Bez augmentácie'
    fig.suptitle(f'Predikovaná vs. skutočná hodnota — {aug_str} ({input_size}×{input_size})',
                 fontsize=12, y=1.01)
    fig.tight_layout()
    savefig(fig, out_dir, f'pred_vs_actual_{input_size}_aug{int(augment)}.png')


# ── Plot 6: Val MAE during training ──────────────────────────────────────────
def plot_val_mae_curves(results, out_dir, input_size=224):
    subset = [r for r in results if r['input_size'] == input_size]
    fig, ax = plt.subplots(figsize=(10, 5))

    linestyles = {False: '--', True: '-'}
    for r in subset:
        history = r.get('val_mae_history', [])
        if not history:
            continue
        label = f"{MODEL_LABELS[r['model']]} ({'aug' if r['augmentation'] else 'no aug'})"
        ax.plot(range(1, len(history) + 1), history,
                label=label,
                color=COLORS[r['model']],
                linestyle=linestyles[r['augmentation']],
                linewidth=1.8)

    ax.set_xlabel('Epocha')
    ax.set_ylabel('Val MAE (W/m²)')
    ax.set_title(f'Validačné MAE počas trénovania ({input_size}×{input_size})')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, out_dir, f'val_mae_curves_{input_size}.png')


# ── Plot 7: Model complexity comparison ──────────────────────────────────────
def plot_complexity(results, out_dir):
    df = pd.DataFrame(results)
    df = df.drop_duplicates('model')[['model', 'n_params', 'model_size_mb', 'inference_time_ms']]
    df['label'] = df['model'].map(MODEL_LABELS)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, col, ylabel, title in [
        (axes[0], 'n_params', 'Počet parametrov', 'Počet parametrov modelu'),
        (axes[1], 'model_size_mb', 'Veľkosť (MB)', 'Veľkosť modelu (MB)'),
        (axes[2], 'inference_time_ms', 'Čas (ms)', 'Čas inferencie (ms/obr.)'),
    ]:
        colors = [COLORS[m] for m in df['model']]
        bars = ax.bar(df['label'], df[col], color=colors, edgecolor='white')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                    f'{h:,.0f}' if col == 'n_params' else f'{h:.2f}',
                    ha='center', va='bottom', fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticklabels(df['label'], rotation=10, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f'{x/1e6:.1f}M' if col == 'n_params' else f'{x:.2f}'
        ))

    fig.tight_layout()
    savefig(fig, out_dir, 'model_complexity.png')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default=os.path.join(config.RESULTS_DIR, 'results_latest.json'))
    p.add_argument('--out', default=config.PLOTS_DIR)
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.results):
        print(f"Results file not found: {args.results}")
        print("Run run_experiments.py first.")
        return

    os.makedirs(args.out, exist_ok=True)
    results = load_results(args.results)
    print(f"Loaded {len(results)} experiment results from {args.results}")

    sizes_present = sorted({r['input_size'] for r in results})
    default_size = 224 if 224 in sizes_present else sizes_present[0]

    print("\nGenerating plots...")
    plot_loss_curves(results, args.out, input_size=default_size)
    plot_aug_comparison(results, args.out, metric='test_mae', input_size=default_size)
    plot_aug_comparison(results, args.out, metric='test_rmse', input_size=default_size)
    plot_val_mae_curves(results, args.out, input_size=default_size)
    plot_training_time(results, args.out, input_size=default_size)
    plot_pred_vs_actual(results, args.out, input_size=default_size, augment=False)
    plot_pred_vs_actual(results, args.out, input_size=default_size, augment=True)
    plot_complexity(results, args.out)

    if len(sizes_present) > 1:
        plot_inputsize_comparison(results, args.out, augment=True)
        plot_inputsize_comparison(results, args.out, augment=False)

    print(f"\nAll plots saved to {args.out}/")


if __name__ == '__main__':
    main()
