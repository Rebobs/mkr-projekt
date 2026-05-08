"""
Generates plots from results/results.json.

Usage:
  python plot.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_FILE = 'results/results.json'
PLOTS_DIR    = 'results/plots'

COLORS = {'resnet18': '#2196F3', 'efficientnet_b0': '#4CAF50', 'mobilenet_v3_small': '#FF9800'}
LABELS = {'resnet18': 'ResNet18', 'efficientnet_b0': 'EfficientNet-B0', 'mobilenet_v3_small': 'MobileNetV3-S'}

def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_aug_comparison(results, size=224):
    """Bar chart: MAE a RMSE bez aug vs. s augmentáciou."""
    subset = [r for r in results if r['input_size'] == size]
    model_names = list(dict.fromkeys(r['model'] for r in subset))
    x = np.arange(len(model_names))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, ylabel in [(axes[0], 'test_mae', 'MAE (W/m²)'),
                                (axes[1], 'test_rmse', 'RMSE (W/m²)')]:
        no_aug   = [next(r[metric] for r in subset if r['model']==m and not r['augmentation']) for m in model_names]
        with_aug = [next(r[metric] for r in subset if r['model']==m and     r['augmentation']) for m in model_names]

        b1 = ax.bar(x - w/2, no_aug,   w, label='Bez augmentácie',    color='#90CAF9', edgecolor='#1565C0')
        b2 = ax.bar(x + w/2, with_aug, w, label='S aug. (rot. 90°)', color='#A5D6A7', edgecolor='#2E7D32')

        for bar in list(b1) + list(b2):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in model_names], rotation=10)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel.split()[0]}: bez vs. s augmentáciou ({size}×{size})')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    save(fig, f'aug_comparison_{size}.png')

def plot_training_curves(results, size=224):
    """MAE na validačnej množine počas trénovania."""
    subset = [r for r in results if r['input_size'] == size]
    fig, ax = plt.subplots(figsize=(10, 5))

    for r in subset:
        label = f"{LABELS[r['model']]} ({'aug' if r['augmentation'] else 'no aug'})"
        ls    = '-' if r['augmentation'] else '--'
        ax.plot(range(1, len(r['val_maes']) + 1), r['val_maes'],
                label=label, color=COLORS[r['model']], linestyle=ls, linewidth=1.8)

    ax.set_xlabel('Epocha')
    ax.set_ylabel('Val MAE (W/m²)')
    ax.set_title(f'Validačné MAE počas trénovania ({size}×{size})')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, f'training_curves_{size}.png')

def plot_pred_vs_actual(results, size=224):
    """Scatter: predikovaná vs. skutočná hodnota."""
    subset = [r for r in results if r['input_size'] == size]
    n      = len(subset)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(5 * ((n + 1) // 2), 9))
    axes = np.array(axes).flatten()

    for i, r in enumerate(subset):
        ax = axes[i]
        y_true = np.array(r['y_true'])
        y_pred = np.array(r['y_pred'])
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color=COLORS[r['model']])
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
        aug_str = 'aug' if r['augmentation'] else 'no aug'
        ax.set_title(f"{LABELS[r['model']]} ({aug_str})\nMAE={r['test_mae']:.2f}, R²={r['test_r2']:.4f}", fontsize=9)
        ax.set_xlabel('Skutočná (W/m²)')
        ax.set_ylabel('Predikovaná (W/m²)')
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Predikovaná vs. skutočná hodnota ({size}×{size})', fontsize=12)
    fig.tight_layout()
    save(fig, f'pred_vs_actual_{size}.png')

def plot_input_sizes(results):
    """MAE pre každý model podľa veľkosti vstupu (s augmentáciou)."""
    subset = [r for r in results if r['augmentation']]
    if not subset:
        return
    model_names = list(dict.fromkeys(r['model'] for r in subset))
    sizes       = sorted({r['input_size'] for r in subset})

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name in model_names:
        maes = [next((r['test_mae'] for r in subset
                      if r['model'] == model_name and r['input_size'] == s), None) for s in sizes]
        ax.plot(sizes, maes, marker='o', label=LABELS[model_name], color=COLORS[model_name], linewidth=2)

    ax.set_xlabel('Veľkosť vstupu (px)')
    ax.set_ylabel('Test MAE (W/m²)')
    ax.set_title('Vplyv veľkosti vstupu na MAE (s augmentáciou)')
    ax.set_xticks(sizes)
    ax.set_xticklabels([f'{s}×{s}' for s in sizes])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, 'input_sizes_mae.png')

def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"Nenájdený súbor {RESULTS_FILE}. Najprv spusti experiment.py.")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    print(f"Načítaných {len(results)} výsledkov.")

    sizes = sorted({r['input_size'] for r in results})
    default_size = 224 if 224 in sizes else sizes[0]

    plot_aug_comparison(results,   size=default_size)
    plot_training_curves(results,  size=default_size)
    plot_pred_vs_actual(results,   size=default_size)
    plot_input_sizes(results)

    print(f"\nVšetky grafy uložené do {PLOTS_DIR}/")

if __name__ == '__main__':
    main()
