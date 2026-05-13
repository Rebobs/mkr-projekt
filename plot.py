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
    # uloží graf do súboru a zatvorí figure, aby neunikala pamäť
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def best_per_config(results):
    # pre každú kombináciu (model, size, aug, train_frac) zachová len výsledok s najnižším test_mae
    # — eliminuje vplyv rôznych learning rates pri porovnávacích grafoch
    best = {}
    for r in results:
        key = (r['model'], r['input_size'], r['augmentation'], r.get('train_frac', 1.0))
        if key not in best or r['test_mae'] < best[key]['test_mae']:
            best[key] = r
    return list(best.values())


def plot_aug_comparison(results, size=224):
    # porovnanie MAE a RMSE bez augmentácie vs. s augmentáciou pre každý model
    subset = [r for r in results if r['input_size'] == size and r.get('train_frac', 1.0) == 1.0]
    model_names = list(dict.fromkeys(r['model'] for r in subset))
    x = np.arange(len(model_names))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, ylabel in [(axes[0], 'test_mae', 'MAE (W/m²)'),
                                (axes[1], 'test_rmse', 'RMSE (W/m²)')]:
        no_aug   = [next(r[metric] for r in subset if r['model']==m and not r['augmentation']) for m in model_names]
        with_aug = [next(r[metric] for r in subset if r['model']==m and     r['augmentation']) for m in model_names]

        b1 = ax.bar(x - w/2, no_aug,   w, label='Bez augmentácie',   color='#90CAF9', edgecolor='#1565C0')
        b2 = ax.bar(x + w/2, with_aug, w, label='S aug. (rot. 90°)', color='#A5D6A7', edgecolor='#2E7D32')

        for bar in list(b1) + list(b2):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in model_names], rotation=10)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel.split()[0]}: bez vs. s augmentáciou ({size}×{size})')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    save(fig, f'aug_comparison_{size}.png')


def plot_training_curves(results, size=224):
    # priebeh validačného MAE počas epoch pre každý model (plná čiara = aug, prerušovaná = bez)
    subset = [r for r in results if r['input_size'] == size and r.get('train_frac', 1.0) == 1.0]
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
    # scatter predikovaná vs. skutočná hodnota; ideálna predikcia leží na diagonále
    subset = [r for r in results if r['input_size'] == size and r.get('train_frac', 1.0) == 1.0]
    n      = len(subset)
    cols   = min(3, n)
    rows   = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
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
    # čiarový graf MAE podľa veľkosti vstupu — zvlášť pre bez aug a s aug
    subset = [r for r in results if r.get('train_frac', 1.0) == 1.0]
    sizes  = sorted({r['input_size'] for r in subset})
    if len(sizes) < 2:
        return

    model_names = list(dict.fromkeys(r['model'] for r in subset))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, aug, title in [(axes[0], False, 'Bez augmentácie'),
                           (axes[1], True,  'S augmentáciou')]:
        s = [r for r in subset if r['augmentation'] == aug]
        for m in model_names:
            maes = [next((r['test_mae'] for r in s if r['model']==m and r['input_size']==sz), None) for sz in sizes]
            ax.plot(sizes, maes, marker='o', label=LABELS[m], color=COLORS[m], linewidth=2)
        ax.set_xlabel('Veľkosť vstupu (px)')
        ax.set_ylabel('Test MAE (W/m²)')
        ax.set_title(f'Vplyv veľkosti vstupu — {title}')
        ax.set_xticks(sizes)
        ax.set_xticklabels([f'{s}×{s}' for s in sizes])
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, 'input_sizes_mae.png')


def plot_training_time(results, size=224):
    # celkový čas trénovania a číslo najlepšej epochy pre každý model
    subset = [r for r in results if r['input_size'] == size and r.get('train_frac', 1.0) == 1.0]
    labels = [f"{LABELS[r['model']]}\n({'aug' if r['augmentation'] else 'no aug'})" for r in subset]
    times  = [r['total_time_s'] for r in subset]
    epochs = [r.get('best_epoch', r.get('total_epochs', 0)) for r in subset]
    colors = [COLORS[r['model']] for r in subset]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for val, ylabel, title, ax in [
        (times,  'Čas (s)',   f'Celkový čas trénovania ({size}×{size})', axes[0]),
        (epochs, 'Epocha',   f'Najlepšia epocha ({size}×{size})',        axes[1]),
    ]:
        bars = ax.bar(labels, val, color=colors, edgecolor='white')
        for bar, v in zip(bars, val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(v), ha='center', va='bottom', fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(bottom=0, top=max(val) * 1.15)

    fig.tight_layout()
    save(fig, f'training_time_{size}.png')


def plot_complexity(results):
    # počet parametrov každého modelu (zobrazí sa raz, bez duplicít)
    seen, rows = set(), []
    for r in results:
        if r['model'] not in seen:
            seen.add(r['model'])
            rows.append(r)

    model_names = [r['model'] for r in rows]
    params      = [r['n_params'] for r in rows]
    colors      = [COLORS[m] for m in model_names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([LABELS[m] for m in model_names], params, color=colors, edgecolor='white')
    for bar, v in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f'{v/1e6:.1f}M', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Počet parametrov')
    ax.set_title('Komplexnosť modelov')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))
    ax.set_ylim(top=max(params) * 1.15)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    save(fig, 'model_complexity.png')


def plot_train_size_effect(results, size=224):
    # MAE v závislosti od počtu trénovacích vzoriek (25 / 50 / 75 / 100 %)
    fracs = sorted({r.get('train_frac', 1.0) for r in results})
    if len(fracs) < 2:
        return

    subset      = [r for r in results if r['input_size'] == size]
    model_names = list(dict.fromkeys(r['model'] for r in subset))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, aug, title in [(axes[0], False, 'Bez augmentácie'),
                           (axes[1], True,  'S augmentáciou')]:
        s = [r for r in subset if r['augmentation'] == aug]
        for m in model_names:
            n_trains = [next((r['n_train'] for r in s if r['model']==m and r.get('train_frac')==f), None) for f in fracs]
            maes     = [next((r['test_mae'] for r in s if r['model']==m and r.get('train_frac')==f), None) for f in fracs]
            ax.plot(n_trains, maes, marker='o', label=LABELS[m], color=COLORS[m], linewidth=2)
        ax.set_xlabel('Počet trénovacích vzoriek')
        ax.set_ylabel('Test MAE (W/m²)')
        ax.set_title(f'Vplyv počtu trénovacích vzoriek na MAE — {title}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, 'train_size_effect.png')


def plot_lr_effect(results, size=224):
    # MAE pre rôzne learning rates — ukazuje citlivosť modelov na voľbu LR
    subset = [r for r in results if r['input_size'] == size and r.get('train_frac', 1.0) == 1.0]
    lrs = sorted({r['lr'] for r in subset})
    if len(lrs) < 2:
        return

    model_names = list(dict.fromkeys(r['model'] for r in subset))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, aug, title in [(axes[0], False, 'Bez augmentácie'),
                           (axes[1], True,  'S augmentáciou')]:
        s = [r for r in subset if r['augmentation'] == aug]
        for m in model_names:
            maes = [next((r['test_mae'] for r in s if r['model'] == m and r['lr'] == lr), None) for lr in lrs]
            ax.plot([str(lr) for lr in lrs], maes, marker='o',
                    label=LABELS[m], color=COLORS[m], linewidth=2)
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Test MAE (W/m²)')
        ax.set_title(f'Vplyv learning rate — {title} ({size}×{size})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, f'lr_effect_{size}.png')


def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"Nenájdený súbor {RESULTS_FILE}. Najprv spusti experiment.py.")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    print(f"Načítaných {len(results)} výsledkov.")

    sizes        = sorted({r['input_size'] for r in results})
    default_size = 224 if 224 in sizes else sizes[0]
    best         = best_per_config(results)

    plot_aug_comparison(best,        size=default_size)
    plot_training_curves(best,       size=default_size)
    plot_pred_vs_actual(best,        size=default_size)
    plot_input_sizes(best)
    plot_training_time(best,         size=default_size)
    plot_complexity(best)
    plot_train_size_effect(best,     size=default_size)
    plot_lr_effect(results,          size=default_size)

    print(f"\nVšetky grafy uložené do {PLOTS_DIR}/")

if __name__ == '__main__':
    main()
