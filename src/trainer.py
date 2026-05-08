import time
import numpy as np
import torch
import torch.nn as nn
from .metrics import compute_metrics


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, targets_all = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(1)
        out = model(images)
        total_loss += criterion(out, targets).item() * images.size(0)
        preds.extend(out.squeeze(1).cpu().numpy())
        targets_all.extend(targets.squeeze(1).cpu().numpy())

    y_true = np.array(targets_all)
    y_pred = np.array(preds)
    metrics = compute_metrics(y_true, y_pred)
    metrics['loss'] = total_loss / len(loader.dataset)
    return metrics, y_true, y_pred


def train_model(model, train_loader, val_loader, cfg, device, checkpoint_path=None):
    """
    cfg keys: lr, optimizer ('adam'|'adamw'), epochs, patience
    Returns history dict.
    """
    lr = cfg.get('lr', 0.001)
    opt_name = cfg.get('optimizer', 'adam')
    epochs = cfg.get('epochs', 30)
    patience = cfg.get('patience', 10)

    OptCls = torch.optim.AdamW if opt_name == 'adamw' else torch.optim.Adam
    optimizer = OptCls(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_epoch = 0
    no_improve = 0

    history = {
        'train_loss': [], 'val_loss': [],
        'val_mae': [], 'val_rmse': [], 'val_r2': [],
        'epoch_times': [],
    }

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - t0

        scheduler.step(val_metrics['loss'])

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_r2'].append(val_metrics['r2'])
        history['epoch_times'].append(epoch_time)

        improved = val_metrics['loss'] < best_val_loss
        tag = '*' if improved else ' '
        print(f"  [{tag}] Ep {epoch:3d}/{epochs} | "
              f"TrainL={train_loss:.4f} | "
              f"ValMAE={val_metrics['mae']:.2f} | "
              f"ValRMSE={val_metrics['rmse']:.2f} | "
              f"R²={val_metrics['r2']:.4f} | "
              f"{epoch_time:.1f}s")

        if improved:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            no_improve = 0
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
                break

    history['total_time'] = time.time() - t_start
    history['best_epoch'] = best_epoch
    history['avg_epoch_time'] = float(np.mean(history['epoch_times']))

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    return history
