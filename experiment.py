import os, json, time, random, shutil
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = 'data'
RESULTS_DIR = 'results'
MODELS         = ['resnet18', 'efficientnet_b0', 'mobilenet_v3_small']
INPUT_SIZES    = [128, 224, 320]
LEARNING_RATES = [0.001, 0.0001, 0.01]
TRAIN_FRACS    = [0.25, 0.5, 0.75, 1.0]  # frakcie datasetu pre analýzu vplyvu počtu dát
EPOCHS         = 20
BATCH_SIZE     = 32
PATIENCE       = 7   # early stopping: počet epoch bez zlepšenia
SEED           = 42

# ── Dataset ───────────────────────────────────────────────────────────────────

class Rot90:
    # augmentácia: náhodne otočí obrázok o 0 / 90 / 180 / 270 stupňov
    def __call__(self, img):
        return transforms.functional.rotate(img, random.choice([0, 90, 180, 270]))


class SkyDataset(Dataset):
    # načíta obrázky z RAM cache a priradí im hodnoty žiarenia z DataFrame
    def __init__(self, df, img_dir, transform, cache):
        self.df     = df.reset_index(drop=True)
        self.tf     = transform
        self.imgs   = [cache[row['filename']] for _, row in self.df.iterrows()]
        self.labels = [float(row['irradiance']) for _, row in self.df.iterrows()]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.tf(self.imgs[idx]), torch.tensor(self.labels[idx], dtype=torch.float32)


def make_transform(size, augment=False):
    # zostaví pipeline: resize → (Rot90 ak augment) → tenzor → ImageNet normalizácia
    ops = [transforms.Resize((size, size))]
    if augment:
        ops.append(Rot90())
    ops += [transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(ops)


def load_image_cache(img_dir, filenames):
    # prednahrá všetky obrázky do RAM a zmenší ich na max. veľkosť vstupu,
    # aby sa eliminovalo opakovné čítanie z disku počas trénovania
    max_size = max(INPUT_SIZES)
    print(f"  Načítavam obrázky do RAM (pre-resize na {max_size}px)...")
    return {
        fn: Image.open(os.path.join(img_dir, fn)).convert('RGB').resize(
            (max_size, max_size), Image.BILINEAR)
        for fn in filenames
    }


def get_loaders(size, augment, cache, train_frac=1.0):
    # rozdelí dataset 70/15/15, augmentáciu aplikuje len na tréning
    df = pd.read_csv(os.path.join(DATA_DIR, 'labels.csv'))

    train_df, tmp_df = train_test_split(df,     test_size=0.30, random_state=SEED)
    val_df,  test_df = train_test_split(tmp_df, test_size=0.50, random_state=SEED)

    if train_frac < 1.0:
        train_df = train_df.sample(frac=train_frac, random_state=SEED)

    eval_tf = make_transform(size, augment=False)
    kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    return (
        DataLoader(SkyDataset(train_df, None, make_transform(size, augment), cache),
                   shuffle=True,  **kw),
        DataLoader(SkyDataset(val_df,   None, eval_tf,                       cache),
                   shuffle=False, **kw),
        DataLoader(SkyDataset(test_df,  None, eval_tf,                       cache),
                   shuffle=False, **kw),
        len(train_df),
    )

# ── Model ─────────────────────────────────────────────────────────────────────

def get_model(name):
    # načíta predtrénovanú sieť z ImageNetu a nahradí poslednú vrstvu
    # jedným neurónom pre regresiu (priama predikcia hodnoty žiarenia)
    if name == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    elif name == 'mobilenet_v3_small':
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
    return m

# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    # jedna trénovacia epocha: forward → loss → backprop → krok optimizera
    model.train()
    total = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    # vyhodnotí model na datasete a vráti MAE, RMSE, R² + samotné predikcie
    model.eval()
    preds, targets = [], []
    for imgs, labels in loader:
        preds.extend(model(imgs.to(device)).squeeze(1).cpu().numpy())
        targets.extend(labels.numpy())
    y_true, y_pred = np.array(targets), np.array(preds)
    return {
        'mae':    float(np.mean(np.abs(y_true - y_pred))),
        'rmse':   float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        'r2':     float(r2_score(y_true, y_pred)),
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
    }

# ── Single experiment ─────────────────────────────────────────────────────────

def run(model_name, size, augment, epochs, lr, device, cache, train_frac=1.0):
    # spustí jeden experiment: tréning s early stoppingom, potom testovanie
    # na konci obnoví váhy z najlepšej epochy (podľa val MAE)
    print(f"\n{'='*60}")
    print(f"  {model_name} | {size}x{size} | aug={'yes' if augment else 'no'} | "
          f"lr={lr} | train_frac={train_frac:.0%}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader, n_train = get_loaders(size, augment, cache, train_frac)
    print(f"  Trénovacích vzoriek: {n_train}")

    model     = get_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_mae = float('inf')
    best_weights = None
    best_epoch   = 0
    no_improve   = 0
    train_losses, val_maes = [], []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        loss    = train_epoch(model, train_loader, optimizer, criterion, device)
        val     = evaluate(model, val_loader, device)
        train_losses.append(loss)
        val_maes.append(val['mae'])

        improved = val['mae'] < best_val_mae
        print(f"  {'*' if improved else ' '} Ep {epoch:3d} | "
              f"loss={loss:.4f} | MAE={val['mae']:.2f} | "
              f"RMSE={val['rmse']:.2f} | R²={val['r2']:.4f}")

        if improved:
            best_val_mae = val['mae']
            best_epoch   = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping (best epoch: {best_epoch})")
                break

    model.load_state_dict(best_weights)
    test = evaluate(model, test_loader, device)
    total_time = round(time.time() - t0, 1)
    print(f"\n  TEST → MAE={test['mae']:.4f}  RMSE={test['rmse']:.4f}  "
          f"R²={test['r2']:.6f}  time={total_time}s")

    return {
        'model':        model_name,
        'input_size':   size,
        'augmentation': augment,
        'lr':           lr,
        'train_frac':   train_frac,
        'n_train':      n_train,
        'n_params':     sum(p.numel() for p in model.parameters()),
        'best_epoch':   best_epoch,
        'total_epochs': len(train_losses),
        'total_time_s': total_time,
        'test_mae':     round(test['mae'],  4),
        'test_rmse':    round(test['rmse'], 4),
        'test_r2':      round(test['r2'],   6),
        'train_losses': train_losses,
        'val_maes':     val_maes,
        'y_true':       test['y_true'],
        'y_pred':       test['y_pred'],
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # archivuje predchádzajúce výsledky, prednahrá cache a spustí všetky experimenty;
    # výsledky sa ukladajú priebežne, aby neboli stratené pri prerušení
    os.makedirs(RESULTS_DIR, exist_ok=True)

    existing = os.path.join(RESULTS_DIR, 'results.json')
    if os.path.exists(existing):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_dir = os.path.join(RESULTS_DIR, 'archive', ts)
        os.makedirs(archive_dir, exist_ok=True)
        for f in ['results.json', 'summary.csv']:
            src = os.path.join(RESULTS_DIR, f)
            if os.path.exists(src):
                shutil.move(src, os.path.join(archive_dir, f))
        print(f"Predchádzajúce výsledky archivované → results/archive/{ts}/")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    df      = pd.read_csv(os.path.join(DATA_DIR, 'labels.csv'))
    img_dir = os.path.join(DATA_DIR, 'images')
    cache   = load_image_cache(img_dir, df['filename'].tolist())

    results = []
    for lr in LEARNING_RATES:
        for train_frac in TRAIN_FRACS:
            for model_name in MODELS:
                for size in INPUT_SIZES:
                    for augment in [False, True]:
                        results.append(run(
                            model_name, size, augment, EPOCHS,
                            lr, device, cache, train_frac,
                        ))
                    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
                        json.dump(results, f, indent=2)

    # uloženie súhrnu bez veľkých polí (predikcie, histórie strát)
    skip = {'train_losses', 'val_maes', 'y_true', 'y_pred'}
    df   = pd.DataFrame([{k: v for k, v in r.items() if k not in skip} for r in results])
    df.to_csv(os.path.join(RESULTS_DIR, 'summary.csv'), index=False)

    print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
    print(df[['model', 'input_size', 'augmentation', 'lr',
              'n_train', 'best_epoch', 'total_time_s',
              'test_mae', 'test_rmse', 'test_r2']].to_string(index=False))

if __name__ == '__main__':
    main()
