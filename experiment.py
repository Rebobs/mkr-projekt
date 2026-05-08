import os, json, time, random, argparse, shutil
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
MODELS      = ['resnet18', 'efficientnet_b0', 'mobilenet_v3_small']
INPUT_SIZES = [224, 256]
EPOCHS      = 20
BATCH_SIZE  = 32
LR          = 0.001
PATIENCE    = 7
SEED        = 42

# ── Dataset ───────────────────────────────────────────────────────────────────
class Rot90:
    def __call__(self, img):
        return transforms.functional.rotate(img, random.choice([0, 90, 180, 270]))

class SkyDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df      = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tf      = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row['filename'])).convert('RGB')
        return self.tf(img), torch.tensor(float(row['irradiance']), dtype=torch.float32)

def make_transform(size, augment=False):
    ops = [transforms.Resize((size, size))]
    if augment:
        ops.append(Rot90())
    ops += [transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(ops)

def get_loaders(size, augment):
    df      = pd.read_csv(os.path.join(DATA_DIR, 'labels.csv'))
    img_dir = os.path.join(DATA_DIR, 'images')

    train_df, tmp_df = train_test_split(df,      test_size=0.30, random_state=SEED)
    val_df,  test_df = train_test_split(tmp_df,  test_size=0.50, random_state=SEED)

    eval_tf = make_transform(size, augment=False)
    return (
        DataLoader(SkyDataset(train_df, img_dir, make_transform(size, augment)),
                   batch_size=BATCH_SIZE, shuffle=True,  num_workers=2),
        DataLoader(SkyDataset(val_df,   img_dir, eval_tf),
                   batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
        DataLoader(SkyDataset(test_df,  img_dir, eval_tf),
                   batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
    )

# ── Model ─────────────────────────────────────────────────────────────────────
def get_model(name):
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
def run(model_name, size, augment, epochs, device):
    print(f"\n{'='*55}")
    print(f"  {model_name} | {size}x{size} | aug={'yes' if augment else 'no'}")
    print(f"{'='*55}")

    train_loader, val_loader, test_loader = get_loaders(size, augment)
    model     = get_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_mae = float('inf')
    best_weights = None
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
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping (best epoch: {epoch - PATIENCE})")
                break

    model.load_state_dict(best_weights)
    test = evaluate(model, test_loader, device)
    print(f"\n  TEST → MAE={test['mae']:.4f}  RMSE={test['rmse']:.4f}  "
          f"R²={test['r2']:.6f}  time={time.time()-t0:.0f}s")

    return {
        'model':        model_name,
        'input_size':   size,
        'augmentation': augment,
        'n_params':     sum(p.numel() for p in model.parameters()),
        'total_time_s': round(time.time() - t0, 1),
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
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=MODELS)
    p.add_argument('--sizes',  nargs='+', type=int, default=INPUT_SIZES)
    p.add_argument('--epochs', type=int,  default=EPOCHS)
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Archivuj predchádzajúce výsledky ak existujú
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

    results = []
    for model_name in args.models:
        for size in args.sizes:
            for augment in [False, True]:
                results.append(run(model_name, size, augment, args.epochs, device))
                with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
                    json.dump(results, f, indent=2)

    skip = {'train_losses', 'val_maes', 'y_true', 'y_pred'}
    df   = pd.DataFrame([{k: v for k, v in r.items() if k not in skip} for r in results])
    df.to_csv(os.path.join(RESULTS_DIR, 'summary.csv'), index=False)

    print("\n" + "="*55 + "\nSUMMARY\n" + "="*55)
    print(df[['model', 'input_size', 'augmentation', 'test_mae', 'test_rmse', 'test_r2']].to_string(index=False))

if __name__ == '__main__':
    main()
