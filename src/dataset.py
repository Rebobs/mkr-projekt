import os
import random
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class SkyIrradianceDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        irradiance = torch.tensor(float(row['irradiance']), dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, irradiance


class RandomRotation90:
    """Randomly rotate image by 0, 90, 180, or 270 degrees."""
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return transforms.functional.rotate(img, angle)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(input_size: int, augment: bool) -> transforms.Compose:
    ops = [transforms.Resize((input_size, input_size))]
    if augment:
        ops.append(RandomRotation90())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def get_eval_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def load_datasets(data_dir, input_size, augment, batch_size,
                  val_split=0.15, test_split=0.15, seed=42, num_workers=2):
    csv_path = os.path.join(data_dir, 'labels.csv')
    img_dir = os.path.join(data_dir, 'images')

    df = pd.read_csv(csv_path)

    train_val_df, test_df = train_test_split(df, test_size=test_split, random_state=seed)
    relative_val = val_split / (1.0 - test_split)
    train_df, val_df = train_test_split(train_val_df, test_size=relative_val, random_state=seed)

    train_ds = SkyIrradianceDataset(train_df, img_dir, get_train_transform(input_size, augment))
    val_ds = SkyIrradianceDataset(val_df, img_dir, get_eval_transform(input_size))
    test_ds = SkyIrradianceDataset(test_df, img_dir, get_eval_transform(input_size))

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, len(train_df), len(val_df), len(test_df)
