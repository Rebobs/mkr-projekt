"""
Generate a synthetic sky-irradiance dataset for testing.

Each image is a procedurally generated sky gradient. The irradiance label
is derived from sky brightness + cloud coverage so the model can learn
a meaningful signal.

Usage:
  python generate_synthetic_data.py              # 300 images
  python generate_synthetic_data.py --n 500
  python generate_synthetic_data.py --n 200 --size 128 --out data
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image


def _sky_image(size: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """
    Returns (RGB image as uint8 array of shape [size, size, 3], irradiance W/m²).
    Sky type is randomly chosen; irradiance is correlated with clarity.
    """
    sky_type = rng.choice(['clear', 'partly_cloudy', 'overcast'], p=[0.35, 0.40, 0.25])

    H, W = size, size
    # Sky blue gradient — brighter at bottom horizon
    y = np.linspace(0.6, 1.0, H).reshape(H, 1)
    brightness = rng.uniform(0.5, 1.0)

    if sky_type == 'clear':
        R = (np.ones((H, W)) * 80 * brightness * y).clip(0, 255)
        G = (np.ones((H, W)) * 140 * brightness * y).clip(0, 255)
        B = (np.ones((H, W)) * 230 * brightness * y).clip(0, 255)
        cloud_frac = rng.uniform(0.0, 0.1)
        base_irr = rng.uniform(700, 1000)

    elif sky_type == 'partly_cloudy':
        R = (np.ones((H, W)) * 120 * brightness * y).clip(0, 255)
        G = (np.ones((H, W)) * 160 * brightness * y).clip(0, 255)
        B = (np.ones((H, W)) * 200 * brightness * y).clip(0, 255)
        cloud_frac = rng.uniform(0.2, 0.6)
        base_irr = rng.uniform(300, 700)

    else:  # overcast
        grey = (np.ones((H, W)) * 190 * brightness * y).clip(0, 255)
        R = G = B = grey
        cloud_frac = rng.uniform(0.6, 0.95)
        base_irr = rng.uniform(30, 300)

    # Add cloud patches (white-ish blobs)
    cloud_mask = rng.random((H, W)) < cloud_frac * 0.5
    R = np.where(cloud_mask, np.clip(R + rng.uniform(50, 80), 0, 255), R)
    G = np.where(cloud_mask, np.clip(G + rng.uniform(50, 80), 0, 255), G)
    B = np.where(cloud_mask, np.clip(B + rng.uniform(50, 80), 0, 255), B)

    # Sun disk (random position, only for clear/partly cloudy)
    if sky_type != 'overcast' and rng.random() < 0.5:
        cx = rng.integers(W // 4, 3 * W // 4)
        cy = rng.integers(H // 4, 3 * H // 4)
        radius = max(3, size // 20)
        yy, xx = np.ogrid[:H, :W]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        R = np.where(disk, 255, R)
        G = np.where(disk, 240, G)
        B = np.where(disk, 180, B)

    img = np.stack([R, G, B], axis=2).astype(np.uint8)

    # Irradiance: base + small Gaussian noise
    irradiance = float(np.clip(base_irr + rng.normal(0, 20), 0, 1100))
    return img, irradiance


def generate(n: int, size: int, out_dir: str, seed: int = 42):
    img_dir = os.path.join(out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    records = []

    for i in range(n):
        arr, irradiance = _sky_image(size, rng)
        filename = f'img_{i:05d}.jpg'
        Image.fromarray(arr).save(os.path.join(img_dir, filename), quality=90)
        records.append({'filename': filename, 'irradiance': round(irradiance, 2)})

        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{n} images...")

    csv_path = os.path.join(out_dir, 'labels.csv')
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"\nDone. {n} images → {img_dir}/")
    print(f"Labels → {csv_path}")
    print(f"Irradiance range: {min(r['irradiance'] for r in records):.1f} – "
          f"{max(r['irradiance'] for r in records):.1f} W/m²")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=300, help='Number of images (default: 300)')
    p.add_argument('--size', type=int, default=64, help='Image size in pixels (default: 64)')
    p.add_argument('--out', default='data', help='Output directory (default: data)')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"Generating {args.n} synthetic sky images ({args.size}×{args.size}) → {args.out}/")
    generate(args.n, args.size, args.out, args.seed)
