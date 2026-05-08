"""
Vygeneruje syntetický dataset oblohy na testovanie.

Usage:
  python generate_data.py          # 300 obrázkov
  python generate_data.py --n 500
"""

import os, argparse
import numpy as np
import pandas as pd
from PIL import Image

def make_sky_image(size, rng):
    sky_type = rng.choice(['clear', 'partly_cloudy', 'overcast'], p=[0.35, 0.40, 0.25])
    H, W     = size, size
    bright   = rng.uniform(0.5, 1.0)
    y        = np.linspace(0.6, 1.0, H).reshape(H, 1) * np.ones((H, W))

    if sky_type == 'clear':
        R, G, B = 80*bright*y, 140*bright*y, 230*bright*y
        base_irr = rng.uniform(700, 1000)
        cloud_p  = 0.05
    elif sky_type == 'partly_cloudy':
        R, G, B = 120*bright*y, 160*bright*y, 200*bright*y
        base_irr = rng.uniform(300, 700)
        cloud_p  = 0.25
    else:
        grey     = 190 * bright * y
        R, G, B  = grey, grey, grey
        base_irr = rng.uniform(30, 300)
        cloud_p  = 0.55

    cloud = rng.random((H, W)) < cloud_p
    R = np.where(cloud, np.clip(R + 70, 0, 255), R)
    G = np.where(cloud, np.clip(G + 70, 0, 255), G)
    B = np.where(cloud, np.clip(B + 70, 0, 255), B)

    irradiance = float(np.clip(base_irr + rng.normal(0, 20), 0, 1100))
    img = np.stack([R, G, B], axis=2).clip(0, 255).astype(np.uint8)
    return img, irradiance

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n',    type=int, default=300)
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--out',  default='data')
    args = p.parse_args()

    img_dir = os.path.join(args.out, 'images')
    os.makedirs(img_dir, exist_ok=True)

    rng     = np.random.default_rng(42)
    records = []
    for i in range(args.n):
        arr, irr = make_sky_image(args.size, rng)
        fname    = f'img_{i:05d}.jpg'
        Image.fromarray(arr).save(os.path.join(img_dir, fname), quality=90)
        records.append({'filename': fname, 'irradiance': round(irr, 2)})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{args.n}")

    pd.DataFrame(records).to_csv(os.path.join(args.out, 'labels.csv'), index=False)
    print(f"Hotovo: {args.n} obrázkov → {img_dir}/")

if __name__ == '__main__':
    main()
