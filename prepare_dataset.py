"""
Pripraví dataset pre projekt:
  1. Načíta GHI merania z LEEER NetCDF
  2. Nájde všetky obrázky (5 staníc) a páruje ich podľa timestampu
  3. Vyhodí záznamy s chýbajúcim GHI (NaN)
  4. Náhodne vyberie N obrázkov rovnomerne rozložených cez deň
  5. Skopíruje do data/images/ a vytvorí data/labels.csv

Použitie:
  python prepare_dataset.py
  python prepare_dataset.py --n 1000
"""

import os
import shutil
import argparse
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timezone
from pathlib import Path

ASI_DIR  = '/home/martin/Downloads/ASI_20220620'
NC_FILE  = '/home/martin/Downloads/LEEER/data/LEEER.cleaned.nc'
OUT_DIR  = 'data'
MAX_DIFF_SECONDS = 60   # max povolený rozdiel medzi timestampom obrázka a GHI meraním


def load_ghi(nc_file):
    """Načíta GHI časový rad z NetCDF, vráti dict {datetime_utc: ghi_value}."""
    ds  = nc.Dataset(nc_file)
    times = ds.variables['time'][:]
    ghi   = ds.variables['GHI'][:]

    result = {}
    for t, g in zip(times, ghi):
        if np.ma.is_masked(g) or np.isnan(float(g)):
            continue
        dt = datetime.fromtimestamp(float(t), tz=timezone.utc)
        result[dt] = float(g)

    print(f"Načítaných GHI meraní (bez NaN): {len(result)}")
    print(f"GHI rozsah: {min(result.values()):.1f} – {max(result.values()):.1f} W/m²")
    return result


def parse_timestamp(filename):
    """Z názvu súboru '20220620134930_160.jpg' vráti datetime UTC."""
    ts_str = Path(filename).stem.split('_')[0]   # '20220620134930'
    return datetime.strptime(ts_str, '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)


def build_ghi_lookup(ghi_dict):
    """Zoradí GHI záznamy ako numpy pole pre rýchle binárne vyhľadávanie."""
    times  = np.array([t.timestamp() for t in ghi_dict.keys()])
    values = np.array(list(ghi_dict.values()))
    order  = np.argsort(times)
    return times[order], values[order]

def find_nearest_ghi(img_ts, ghi_times, ghi_values):
    """Binárne vyhľadávanie najbližšieho GHI merania (O(log n))."""
    idx = np.searchsorted(ghi_times, img_ts)
    candidates = []
    for i in [idx - 1, idx]:
        if 0 <= i < len(ghi_times):
            candidates.append((abs(ghi_times[i] - img_ts), ghi_values[i]))
    if not candidates:
        return None, float('inf')
    diff, ghi_val = min(candidates)
    if diff > MAX_DIFF_SECONDS:
        return None, diff
    return ghi_val, diff


def collect_all_images(asi_dir):
    """Nájde všetky JPG obrázky vo všetkých staniciach."""
    images = []
    for jpg in Path(asi_dir).rglob('*.jpg'):
        station = jpg.parts[len(Path(asi_dir).parts)]
        images.append({'path': jpg, 'station': station, 'filename': jpg.name})
    print(f"Nájdených obrázkov: {len(images)} z {len(set(i['station'] for i in images))} staníc")
    return images


def stratified_sample(records, n):
    """
    Vyberie n záznamov rovnomerne rozložených cez hodiny dňa,
    aby dataset pokrýval celý deň (ráno, poludnie, večer).
    """
    df = pd.DataFrame(records)
    df['hour'] = df['img_dt'].apply(lambda dt: dt.hour)
    hours = sorted(df['hour'].unique())
    per_hour = max(1, n // len(hours))

    sampled = []
    for hour in hours:
        hour_df = df[df['hour'] == hour]
        k = min(per_hour, len(hour_df))
        sampled.append(hour_df.sample(k, random_state=42))

    result = pd.concat(sampled)

    # Ak treba doplniť do presného počtu n
    if len(result) < n:
        remaining = df[~df.index.isin(result.index)]
        extra = remaining.sample(min(n - len(result), len(remaining)), random_state=42)
        result = pd.concat([result, extra])

    return result.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1000, help='Počet obrázkov (default: 1000)')
    args = p.parse_args()

    os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)

    print("=== 1. Načítavam GHI dáta (LEEER) ===")
    ghi_dict = load_ghi(NC_FILE)

    print("\n=== 2. Hľadám obrázky ===")
    images = collect_all_images(ASI_DIR)

    print("\n=== 3. Párujeme timestampy ===")
    ghi_times, ghi_values = build_ghi_lookup(ghi_dict)

    matched = []
    skipped_nan = 0
    skipped_far = 0

    for img in images:
        try:
            img_dt = parse_timestamp(img['filename'])
        except Exception:
            continue

        ghi_val, diff = find_nearest_ghi(img_dt.timestamp(), ghi_times, ghi_values)

        if ghi_val is None:
            skipped_far += 1
            continue
        if ghi_val < 0:
            skipped_nan += 1
            continue

        matched.append({
            'path':     img['path'],
            'station':  img['station'],
            'img_dt':   img_dt,
            'ghi':      round(ghi_val, 2),
            'time_diff': round(diff, 1),
        })

    print(f"Spárovaných: {len(matched)}")
    print(f"Vyhodených (NaN/záporné GHI): {skipped_nan}")
    print(f"Vyhodených (príliš ďaleko v čase): {skipped_far}")

    print(f"\n=== 4. Vyberám {args.n} obrázkov (stratifikované podľa hodiny) ===")
    selected = stratified_sample(matched, args.n)
    print(f"Vybraných: {len(selected)} obrázkov")
    print(f"Pokryté hodiny: {sorted(selected['hour'].unique().tolist())}")

    print(f"\n=== 5. Kopírujem obrázky a vytváram labels.csv ===")
    records = []
    for _, row in selected.iterrows():
        # Unikátny názov: station_originalnazov.jpg
        new_name = f"{row['station']}_{row['path'].name}"
        dst = os.path.join(OUT_DIR, 'images', new_name)
        shutil.copy2(row['path'], dst)
        records.append({'filename': new_name, 'irradiance': row['ghi']})

    df_labels = pd.DataFrame(records)
    df_labels.to_csv(os.path.join(OUT_DIR, 'labels.csv'), index=False)

    print(f"\nHotovo!")
    print(f"  Obrázky: {OUT_DIR}/images/ ({len(records)} súborov)")
    print(f"  Labels:  {OUT_DIR}/labels.csv")
    print(f"  GHI rozsah: {df_labels['irradiance'].min():.1f} – {df_labels['irradiance'].max():.1f} W/m²")
    print(f"  GHI priemer: {df_labels['irradiance'].mean():.1f} W/m²")
    print(f"\nRozloženie podľa hodín:")
    print(selected.groupby('hour')['ghi'].agg(['count', 'mean']).rename(
        columns={'count': 'n_obrazkov', 'mean': 'avg_GHI'}).round(1).to_string())


if __name__ == '__main__':
    main()
