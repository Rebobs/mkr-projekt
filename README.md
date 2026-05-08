# Vplyv rotačných augmentácií na presnosť odhadu slnečného žiarenia

Semestrálny projekt skúmajúci, či augmentácia vstupných fotografií oblohy pomocou rotácií o 90°, 180° a 270° zlepší presnosť odhadu slnečného žiarenia (GHI, W/m²) pomocou predtrénovaných CNN.

## Súbory

| Súbor | Popis |
|-------|-------|
| `experiment.py` | Dataset, modely, tréning, vyhodnotenie — celý experiment |
| `plot.py` | Generovanie grafov z výsledkov |
| `requirements.txt` | Python závislosti |

## Inštalácia

```bash
git clone https://github.com/Rebobs/mkr-projekt.git
cd mkr-projekt

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Dataset

Dataset pochádza z projektu [Eye2Sky](https://zenodo.org/records/12804613) — fotografie oblohy z nemeckých meteorologických staníc s reálnymi GHI meraniami z pyranometra (W/m²).

Štruktúra:
```
data/
├── images/   ← fotografie oblohy (JPG)
└── labels.csv
```

`labels.csv`:
```
filename,irradiance
AURIC_20220620134930_160.jpg,314.3
LEEER_20220620120000_160.jpg,599.2
...
```

## Spustenie

```bash
# Plný experiment (všetky modely × veľkosti × aug/no-aug)
venv/bin/python experiment.py

# Rýchly test
venv/bin/python experiment.py --models resnet18 --sizes 224 --epochs 5

# Grafy (po dokončení experimentu)
venv/bin/python plot.py
```

Výsledky sa uložia do `results/`:
- `results.json` — plné výsledky vrátane histórie trénovania
- `summary.csv` — tabuľka metrík
- `plots/` — grafy

Pri novom spustení sa predchádzajúce výsledky automaticky archivujú do `results/archive/`.

## Modely

Každý model má nahradenú poslednú vrstvu regresnou hlavou s 1 výstupom (W/m²):

| Model | Parametre |
|-------|-----------|
| ResNet18 | ~11M |
| EfficientNet-B0 | ~5M |
| MobileNetV3-Small | ~2M |

## Metriky

- **MAE** — priemerná absolútna chyba (W/m²)
- **RMSE** — odmocnina strednej kvadratickej chyby (W/m²)
- **R²** — koeficient determinácie (1 = perfektný model)
