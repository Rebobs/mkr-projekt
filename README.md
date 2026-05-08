# Vplyv rotačných augmentácií na presnosť odhadu slnečného žiarenia

Semestrálny projekt skúmajúci, či augmentácia vstupných fotografií oblohy pomocou rotácií o 90°, 180° a 270° zlepší presnosť odhadu slnečného žiarenia (GHI, W/m²) pomocou predtrénovaných CNN.

## Súbory

| Súbor | Popis |
|-------|-------|
| `experiment.py` | Dataset, modely, tréning, vyhodnotenie — celý experiment |
| `plot.py` | Generovanie grafov z výsledkov |
| `generate_data.py` | Syntetický dataset na testovanie |
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

Vlož fotografie oblohy do `data/images/` a vytvor `data/labels.csv`:

```
filename,irradiance
img_001.jpg,523.4
img_002.jpg,681.2
```

Ak nemáš vlastné dáta, vygeneruj syntetické:

```bash
python generate_data.py --n 300
```

## Spustenie

```bash
# Spustí všetky modely × veľkosti × aug/no-aug
python experiment.py

# Rýchly test: 1 model, 1 veľkosť, 5 epoch
python experiment.py --models resnet18 --sizes 224 --epochs 5

# Grafy z výsledkov
python plot.py
```

Výsledky sa uložia do `results/`:
- `results.json` — plné výsledky vrátane histórie trénovania
- `summary.csv` — tabuľka metrík
- `plots/` — grafy

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
