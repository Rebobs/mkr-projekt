# Vplyv rotačných augmentácií na presnosť odhadu slnečného žiarenia

Semestrálny projekt skúmajúci, či augmentácia vstupných fotografií oblohy pomocou rotácií o 90°, 180° a 270° zlepší presnosť odhadu slnečného žiarenia (GHI, W/m²) pomocou predtrénovaných CNN.

## Obsah

- [Popis projektu](#popis-projektu)
- [Štruktúra repozitára](#štruktúra-repozitára)
- [Inštalácia](#inštalácia)
- [Dataset](#dataset)
- [Spustenie experimentov](#spustenie-experimentov)
- [Výsledky](#výsledky)

---

## Popis projektu

Vstupom do modelu sú fotografie oblohy, výstupom je jedna číselná hodnota — intenzita slnečného žiarenia v W/m². Posledná vrstva predtrénovaných sietí je nahradená regresnou vrstvou s jedným neurónom.

Experimenty porovnávajú dva scenáre:

| Scenár | Popis |
|--------|-------|
| **Bez augmentácie** | Model trénovaný iba na pôvodných snímkach |
| **S rotačnou augmentáciou** | Každý obrázok je počas tréningu náhodne rotovaný o 0°, 90°, 180° alebo 270° |

Testované modely:

| Model | Parametre | Typ |
|-------|-----------|-----|
| ResNet18 | ~11M | stredne náročný |
| EfficientNet-B0 | ~5M | efektívny |
| MobileNetV3-Small | ~2M | ľahký |

Testované veľkosti vstupu: **128×128, 224×224, 256×256, 320×320**

Metriky: **MAE**, **RMSE**, **R²**, čas trénovania, počet parametrov

---

## Štruktúra repozitára

```
mkr-projekt/
├── config.py                    # hyperparametre a cesty
├── requirements.txt
├── quickstart.sh                # inštalácia + rýchly test
│
├── generate_synthetic_data.py   # generovanie testovacieho datasetu
├── run_experiments.py           # hlavný experiment (grid)
├── tune_hyperparams.py          # ladenie hyperparametrov
├── plot_results.py              # generovanie grafov
│
├── src/
│   ├── dataset.py               # Dataset trieda + RandomRotation90 + DataLoadery
│   ├── models.py                # factory funkcia pre modely (→ regresná hlava)
│   ├── metrics.py               # MAE, RMSE, R²
│   └── trainer.py               # tréningová slučka + early stopping
│
└── data/
    ├── images/                  # fotografie oblohy
    └── labels.csv               # filename, irradiance
```

---

## Inštalácia

Vyžaduje Python 3.10+.

```bash
git clone https://github.com/Rebobs/mkr-projekt.git
cd mkr-projekt

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Dataset

### Vlastný dataset

Vlož fotografie oblohy do `data/images/` a vytvor súbor `data/labels.csv`:

```
filename,irradiance
img_001.jpg,523.4
img_002.jpg,681.2
img_003.jpg,240.8
```

### Syntetický dataset (na testovanie)

Ak nemáš vlastné dáta, vygeneruj syntetické snímky oblohy:

```bash
python generate_synthetic_data.py --n 300 --out data
```

---

## Spustenie experimentov

### Rýchly test

```bash
bash quickstart.sh
```

### Hlavný grid experiment

Spustí všetky kombinácie modelov × augmentácia × veľkosť vstupu:

```bash
python run_experiments.py
```

Výber konkrétnych modelov a veľkostí:

```bash
python run_experiments.py --models resnet18 efficientnet_b0 --sizes 224 256 --epochs 30
```

Výsledky sa ukladajú do `results/results_latest.json` a `results/summary_latest.csv`.

### Ladenie hyperparametrov

Automaticky vyberie najlepší model z predchádzajúceho behu a prehľadá priestor:
- learning rate: `0.001, 0.0005, 0.0001`
- batch size: `16, 32`
- optimizer: `Adam, AdamW`

```bash
python tune_hyperparams.py

# alebo pre konkrétny model
python tune_hyperparams.py --model resnet18 --size 224 --augment
```

### Generovanie grafov

```bash
python plot_results.py
```

Grafy sa uložia do `results/plots/`:

| Graf | Popis |
|------|-------|
| `loss_curves_224.png` | Tréningová a validačná strata počas epoch |
| `val_mae_curves_224.png` | Validačné MAE počas trénovania |
| `aug_comparison_test_mae_224.png` | MAE: bez augmentácie vs. s rotáciami |
| `aug_comparison_test_rmse_224.png` | RMSE: bez augmentácie vs. s rotáciami |
| `inputsize_mae_aug1.png` | Heatmapa MAE podľa veľkosti vstupu |
| `pred_vs_actual_224_aug0/1.png` | Predikovaná vs. skutočná hodnota |
| `training_time_224.png` | Porovnanie časov trénovania |
| `model_complexity.png` | Počet parametrov, veľkosť, čas inferencie |

---

## Výsledky

Po dokončení `run_experiments.py` sa vypíše tabuľka výsledkov:

```
model              input_size  augmentation  test_mae  test_rmse  test_r2  total_time_s
resnet18                  224         False     ...       ...       ...        ...
resnet18                  224          True     ...       ...       ...        ...
efficientnet_b0           224         False     ...       ...       ...        ...
...
```
