# Interná dokumentácia — MKR projekt

Tento dokument sa neodovzdáva. Slúži ako podrobný prehľad kódu pre potreby obhajoby.

---

## experiment.py

Hlavný skript, ktorý trénuje neurónové siete a ukladá výsledky.

---

### Konštanty (nastavenia experimentu)

```python
DATA_DIR    = 'data'
RESULTS_DIR = 'results'
MODELS         = ['resnet18', 'efficientnet_b0', 'mobilenet_v3_small']
INPUT_SIZES    = [128, 224, 320]
LEARNING_RATES = [0.001, 0.0001, 0.01]
TRAIN_FRACS    = [0.25, 0.5, 0.75, 1.0]
EPOCHS         = 20
BATCH_SIZE     = 32
PATIENCE       = 7
SEED           = 42
```

**DATA_DIR** — priečinok kde sú uložené obrázky a súbor `labels.csv` s hodnotami žiarenia.

**RESULTS_DIR** — kam sa ukladajú výsledky experimentov (JSON, CSV, grafy).

**MODELS** — zoznam troch predtrénovaných sietí ktoré porovnávame. Sú to reťazce, ktoré neskôr použijeme v `get_model()` na načítanie správnej architektúry.

**INPUT_SIZES** — tri rozlíšenia do ktorých sa každý obrázok zmenší pred vstupom do siete. Porovnávame, či väčší vstup dáva lepšie výsledky.

**LEARNING_RATES** — rýchlosť učenia (krok optimizera). Malá hodnota = pomalé, stabilné učenie. Veľká = rýchle, ale môže prestreliť. Testujeme tri hodnoty a vyberáme najlepšiu.

**TRAIN_FRACS** — frakcie trénovacej množiny (25 %, 50 %, 75 %, 100 %). Slúžia na experiment „čo ak by sme mali menej dát".

**EPOCHS** — maximálny počet kôl trénovania. Jedno kolo = sieť raz prejde cez celý trénovací dataset.

**BATCH_SIZE** — počet obrázkov spracovaných naraz v jednom kroku. 32 je štandardná hodnota — dosť veľká na stabilný gradient, dosť malá aby sa zmestila do GPU pamäte.

**PATIENCE** — počet epoch bez zlepšenia po ktorom sa tréning zastaví predčasne (early stopping). Zabraňuje preučeniu (overfitting) a zbytočnému čakaniu.

**SEED** — pevné náhodné semeno. Zaručuje, že rozdelenie datasetu a vzorkovanie sú vždy rovnaké → porovnanie výsledkov je férové.

---

### Trieda `Rot90`

```python
class Rot90:
    def __call__(self, img):
        return transforms.functional.rotate(img, random.choice([0, 90, 180, 270]))
```

Táto trieda predstavuje jednu augmentačnú operáciu — náhodné otočenie obrázka.

`__call__` je špeciálna metóda v Pythone. Keď napíšeme `rot = Rot90()` a potom `rot(obrazok)`, zavolá sa práve táto metóda. Vďaka tomu sa `Rot90` správa ako funkcia a dá sa zaradiť do pipeline transformácií.

`random.choice([0, 90, 180, 270])` — náhodne vyberie jeden z čísel v zozname. To znamená, že každý obrázok môže zostať nezmenený (0°), otočiť sa o štvrť otáčky (90°), naopak (180°) alebo o tri štvrtiny (270°).

`transforms.functional.rotate(img, ...)` — aplikuje rotáciu na PIL obrázok. PIL je knižnica na prácu s obrázkami v Pythone.

---

### Trieda `SkyDataset`

```python
class SkyDataset(Dataset):
    def __init__(self, df, img_dir, transform, cache):
        self.df     = df.reset_index(drop=True)
        self.tf     = transform
        self.imgs   = [cache[row['filename']] for _, row in self.df.iterrows()]
        self.labels = [float(row['irradiance']) for _, row in self.df.iterrows()]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.tf(self.imgs[idx]), torch.tensor(self.labels[idx], dtype=torch.float32)
```

Dataset je objekt, ktorý PyTorch používa na načítavanie dát počas trénovania. Musí implementovať tri metódy: `__init__`, `__len__` a `__getitem__`.

**`__init__`** — inicializácia, zavolá sa raz pri vytvorení datasetu.

- `df.reset_index(drop=True)` — obnoví indexy riadkov v tabuľke. Po rozdelení datasetu (train/val/test) môžu byť indexy neusporiadané, reset to opraví.
- `self.tf = transform` — uloží transformáciu (pipeline operácií nad obrázkom) pre neskoršie použitie.
- `self.imgs = [cache[row['filename']] ...]` — prejde každý riadok tabuľky, vezme názov súboru (`filename`) a nájde zodpovedajúci obrázok v cache (slovník {názov: PIL obrázok}). Výsledok je zoznam PIL obrázkov.
- `self.labels = [float(row['irradiance']) ...]` — podobne, ale pre hodnoty žiarenia. `float()` konvertuje hodnotu na desatinné číslo.

**`__len__`** — vráti počet vzoriek v datasete. PyTorch to potrebuje vedieť na správne rozdelenie do dávok.

**`__getitem__`** — vráti jednu vzorku (obrázok + hodnota) podľa indexu `idx`.
- `self.tf(self.imgs[idx])` — aplikuje transformácie na PIL obrázok (resize, normalizácia, prípadne rotácia). Výsledkom je tenzor — n-rozmerné pole čísel, formát ktorý PyTorch požaduje.
- `torch.tensor(self.labels[idx], dtype=torch.float32)` — konvertuje číslo žiarenia na tenzor s typom float32 (32-bitové desatinné číslo).

---

### Funkcia `make_transform`

```python
def make_transform(size, augment=False):
    ops = [transforms.Resize((size, size))]
    if augment:
        ops.append(Rot90())
    ops += [transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(ops)
```

Zostaví zoznam operácií (pipeline), ktoré sa aplikujú na každý obrázok v poradí.

`ops = [transforms.Resize((size, size))]` — prvá operácia je vždy zmenšenie na cieľové rozlíšenie (napr. 224×224 pixelov). Obrázky v datasete majú rôzne veľkosti, sieť vyžaduje pevnú veľkosť vstupu.

`if augment: ops.append(Rot90())` — ak je augmentácia zapnutá, zaradí za resize náhodné otočenie.

`transforms.ToTensor()` — konvertuje PIL obrázok (hodnoty 0–255) na PyTorch tenzor s hodnotami 0.0–1.0. Tvar sa zmení z (výška, šírka, 3 kanály) na (3, výška, šírka) — PyTorch pracuje s kanálmi ako prvou dimenziou.

`transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])` — normalizácia pomocou priemeru a štandardnej odchýlky ImageNet datasetu. Každý kanál (R, G, B) sa normalizuje zvlášť: `(hodnota - priemer) / std`. Toto je dôležité, pretože sieť bola predtrénovaná na ImageNete s týmito štatistikami — vstupné dáta musia mať rovnaké rozloženie.

`transforms.Compose(ops)` — zlúči všetky operácie do jedného objektu, ktorý ich aplikuje postupne.

---

### Funkcia `load_image_cache`

```python
def load_image_cache(img_dir, filenames):
    max_size = max(INPUT_SIZES)
    print(f"  Načítavam obrázky do RAM (pre-resize na {max_size}px)...")
    return {
        fn: Image.open(os.path.join(img_dir, fn)).convert('RGB').resize(
            (max_size, max_size), Image.BILINEAR)
        for fn in filenames
    }
```

Prednahrá všetky obrázky do operačnej pamäte (RAM) pred začiatkom trénovania.

**Prečo?** Bez cache by každý trénovací krok čítal obrázky z disku. Disk je pomalý — GPU by čakalo na dáta a ostávalo by väčšinu času nečinné. Prednahranie do RAM eliminuje tento bottleneck.

`max_size = max(INPUT_SIZES)` — nájde najväčšiu testovanú veľkosť vstupu (320). Obrázky sa predresizujú len na túto veľkosť, nie na pôvodné rozlíšenie. Ak by sme uložili pôvodné rozlíšenie (napr. 2000×1500), cache by zabrala desiatky GB RAM. Na 320px je to len ~300 MB.

`Image.open(...).convert('RGB')` — otvorí súbor ako PIL obrázok a konvertuje na RGB (3 kanály). Niektoré súbory môžu byť RGBA (s alfa kanálom) alebo odtiene sivej — konverzia zaručuje jednotný formát.

`.resize((max_size, max_size), Image.BILINEAR)` — zmenší obrázok na 320×320 pomocou bilineárnej interpolácie (jemné vyhladenie pri zmenšovaní).

Výsledok je Python slovník `{názov_súboru: PIL_obrázok}` — rýchle vyhľadávanie podľa názvu.

---

### Funkcia `get_loaders`

```python
def get_loaders(size, augment, cache, train_frac=1.0):
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
        DataLoader(SkyDataset(val_df,   None, eval_tf, cache), shuffle=False, **kw),
        DataLoader(SkyDataset(test_df,  None, eval_tf, cache), shuffle=False, **kw),
        len(train_df),
    )
```

Načíta dataset, rozdelí ho a vytvorí DataLoadery — objekty ktoré automaticky vydávajú dáta po dávkach počas trénovania.

`pd.read_csv(...)` — načíta CSV súbor do tabuľky (DataFrame). Každý riadok = jeden obrázok s názvom súboru a hodnotou žiarenia.

`train_test_split(df, test_size=0.30, random_state=SEED)` — náhodne rozdelí tabuľku na 70 % (tréning) a 30 % (zvyšok). `random_state=SEED` zaručuje rovnaké rozdelenie pri každom spustení.

`train_test_split(tmp_df, test_size=0.50, ...)` — zvyšných 30 % rozdelí na polovice: 15 % validácia, 15 % test. Validácia slúži na sledovanie výkonu počas trénovania, test sa použije len raz na záverečné vyhodnotenie.

`train_df.sample(frac=train_frac, ...)` — ak `train_frac < 1.0`, náhodne vyberie len danú frakciu trénovacej množiny. Napr. `frac=0.25` použije len štvrtinu.

`eval_tf = make_transform(size, augment=False)` — transformácia bez augmentácie. Validácia a test sa nikdy neaugmentujú — výsledky by neboli porovnateľné medzi epochami.

`num_workers=0` — načítavanie dát prebieha v hlavnom procese (nie v paralelných procesoch). Na Windowse a s RAM cache je to správne nastavenie.

`pin_memory=True` — alokuje dáta v "pinned" pamäti, čo urýchľuje prenos na GPU.

`shuffle=True` pre tréning — dáta sa každú epochu zamiešajú. Zabraňuje tomu, aby sa sieť naučila poradie vzoriek namiesto ich obsahu.

`shuffle=False` pre val/test — zachová poradie, čo nie je nutné, ale je prehľadné.

Funkcia vracia tuple: `(train_loader, val_loader, test_loader, počet_trénovacích_vzoriek)`.

---

### Funkcia `get_model`

```python
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
```

Načíta predtrénovanú sieť z knižnice torchvision a upraví jej výstupnú vrstvu pre regresiu.

**Čo je predtrénovaná sieť?** Siete boli pôvodne natrénované na ImageNet datasete — 1,2 milióna obrázkov, 1000 kategórií. Naučili sa rozpoznávať základné vizuálne vzory (hrany, textúry, tvary). Tieto znalosti prenesieme na náš problém — tomu sa hovorí transfer learning.

`weights=models.ResNet18_Weights.IMAGENET1K_V1` — načíta predtrénované váhy. Bez tohto argumentu by sieť mala náhodné váhy a musela by sa učiť od nuly.

**Výmena poslednej vrstvy:**

Pôvodná výstupná vrstva produkuje 1000 čísel (pravdepodobnosti pre 1000 tried). My potrebujeme jedno číslo — hodnotu žiarenia. Preto ju vymeníme:

- **ResNet18**: posledná vrstva je `m.fc` (fully connected). `m.fc.in_features` je počet vstupných neurónov tej vrstvy (512 pre ResNet18). Nahradíme ju `nn.Linear(512, 1)` — lineárna vrstva so 512 vstupmi a 1 výstupom.
- **EfficientNet**: posledná vrstva je `m.classifier[1]` (druhý prvok v klasifikátore). Podobná výmena.
- **MobileNet**: posledná vrstva je `m.classifier[3]`. Podobná výmena.

Výsledkom je sieť, ktorá na vstupe berie obrázok a na výstupe vydáva jedno číslo — odhadovanú hodnotu žiarenia v W/m².

---

### Funkcia `train_epoch`

```python
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
```

Spustí jednu trénovaciu epochu — sieť raz prejde cez celý trénovací dataset a upraví váhy.

`model.train()` — prepne sieť do trénovacieho režimu. Niektoré vrstvy (Dropout, BatchNorm) sa správajú inak pri trénovaní vs. vyhodnocovaní. Toto nastavenie zaručí správne správanie.

`for imgs, labels in loader` — DataLoader vydáva dáta po dávkach. `imgs` je tenzor tvaru `(32, 3, 224, 224)` — 32 obrázkov, 3 kanály, 224×224 pixelov. `labels` je tenzor 32 čísel.

`imgs.to(device)` — prenesie dáta na GPU (ak je dostupná). Výpočty na GPU sú desiatky až stovky krát rýchlejšie ako na CPU pre maticové operácie.

`labels.to(device).unsqueeze(1)` — prenesie na GPU a pridá rozmer: z tvaru `(32,)` urobí `(32, 1)`. Výstup siete má tvar `(32, 1)`, labels musia mať rovnaký tvar pre správny výpočet straty.

`optimizer.zero_grad()` — vymaže gradienty z predchádzajúceho kroku. Gradienty sa v PyTorche akumulujú, čo by bez tohto kroku viedlo k nesprávnym aktualizáciám váh.

`loss = criterion(model(imgs), labels)` — forward pass: `model(imgs)` spustí sieť a dostaneme predikcie. `criterion` je MSE strata (mean squared error): `mean((predikcia - skutočná)²)`. Výsledok je jedno číslo — miera chyby na tejto dávke.

`loss.backward()` — backpropagation: vypočíta gradient straty voči každej váhe siete. Gradient hovorí, ktorým smerom a o koľko treba váhu zmeniť, aby sa strata znížila.

`optimizer.step()` — Adam optimizer aktualizuje váhy podľa vypočítaných gradientov a learning rate.

`total += loss.item() * imgs.size(0)` — akumuluje stratu váhovanú počtom vzoriek v dávke. `.item()` konvertuje jednoprvkový tenzor na Python číslo. Delenie na konci dá priemernú stratu na vzorku.

---

### Funkcia `evaluate`

```python
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
```

Vyhodnotí model na celom datasete a vráti metriky.

`@torch.no_grad()` — dekorátor, ktorý vypne výpočet gradientov pre celú funkciu. Pri vyhodnocovaní gradienty nepotrebujeme, vypnutím ušetríme pamäť a zrýchlime výpočet.

`model.eval()` — prepne do vyhodnocovacieho režimu (opak `model.train()`).

`preds, targets = [], []` — prázdne zoznamy, do ktorých budeme zbierať predikcie a skutočné hodnoty postupne po dávkach.

`.squeeze(1)` — odstráni rozmer veľkosti 1: z tvaru `(32, 1)` urobí `(32,)`.

`.cpu().numpy()` — prenesie tenzor z GPU do CPU a konvertuje na NumPy pole. NumPy je knižnica pre numerické výpočty, pracuje len na CPU.

`np.array(targets)` — zlúči zoznam zoznamov do jedného NumPy poľa.

**Metriky:**

- **MAE** (Mean Absolute Error): `mean(|y_true - y_pred|)` — priemerná absolútna odchýlka v W/m². Ak MAE = 125, priemerne sa mýlime o 125 W/m².
- **RMSE** (Root Mean Squared Error): `sqrt(mean((y_true - y_pred)²))` — podobné ako MAE, ale väčšie chyby penalizuje viac (kvadrát). Citlivejšie na výnimky.
- **R²** (koeficient determinácie): hodnotí, koľko percenta variability žiarenia model vysvetľuje. R² = 1.0 je perfektná predikcia, R² = 0 je rovnaké ako predpovedať vždy priemer, R² < 0 je horší ako priemer.

---

### Funkcia `run`

```python
def run(model_name, size, augment, epochs, lr, device, cache, train_frac=1.0):
```

Spustí jeden kompletný experiment — tréning a testovanie jednej konfigurácie.

**Inicializácia:**

```python
model     = get_model(model_name).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
```

- `get_model(model_name).to(device)` — načíta sieť a prenesie ju na GPU.
- `torch.optim.Adam(model.parameters(), lr=lr)` — Adam optimizer sleduje všetky trénovateľné parametre (váhy) siete a aktualizuje ich podľa gradientov. Adam je adaptívny optimizer — automaticky prispôsobuje learning rate pre každý parameter.
- `nn.MSELoss()` — stratová funkcia, mean squared error.

**Trénovacia slučka:**

```python
best_val_mae = float('inf')
best_weights = None
best_epoch   = 0
no_improve   = 0
```

Premenné pre early stopping. `float('inf')` = nekonečno — akákoľvek reálna hodnota MAE bude menšia.

```python
for epoch in range(1, epochs + 1):
    loss = train_epoch(...)
    val  = evaluate(model, val_loader, device)
```

Každú epochu: natrénujeme sieť na trénovacej množine, potom vyhodnotíme na validačnej (bez úpravy váh).

```python
if improved:
    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
```

Ak je validačné MAE lepšie ako doteraz, uložíme kópiu váh. `state_dict()` je slovník všetkých váh siete. `.cpu().clone()` — prenesieme na CPU a vytvoríme nezávislú kópiu (nie referenciu), aby neskoršie zmeny váh neovplyvnili uloženú kópiu.

```python
else:
    no_improve += 1
    if no_improve >= PATIENCE:
        break
```

Ak sa MAE nezlepšilo, zvýšime počítadlo. Po 7 epochách bez zlepšenia zastavíme tréning.

```python
model.load_state_dict(best_weights)
test = evaluate(model, test_loader, device)
```

Obnovíme najlepšie váhy a vyhodnotíme na testovacej množine. Test sa robí len raz, na úplnom konci — nie počas trénovania.

**Návratová hodnota** — slovník so všetkými metrikami, nastaveniami a históriou trénovania. Uloží sa do JSON súboru.

---

### Funkcia `main`

```python
def main():
```

Riadi celý experiment od začiatku do konca.

**Archivácia predchádzajúcich výsledkov:**

```python
if os.path.exists(existing):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_dir = os.path.join(RESULTS_DIR, 'archive', ts)
    shutil.move(src, os.path.join(archive_dir, f))
```

Ak existujú výsledky z predchádzajúceho behu, presunú sa do priečinka `archive/` s časovou pečiatkou. Zabraňuje prepísaniu starých výsledkov.

**Načítanie cache:**

```python
cache = load_image_cache(img_dir, df['filename'].tolist())
```

Prednahrá obrázky do RAM — zavolá sa raz, cache sa zdieľa naprieč všetkými experimentmi.

**Hlavná slučka:**

```python
for lr in LEARNING_RATES:
    for train_frac in TRAIN_FRACS:
        for model_name in MODELS:
            for size in INPUT_SIZES:
                for augment in [False, True]:
                    results.append(run(...))
                with open(..., 'w') as f:
                    json.dump(results, f)
```

Päť vnorených cyklov prechádza všetky kombinácie hyperparametrov. `3 × 4 × 3 × 3 × 2 = 216 experimentov`.

Výsledky sa ukladajú po každej dvojici (bez aug / s aug) pre rovnakú konfiguráciu — pri prerušení sa nestratia hotové výsledky.

**Záverečný súhrn:**

```python
skip = {'train_losses', 'val_maes', 'y_true', 'y_pred'}
df = pd.DataFrame([{k: v for k, v in r.items() if k not in skip} for r in results])
df.to_csv(...)
```

Vytvorí CSV súbor so súhrnom bez veľkých polí (histórie strát, predikcie) — tie ostávajú len v JSON pre grafy.

---

## plot.py

Skript na vizualizáciu výsledkov z `results/results.json`.

---

### Konštanty

```python
COLORS = {'resnet18': '#2196F3', 'efficientnet_b0': '#4CAF50', 'mobilenet_v3_small': '#FF9800'}
LABELS = {'resnet18': 'ResNet18', 'efficientnet_b0': 'EfficientNet-B0', 'mobilenet_v3_small': 'MobileNetV3-S'}
```

Slovníky pre konzistentnú farbu a popis každého modelu naprieč všetkými grafmi. Kľúč je interný identifikátor (reťazec z JSON), hodnota je farba (hex kód) alebo čitateľný popis.

---

### Funkcia `save`

```python
def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
```

Uloží matplotlib figure do súboru.

`dpi=150` — rozlíšenie 150 bodov na palec. Dosť vysoké pre prezentáciu, nie zbytočne veľký súbor.

`bbox_inches='tight'` — oreže prázdne okraje okolo grafu.

`plt.close(fig)` — uvoľní pamäť. Matplotlib drží figures v pamäti kým ich explicitne nezatvoríme. Bez tohto by sa pamäť postupne zaplnila.

---

### Funkcia `best_per_config`

```python
def best_per_config(results):
    best = {}
    for r in results:
        key = (r['model'], r['input_size'], r['augmentation'], r.get('train_frac', 1.0))
        if key not in best or r['test_mae'] < best[key]['test_mae']:
            best[key] = r
    return list(best.values())
```

Z výsledkov pre všetky learning rates vyberie pre každú konfiguráciu len ten najlepší výsledok.

**Prečo?** Experiment testoval 3 learning rates. Pri porovnávaní vplyvu augmentácie alebo veľkosti vstupu nechceme byť znevýhodnení zlým LR — chceme porovnávať najlepší možný výsledok každej konfigurácie.

`key = (model, veľkosť, augmentácia, frakcia)` — n-tica, ktorá jednoznačne identifikuje konfiguráciu bez ohľadu na LR.

`r.get('train_frac', 1.0)` — bezpečné čítanie kľúča zo slovníka s predvolenou hodnotou. Ak kľúč neexistuje (staré výsledky), predpokladáme 100 % datasetu.

Výsledok: zoznam najlepších výsledkov, jeden pre každú unikátnu kombináciu.

---

### Funkcia `plot_aug_comparison`

```python
def plot_aug_comparison(results, size=224):
```

Dvojitý bar chart: MAE a RMSE pre každý model, vedľa seba bez aug a s aug.

`subset = [r for r in results if r['input_size'] == size and r.get('train_frac', 1.0) == 1.0]` — filtruje len výsledky pre danú veľkosť vstupu a plný dataset (100 %).

`model_names = list(dict.fromkeys(r['model'] for r in subset))` — extrahuje unikátne názvy modelov v poradí v akom sa vyskytujú. `dict.fromkeys()` zachová poradie (na rozdiel od `set()`).

`x = np.arange(len(model_names))` — pole `[0, 1, 2]` — pozície stĺpcov na osi X.

`w = 0.35` — šírka jedného stĺpca. Dva stĺpce (bez aug, s aug) majú spolu šírku 0.7, čo nechá medzery medzi skupinami.

`ax.bar(x - w/2, no_aug, w, ...)` — nakreslí stĺpce pre "bez aug" posunuté doľava o polovicu šírky.

`ax.bar(x + w/2, with_aug, w, ...)` — nakreslí stĺpce pre "s aug" posunuté doprava.

```python
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.2f}', ...)
```

Pridá číselnú hodnotu nad každý stĺpec. `bar.get_x() + bar.get_width()/2` = stred stĺpca na osi X. `bar.get_height() + 0.3` = tesne nad vrcholom stĺpca.

`bbox_to_anchor=(0.5, -0.12)` — umiestni legendu pod graf (mimo plochy grafu), aby neprekrývala stĺpce.

`fig.subplots_adjust(bottom=0.15)` — pridá spodný okraj, aby sa legenda zmestila.

---

### Funkcia `plot_training_curves`

```python
def plot_training_curves(results, size=224):
```

Čiarový graf: vývoj validačného MAE počas epoch pre každý model.

`ls = '-' if r['augmentation'] else '--'` — plná čiara pre aug, prerušovaná pre bez aug. Vizuálne odlíšenie bez nutnosti extra farieb.

`range(1, len(r['val_maes']) + 1)` — os X: epochy od 1 po posledná. `val_maes` je zoznam MAE hodnôt uložených počas trénovania.

Graf umožňuje vidieť, či model konverguje (klesá), stagnuje, alebo osciluje. Tiež vidno kedy nastalo early stopping.

---

### Funkcia `plot_pred_vs_actual`

```python
def plot_pred_vs_actual(results, size=224):
```

Scatter plot: predikovaná hodnota (os Y) vs. skutočná hodnota (os X) pre každý model.

`cols = min(3, n)` — maximálne 3 grafy vedľa seba. `rows = (n + cols - 1) // cols` — potrebný počet riadkov (ceiling division).

`axes = np.array(axes).flatten()` — konvertuje 2D pole podgrafov na 1D zoznam pre jednoduchšiu iteráciu.

`ax.plot([lo, hi], [lo, hi], 'k--', ...)` — nakreslí čiernu prerušovanú diagonálu. Body na diagonále = perfektná predikcia. Body nad diagonálou = model nadhodnocuje, pod = podhodnocuje.

`for j in range(i + 1, len(axes)): axes[j].set_visible(False)` — schová nevyužité podgrafy (ak počet modelov nie je deliteľný počtom stĺpcov).

---

### Funkcia `plot_input_sizes`

```python
def plot_input_sizes(results):
```

Čiarový graf: ako sa mení MAE pri rôznych veľkostiach vstupu (128, 224, 320 px).

Dva panely vedľa seba: ľavý bez augmentácie, pravý s augmentáciou. Umožňuje vidieť, či veľkosť vstupu interaguje s augmentáciou.

`if len(sizes) < 2: return` — ak máme len jednu veľkosť vstupu, graf nemá zmysel. Bezpečnostná kontrola.

---

### Funkcia `plot_training_time`

```python
def plot_training_time(results, size=224):
```

Dva bar charty: celkový čas trénovania a číslo najlepšej epochy pre každý model.

`epochs = [r.get('best_epoch', r.get('total_epochs', 0)) for r in subset]` — preferuje `best_epoch` (epocha s najlepším val MAE), záložne `total_epochs`. Hovorí, ako skoro model konvergoval.

`ax.set_ylim(bottom=0, top=max(val) * 1.15)` — os Y začína od nuly a sega 15 % nad najvyššou hodnotou. Priestor pre číselné popisky nad stĺpcami.

---

### Funkcia `plot_complexity`

```python
def plot_complexity(results):
```

Bar chart: počet parametrov každého modelu.

```python
seen, rows = set(), []
for r in results:
    if r['model'] not in seen:
        seen.add(r['model'])
        rows.append(r)
```

Každý model sa v results opakuje mnohokrát (rôzne LR, veľkosti...). Tento kód zachová len prvý výskyt každého modelu — počet parametrov je vždy rovnaký, nezávisí od ostatných nastavení.

`ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))` — formátuje os Y: namiesto `11200000` zobrazí `11M`. `lambda x, _` je anonymná funkcia s dvoma parametrami (hodnota, pozícia) — druhý ignorujeme.

---

### Funkcia `plot_train_size_effect`

```python
def plot_train_size_effect(results, size=224):
```

Čiarový graf: ako závisí MAE od počtu trénovacích vzoriek.

`fracs = sorted({r.get('train_frac', 1.0) for r in results})` — množina (set) unikátnych frakcií zo všetkých výsledkov, zoradená vzostupne.

`if len(fracs) < 2: return` — ak máme len jednu frakciu, krivku nemožno nakresliť.

`subset = [r for r in results if r['input_size'] == size and r['augmentation']]` — berie len výsledky s augmentáciou. (Voľba — efekt veľkosti datasetu je podobný pre oba prípady.)

`n_trains = [next((r['n_train'] for r in subset if r['model']==m and r.get('train_frac')==f), None) for f in fracs]` — pre každú frakciu nájde počet skutočných trénovacích vzoriek. `next(..., None)` vráti prvú zhodu alebo `None` ak sa nenájde.

---

### Funkcia `plot_lr_effect`

```python
def plot_lr_effect(results, size=224):
```

Čiarový graf: závislosť MAE od hodnoty learning rate — bez redukcie na best_per_config, pretože LR je práve tá premenná ktorú sledujeme.

`lrs = sorted({r['lr'] for r in subset})` — unikátne learning rates zoradené vzostupne.

Dva panely: bez augmentácie a s augmentáciou. Umožňuje vidieť, či optimálne LR závisí od augmentácie.

---

### Funkcia `main` (plot.py)

```python
def main():
    ...
    best = best_per_config(results)

    plot_aug_comparison(best, ...)
    ...
    plot_lr_effect(results, ...)
```

Väčšina grafov dostane `best` (najlepší výsledok pre každú konfiguráciu). Iba `plot_lr_effect` dostane všetky `results` — potrebuje vidieť výsledky pre každé LR zvlášť, nie len to najlepšie.

`default_size = 224 if 224 in sizes else sizes[0]` — predvolená veľkosť vstupu pre grafy je 224 (štandardná pre ImageNet). Ak v datasete chýba, vezme prvú dostupnú.

---

## Čo sa stane ak zmením hodnoty v kóde

---

### `BATCH_SIZE = 32`

**Ak zvýšim (napr. 128):**
- Každý krok rýchlejší, gradient presnejší (priemer cez viac vzoriek)
- Model môže konvergovať do „ostrých" miním → horšia generalizácia na nové dáta
- Menej krokov za epochu pri rovnako veľkom datasete

**Ak znížim (napr. 8):**
- Gradient šumivý → nestabilné trénovanie
- GPU nie je plne vyťažené → pomalší tréning
- Môže pomôcť uniknúť lokálnym minimám

**32** je štandardný kompromis pre tento dataset a GPU.

---

### `LEARNING_RATES = [0.001, 0.0001, 0.01]`

**Ak použijem príliš veľké LR (napr. 0.1):**
- Optimizer preskakuje cez minimu → val MAE osciluje alebo rastie
- Model nekonverguje vôbec

**Ak použijem príliš malé LR (napr. 0.000001):**
- Váhy sa menia minimálne → za 20 epoch skoro žiadne zlepšenie
- Stabilný tréning, ale nezmyselne pomalý

**Prečo testujeme tri:** Optimálne LR závisí od architektúry a datasetu, bez experimentu to nevieme. `0.001` je štandardné pre Adam — čo potvrdili aj naše výsledky.

---

### `EPOCHS = 20`

**Ak zvýšim (napr. 100):**
- Early stopping zastaví tréning aj tak najneskôr `PATIENCE` epoch po najlepšej
- Zvýšenie nad ~30 epoch nemá pri tomto datasete takmer žiadny efekt
- Výnimka: ak by sme výrazne znížili LR, pomalšia konvergencia by viac epoch využila

**Ak znížim (napr. 5):**
- Model nemá čas konvergovať, validačná MAE ešte klesá ale zastavíme predčasne
- Výsledky výrazne horšie

---

### `PATIENCE = 7`

**Ak zvýšim (napr. 15):**
- Tolerujeme dlhšiu stagnáciu — môže zachytiť neskorú konvergenciu
- Tréning trvá dlhšie, riziko overfittingu v čase čakania

**Ak znížim (napr. 2–3):**
- Zastavíme príliš skoro — model možno ešte konvergoval
- Vhodné len pre rýchle orientačné experimenty

---

### `INPUT_SIZES = [128, 224, 320]`

**Ak pridám menší (napr. 64px):**
- Možná strata informácie o pozícii slnka
- Rýchlejší tréning, menšia pamäťová náročnosť

**Ak pridám väčší (napr. 512px):**
- Výrazne pomalší tréning (výpočty rastú kvadraticky s rozlíšením)
- Pri 946 obrázkoch takmer istý overfitting
- Väčšie nároky na GPU pamäť

**Prečo 128px vyhral:** Pre odhad žiarenia stačí vedieť kde je slnko a koľko oblohy pokrývajú oblaky — to 128px zachytí. Väčší vstup pridáva komplexitu bez pridanej hodnoty pri malom datasete.

---

### `TRAIN_FRACS = [0.25, 0.5, 0.75, 1.0]`

Nie je hyperparameter na ladenie — je to experiment na pochopenie koľko dát potrebujeme.

| Frakcia | Vzorky | Výsledok |
|---------|--------|----------|
| 25 % | ~175 | R² záporné — horší ako predpovedanie priemeru |
| 50 % | ~330 | Výrazné zlepšenie, model začína zachytávať vzory |
| 75 % | ~497 | Ďalšie zlepšenie, krivka sa vyrovnáva |
| 100 % | ~662 | Najlepší výsledok |

Minimum pre zmysluplné výsledky pri transfer learningu je ~300–400 vzoriek.

---

### `MODELS`

**Ak pridám ResNet50 (25M parametrov):**
- Hlbšia sieť, viac parametrov → pri 946 obrázkoch silný overfitting
- Pomalší tréning, výsledky pravdepodobne horšie ako ResNet18

**Ak pridám ViT (Vision Transformer):**
- Transformery potrebujú omnoho viac dát (typicky stovky tisíc obrázkov)
- Pri 946 obrázkoch by takmer určite zlyhali
- CNN majú vstavanú indukčnú biasovosť (lokálne spojenia) ktorá pomáha pri malom datasete

---

### Augmentácia — alternatívy k rotácii

Rotácia o 90° ukázala, že škodí. Iné augmentácie by mohli byť vhodnejšie:

**Horizontálny flip:**
- Obloha je fyzikálne symetrická ľavo-pravý flip (slnko môže byť na ľavej alebo pravej strane)
- Nenarúša horizont → mohlo by pomôcť

**Zmena jasu/kontrastu:**
- Simuluje rôzne podmienky osvetlenia kamery na rôznych staniciach
- Mohlo by zlepšiť generalizáciu naprieč stanicami

**Náhodné orezy (random crop):**
- Riziko orezania slnka z obrázka → nezmyselný vstup
- Menej vhodné pre tento typ dát

Voľba augmentácie musí zodpovedať fyzike problému — nie každá augmentácia je vhodná pre každú úlohu.

---

### Čo by sa stalo s väčším datasetom

| Dataset | Očakávané R² | Poznámka |
|---------|-------------|----------|
| 946 (naše) | ~0.54 | Transfer learning, malý dataset |
| ~5 000 | ~0.70–0.75 | Môžeme použiť väčšie modely |
| ~50 000 | ~0.80–0.85 | Tréning od nuly je možný |
| Kompletné Eye2Sky | >0.90 | State-of-the-art, temporálne modelovanie |
