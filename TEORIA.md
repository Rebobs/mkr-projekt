# Teória — od základov po celý projekt

Tento dokument sa neodovzdáva. Slúži ako podrobný prehľad teórie pre potreby obhajoby.

---

## 1. Čo je obrázok pre počítač

Ľudské oko vidí obrázok ako scénu. Počítač vidí obrázok ako trojrozmernú tabuľku čísel.

Každý obrázok má:
- **šírku** — napr. 224 pixelov
- **výšku** — napr. 224 pixelov
- **3 kanály** — červený (R), zelený (G), modrý (B)

Každý pixel v každom kanáli je číslo od 0 do 255. Čierna = 0, biela = 255.

Jeden obrázok 224×224 teda obsahuje: `224 × 224 × 3 = 150 528 čísel`.

Keď sieť dostane obrázok na vstup, dostane práve tento obrovský zoznam čísel — nie „obrázok" tak ako ho vidíme my.

---

## 2. Čo je neurónová sieť

Neurónová sieť je matematická funkcia. Dostane čísla na vstupe, vydá čísla na výstupe.

### Jeden neurón

Základnou stavebnou jednotkou je **neurón**. Neurón:
1. Dostane niekoľko čísel na vstupe ($x_1, x_2, ..., x_n$)
2. Každý vstup vynásobí svojou váhou ($w_1, w_2, ..., w_n$)
3. Všetko sčíta, pridá bias $b$
4. Výsledok pošle cez aktivačnú funkciu

Matematicky:
$$\text{výstup} = f\left(\sum_{i=1}^{n} w_i x_i + b\right)$$

kde $f$ je aktivačná funkcia (napr. ReLU: $f(x) = \max(0, x)$).

**Váhy a bias** sú parametre — čísla ktoré sieť mení počas učenia.

### Vrstvy

Neuróny sú usporiadané do vrstiev. Výstup jednej vrstvy je vstupom ďalšej. Informácia tečie dopredu — od vstupu k výstupu. Tomu hovoríme **forward pass**.

```
vstup (obrázok) → vrstva 1 → vrstva 2 → ... → výstup (číslo)
```

### Konvolučné vrstvy (CNN)

Naše siete sú **konvolučné** (Convolutional Neural Network). Namiesto toho aby každý neurón bol prepojený so všetkými vstupmi (čo by bolo pri obrázku 150 000 prepojení na jeden neurón), konvolučné neuróny sa pozerajú vždy len na malé okienko obrázka (napr. 3×3 pixely).

Toto okienko sa posúva po celom obrázku. Táto operácia sa volá **konvolúcia** a je efektívna na rozpoznávanie lokálnych vzorov (hrany, rohy, textúry).

---

## 3. Transfer learning — prečo nestačí trénovať od nuly

### Problém s malým datasetom

Sieť ResNet18 má **11,2 milióna parametrov**. Aby sa sieť naučila nastaviť 11 miliónov čísel správne, potrebuje veľa príkladov.

My máme 946 obrázkov. To je extrémne málo.

Ak by sme trénovali od náhodných váh, sieť by sa buď vôbec nenaučila nič, alebo by sa „naučila naspamäť" trénovacie dáta a na nových obrázkoch by zlyhala. Tomu hovoríme **overfitting** (preučenie).

### Riešenie: predtrénované váhy

Siete ResNet18, EfficientNet a MobileNet boli natrénované na **ImageNete** — datasete s 1,2 milióna obrázkov a 1000 kategóriami (mačky, psy, autá, huby...).

Po tomto trénovaní siete vedia rozpoznávať:
- v prvých vrstvách: hrany, farebné prechody
- v stredných vrstvách: textúry, tvary
- v hlbších vrstvách: časti objektov (koleso, ucho, krídlo)

Tieto znalosti sú **prenositeľné**. Schopnosť rozpoznávať hrany a textúry je užitočná aj pre fotky oblohy.

### Čo presne meníme

Pôvodná výstupná vrstva siete vydáva 1000 čísel (pravdepodobnosť každej z 1000 tried ImageNetu). My potrebujeme jedno číslo. Preto:

```
[predtrénované vrstvy: rozpoznávajú vizuálne vzory]
                ↓
[nová výstupná vrstva: nn.Linear(512, 1)]
                ↓
        predikcia žiarenia (W/m²)
```

`nn.Linear(512, 1)` je lineárna vrstva: 512 vstupov (výstup poslednej predtrénovanej vrstvy), 1 výstup (hodnota žiarenia). Matematicky:

$$\hat{y} = \mathbf{w}^T \mathbf{x} + b$$

kde $\mathbf{x}$ je 512-dimenzionálny vektor z predposlednej vrstvy, $\mathbf{w}$ je 512 váh a $b$ je bias.

Predtrénované váhy sa dolaďujú (fine-tuning) — nie sú zmrazené, ale začínajú z dobrého bodu namiesto z náhodného.

---

## 4. Trénovací cyklus krok po kroku

### Krok 1: Forward pass

Pošleme dávku 32 obrázkov cez sieť. Každý obrázok je tenzor tvaru `(3, 224, 224)`. Dávka má tvar `(32, 3, 224, 224)`. Na výstupe dostaneme 32 čísel — predikcie žiarenia.

### Krok 2: Výpočet straty (loss)

Porovnáme predikcie so skutočnými hodnotami pomocou **MSE**:

$$\mathcal{L} = \frac{1}{32} \sum_{i=1}^{32} (\hat{y}_i - y_i)^2$$

Strata je jedno číslo — miera celkovej chyby na tejto dávke.

### Krok 3: Backpropagation

Toto je algoritmus, ktorý vypočíta gradient straty voči každej váhe siete.

**Gradient** $\frac{\partial \mathcal{L}}{\partial w}$ hovorí: „ak zvýšim váhu $w$ o veľmi malú hodnotu $\epsilon$, o koľko sa zmení strata?"

Backpropagation prechádza sieť odzadu (od výstupu k vstupu) a aplikuje **chain rule** (pravidlo reťazenia derivácií):

$$\frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h} \cdot \frac{\partial h}{\partial w_1}$$

kde $h$ je medziľahlý výstup. V sieti s miliónmi váh sa toto robí automaticky — v PyTorche stačí zavolať `loss.backward()`.

### Krok 4: Aktualizácia váh (Adam optimizer)

Základný gradient descent by aktualizoval váhy takto:

$$w \leftarrow w - \alpha \cdot \frac{\partial \mathcal{L}}{\partial w}$$

kde $\alpha$ je learning rate (krok). Toto je ale primitívne — rovnaký krok pre všetky váhy.

**Adam** (Adaptive Moment Estimation) je sofistikovanejší. Sleduje pre každú váhu:
- $m_t$ — exponenciálny kĺzavý priemer gradientov (smer pohybu)
- $v_t$ — exponenciálny kĺzavý priemer kvadrátov gradientov (variabilita)

Aktualizácia:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$w \leftarrow w - \alpha \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}$$

Štandardné hodnoty: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

Efekt: váhy ktoré oscilujú dostanú menší efektívny krok (veľké $v_t$), váhy ktoré konzistentne idú jedným smerom dostanú väčší. Adam konverguje rýchlejšie a stabilnejšie ako základný gradient descent.

### Krok 5: Opakuj

Toto sa opakuje pre každú dávku, pre každú epochu. Po každej epoche vyhodnotíme validačnú množinu (bez aktualizácie váh) — vidíme, či sa model zlepšuje na dátach ktoré nevidel počas trénovania.

---

## 5. Rozdelenie datasetu a prečo

### Tréning (70 %) — 662 obrázkov

Tieto dáta sieť vidí počas trénovania. Na nich sa počítajú gradienty a aktualizujú váhy.

### Validácia (15 %) — 142 obrázkov

Tieto dáta sieť nevidí počas trénovania (nepoužívajú sa na backpropagation). Po každej epoche na nich vyhodnotíme model — sledujeme, či sa model zlepšuje na nevidených dátach, alebo len na trénovacích (overfitting).

Validácia slúži aj na **early stopping** a výber najlepšej epochy.

### Test (15 %) — 142 obrázkov

Tieto dáta sa použijú **iba raz** — na záverečné vyhodnotenie najlepšieho modelu. Ak by sme optimalizovali na testovacej množine (napr. ladili hyperparametre podľa testovacích výsledkov), výsledky by boli optimisticky skreslené a nepredstavovali by reálny výkon.

**Prečo je toto dôležité?** Ak by sme použili test na ladenie, efektívne by sme ho zaradili do trénovania. Reálny výkon na úplne nových dátach by bol horší.

---

## 6. Overfitting a ako mu predchádzame

**Overfitting** nastane keď sa model naučí trénovacie dáta naspamäť — vrátane ich šumu a zvláštností — namiesto toho aby sa naučil všeobecné vzory.

Vizuálny príklad:
```
Trénovacia strata:  ↓↓↓↓↓↓↓↓↓↓  (klesá)
Validačná MAE:      ↓↓↓↓↑↑↑↑↑↑  (klesá, potom rastie)
```

Keď validačná MAE začne rásť, model sa preučuje.

### Early stopping

Sledujeme validačné MAE. Ak sa nezlepší 7 epoch za sebou, zastavíme tréning a obnovíme váhy z epochy kde bola validačná MAE najlepšia.

Toto je dôvod prečo `best_weights` ukladáme počas trénovania — nie posledné váhy, ale najlepšie.

### Transfer learning sám o sebe pomáha

Predtrénované váhy sú dobrým štartovacím bodom. Model nemusí objaviť od nuly ako vyzerajú hrany — už to vie. Preto sa rýchlejšie naučí zmysluplné veci a menej preučí šum.

---

## 7. Normalizácia — prečo a ako

Pred vstupom do siete normalizujeme pixely:

$$x'_c = \frac{x_c - \mu_c}{\sigma_c}$$

Pre každý kanál $c \in \{R, G, B\}$ zvlášť:
- $\mu = [0.485,\ 0.456,\ 0.406]$ — priemerné hodnoty pixelov ImageNetu
- $\sigma = [0.229,\ 0.224,\ 0.225]$ — štandardné odchýlky

**Prečo tieto konkrétne čísla?** Siete boli predtrénované na ImageNete a ich váhy sú kalibrované na vstup s týmto štatistickým rozložením. Ak by sme dali surové pixely (0–1), výstupy prvých vrstiev by boli úplne iné ako pri predtrénovaní — transfer learning by nefungoval správne.

**Prečo normalizácia všeobecne?** Bez normalizácie by rôzne kanály mali rôzne rozsahy. Optimizer by musel nastaviť veľmi rôzne veľké gradienty pre rôzne váhy, čo spomaluje konvergenciu.

---

## 8. Augmentácia — teória a prax

**Augmentácia** je umelé zväčšovanie trénovacej množiny transformáciami existujúcich obrázkov. Logika: ak model vidí obrázok mačky otočenej o 90°, naučí sa, že otočenie nie je dôležité pre rozpoznanie mačky.

### Kedy augmentácia pomáha

Pri úlohách kde transformácia **nemení** triedu alebo hodnotu:
- rozpoznávanie buniek pod mikroskopom (bunka otočená o 90° je stále bunka)
- röntgenové snímky (zápal je zápal v akejkoľvek orientácii)
- satelitné snímky terénu (pole otočené o 90° je stále pole)

### Prečo pri oblohe nepomáha

Fotka oblohy má **gravitáciou danú orientáciu**:
- horizont je vždy dole
- slnko stúpa a klesá — jeho výška nad horizontom priamo určuje intenzitu žiarenia
- oblaky v rôznych výškach nesú rôznu informáciu

Keď otočíme obrázok o 90°, horizont je zrazu na boku. Model dostáva fyzikálne nezmyselnú situáciu. Namiesto toho aby sa naučil „slnko vysoko = veľa žiarenia", musí sa naučiť ignorovať orientáciu — čím stráca kľúčovú informáciu.

**Výsledok z nášho experimentu:** augmentácia zhoršila MAE o 5–20 W/m² vo všetkých konfiguráciách. Toto je zaujímavý výsledok — nie chyba, ale zmysluplný fyzikálny záver.

---

## 9. Metriky — hlbší pohľad

### MAE vs RMSE

Obe merajú chybu predikcie, ale rôzne:

$$\text{MAE} = \frac{1}{n}\sum|y_i - \hat{y}_i|, \qquad \text{RMSE} = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$$

Kľúčový rozdiel: RMSE **kvadraticky penalizuje** veľké odchýlky. Ak máme jeden extrémne zlý odhad (napr. 500 W/m² chyba), RMSE to zachytí oveľa viac ako MAE.

Ak RMSE >> MAE, máme v dátach outliere — niekoľko vzoriek kde sa model veľmi mýli.

### R² — prečo môže byť záporné

$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

- $\text{SS}_{\text{res}}$ — suma kvadrátov zvyškov (chyba modelu)
- $\text{SS}_{\text{tot}}$ — celková suma kvadrátov (variabilita dát)

Ak je chyba modelu väčšia ako variabilita dát ($\text{SS}_{\text{res}} > \text{SS}_{\text{tot}}$), R² je záporné.

To nastalo pri 25 % datasetu (175 obrázkov). Model sa nenaučil nič užitočné — bol horší ako keby sme vždy predpovedali priemer.

**Intuícia:** Priemerné žiarenie v datasete je napr. 400 W/m². Ak vždy predpovedám 400, dostanem $R^2 = 0$. Ak predpovedám horšie ako vždy predpovedať priemer, $R^2 < 0$.

---

## 10. Tri porovnávané architektúry

### ResNet18

ResNet (Residual Network) zaviedol kľúčovú inováciu: **skip connections** (reziduálne spojenia). Namiesto toho aby každá vrstva len transformovala vstup, pridá k transformácii aj pôvodný vstup:

$$\text{výstup} = F(x) + x$$

kde $F(x)$ je to čo sa vrstva naučila, $x$ je pôvodný vstup. Prečo to pomáha? Ak je $F(x) = 0$ (vrstva sa nenaučila nič), výstup je stále $x$ — gradient preteká priamo dozadu bez degradácie. Umožňuje trénovanie oveľa hlbších sietí.

ResNet18 má 18 vrstiev, 11,2M parametrov. Napriek tomu že je najväčší z trojice, vyhral — pravdepodobne preto, že jeho architektonické vzory lepšie zachytávajú vizuálne prvky oblohy.

### EfficientNet-B0

EfficientNet škáluje sieť systematicky — zväčšuje šírku, hĺbku aj rozlíšenie vstupu proporcionálne. B0 je základná (najmenšia) verzia. Používa **depthwise separable convolutions** — efektívnejšia varianta konvolúcie, ktorá rozdeľuje priestorovú a kanálovú konvolúciu:

Štandardná konvolúcia: $O(k^2 \cdot C_{in} \cdot C_{out})$ operácií
Depthwise separable: $O(k^2 \cdot C_{in} + C_{in} \cdot C_{out})$ — výrazne menej

Paradoxne bol pomalší na trénovaní napriek menšiemu počtu parametrov — jeho architektúra má viac sekvenčných operácií čo je menej paralelizovateľné na GPU.

### MobileNetV3-Small

Navrhnutý pre mobilné zariadenia — minimálny počet parametrov (1,5M) pri prijateľnom výkone. Používa **squeeze-and-excitation bloky**: globálne priemerné poolovanie → dve FC vrstvy → škálovanie kanálov. Sieť sa naučí, ktoré kanály sú dôležité pre daný vstup.

Bol najrýchlejší (~29s/experiment) ale nedosiahol kvalitu ResNetu pri plnom datasete.

---

## 11. Veľkosť vstupu — prečo 128px vyhral

Sieť spracováva obrázky fixnej veľkosti. My testujeme 128×128, 224×224, 320×320 pixelov.

Väčší vstup v princípe nesie viac informácie. Ale:

**Väčší vstup = väčší model.** Prvá konvolučná vrstva musí spracovať oveľa viac pixelov — viac výpočtov, viac váh. Sieť s väčším vstupom je efektívne komplikovanejšia.

**Komplikovanejšia sieť potrebuje viac dát.** Pri 946 obrázkoch väčší vstup zvyšuje riziko overfittingu.

**Fyzikálna úvaha:** pre odhad žiarenia nie sú dôležité detaily (ostrosť hrán oblakov), ale globálne rozloženie (kde je slnko, koľko oblohy pokrývajú oblaky). 128×128 zachytáva toto globálne rozloženie dostatočne.

---

## 12. Batch size — prečo 32

**Batch size** je počet obrázkov spracovaných naraz pred jednou aktualizáciou váh.

**Malý batch (napr. 1–4):**
- gradient je vypočítaný z jednej vzorky → šumivý, nestabilný
- časté aktualizácie → pomalý tréning na GPU (GPU nie je plne vyťažené)
- môže pomôcť uniknúť lokálnym minimám

**Veľký batch (napr. 256–1024):**
- gradient je presnejší (priemerovaný cez viac vzoriek)
- GPU je plne vyťažené → rýchly tréning
- môže konvergovať do ostrých miním (horšia generalizácia)
- potrebuje viac GPU pamäte

**32** je štandardný kompromis — dosť veľký na stabilný gradient, dosť malý aby sa zmestil do pamäte a zachoval dobrú generalizáciu.

---

## 13. Čo je learning rate a prečo záleží

Learning rate $\alpha$ určuje veľkosť kroku pri aktualizácii váh:

$$w \leftarrow w - \alpha \cdot \nabla_w \mathcal{L}$$

Predstav si hľadanie minima na kopcovitej krajine:
- $\alpha$ príliš malý → robíš miniatúrne kroky, trvá to večne
- $\alpha$ príliš veľký → preskočíš cez minimum, osciluje si, nekonverguješ
- $\alpha$ správny → plynule klesáš do minima

My testujeme tri hodnoty:
- `0.001` — štandardný Adam LR, najčastejší v literatúre
- `0.0001` — opatrnejší, pomalší
- `0.01` — agresívny, môže byť nestabilný

---

## 14. Celkový tok experimentu

```
labels.csv + obrázky
        ↓
load_image_cache()  →  obrázky v RAM (predresizované na 320px)
        ↓
pre každé LR × train_frac × model × size × aug:
        ↓
    get_loaders()  →  train/val/test DataLoader
        ↓
    get_model()    →  predtrénovaná sieť, vymenená výstupná vrstva
        ↓
    tréning (max 20 epoch, early stopping PATIENCE=7):
        - train_epoch() → forward pass, MSE loss, backprop, Adam krok
        - evaluate() na val  → ak najlepšie MAE, ulož váhy
        ↓
    evaluate() na test  →  MAE, RMSE, R², predikcie
        ↓
    výsledok uložený do results.json
        ↓
plot.py  →  best_per_config()  →  grafy
```

Celkovo: **216 experimentov** (3 LR × 4 fracs × 3 modely × 3 veľkosti × 2 aug stavy).

---

## 15. Čo sa stane ak zmením hodnoty

### `BATCH_SIZE = 32`

**Ak zvýšim na 128 alebo 256:**
- Každý krok rýchlejší, gradient presnejší (priemer cez viac vzoriek)
- Ale: model konverguje do „ostrých" miním → horšia generalizácia na nové dáta
- Pri väčšom batch sa odporúča zvýšiť aj LR (lineárne: `lr × batch/32`)
- Pri 946 vzorkách: menej krokov za epochu = menej aktualizácií

**Ak znížim na 8:**
- Gradient šumivý → nestabilné trénovanie, pomalší tréning celkovo
- Môže pomôcť uniknúť lokálnym minimám (šum = náhodné skoky)

**Záver:** 32 je vyvážená voľba pre tento dataset a GPU.

---

### `LEARNING_RATES = [0.001, 0.0001, 0.01]`

**Ak použijem príliš veľké LR (napr. 0.1):**
- Optimizer „preskakuje" cez minimu → val MAE osciluje alebo rastie
- Model nekonverguje, výsledky sú náhodné

**Ak použijem príliš malé LR (napr. 0.000001):**
- Váhy sa menia len minimálne → za 20 epoch skoro žiadne zlepšenie
- Tréning je stabilný, ale nezmyselne pomalý

**Prečo testujeme tri hodnoty:** Optimálne LR závisí od architektúry a datasetu. Bez experimentu to nevieme vopred. `0.001` je štandardné pre Adam — čo sa potvrdilo aj v našich výsledkoch.

---

### `EPOCHS = 20`

**Ak zvýšim na 100:**
- Early stopping aj tak zastaví tréning najneskôr `PATIENCE` epoch po najlepšej
- Fakticky: pri `PATIENCE=7` sa tréning zastaví max. 7 epoch po optimálnej — zvýšenie EPOCHS nad ~30 nemá takmer žiadny efekt
- Výnimka: s nižším LR by sieť konvergovala pomalšie a viac epoch by pomohlo

**Ak znížim na 5:**
- Model nemá čas konvergovať, validačná MAE ešte klesá ale zastavíme predčasne
- Výsledky výrazne horšie

---

### `PATIENCE = 7`

**Ak zvýšim na 15:**
- Tréning trvá dlhšie, tolerujeme dlhšiu stagnáciu
- Môže zachytiť neskorú konvergenciu (MAE niekedy stagnuje a potom skokovo klesne)
- Riziko: model sa medzičasom začne preučovať

**Ak znížim na 2–3:**
- Zastavíme príliš skoro — model možno ešte konvergoval
- Vhodné len pre rýchle orientačné experimenty

---

### `INPUT_SIZES = [128, 224, 320]`

**Ak pridám menší vstup (64px):**
- Možná strata informácie — pozícia slnka nemusí byť presne zachytená
- Rýchlejší tréning
- Pri oblohe kde záleží na globálnom rozložení môže byť 64px hranične dostačujúce

**Ak pridám väčší vstup (512px):**
- Výrazne pomalší tréning (výpočty rastú kvadraticky s rozlíšením)
- Pri 946 obrázkoch takmer istý overfitting
- Väčšie GPU pamäťové nároky

**Prečo 128px vyhral:** Pre odhad žiarenia stačí vedieť „kde je slnko a koľko oblohy pokrývajú oblaky". 128px to zachytí. Väčší vstup pridáva komplexitu bez pridanej hodnoty pri malom datasete.

---

### `TRAIN_FRACS = [0.25, 0.5, 0.75, 1.0]`

Toto nie je hyperparameter — je to experiment na pochopenie koľko dát potrebujeme.

| Frakcia | Vzorky | Výsledok |
|---------|--------|---------|
| 25 % | ~175 | R² záporné — model horší ako predpovedanie priemeru |
| 50 % | ~330 | Výrazné zlepšenie — model začína zachytávať vzory |
| 75 % | ~497 | Ďalšie zlepšenie, krivka sa vyrovnáva |
| 100 % | ~662 | Najlepší výsledok |

**Záver:** Minimum pre zmysluplné výsledky je ~300–400 vzoriek pri transfer learningu.

---

### Výber modelu

**Ak by som pridal ResNet50 (25M parametrov):**
- Hlbšia sieť, viac parametrov
- Pri 946 obrázkoch pravdepodobne silný overfitting
- Tréning pomalší, výsledky môžu byť horšie ako ResNet18

**Ak by som pridal ViT (Vision Transformer):**
- Transformery vyžadujú omnoho viac dát ako CNN (typicky stovky tisíc obrázkov)
- Pri 946 obrázkoch by takmer určite zlyhali
- CNN majú vstavanú indukčnú biasovosť (lokálne spojenia, transliačná invariantnosť) ktorá pomáha pri malom datasete

---

### Augmentácia — čo iné by sme mohli skúsiť

Naša rotačná augmentácia ukázala, že nepomáha. Iné augmentácie by mohli byť vhodnejšie:

**Horizontálny flip (zrkadlenie):**
- Obloha je fyzikálne symetrická ľavo-pravý flip — slnko môže byť na ľavej alebo pravej strane
- Toto by mohlo pomôcť, keďže nenarúša horizont

**Zmena jasu/kontrastu:**
- Simuluje rôzne podmienky osvetlenia kamery na rôznych staniciach
- Mohlo by pomôcť generalizácii

**Náhodné orezy (random crop):**
- Riziko: odrežeme slnko → nezmyselný vstup
- Menej vhodné pre tento typ dát

**Záver:** Voľba augmentácie musí zodpovedať fyzike problému. Nie každá augmentácia je vhodná pre každú úlohu.

---

### Čo by sa stalo s väčším datasetom

| Dataset | Očakávané R² | Poznámka |
|---------|-------------|---------|
| 946 (naše) | ~0.54 | Transfer learning, malý dataset |
| ~5 000 | ~0.70–0.75 | Môžeme použiť väčšie modely |
| ~50 000 | ~0.80–0.85 | Tréning od nuly možný |
| Kompletné Eye2Sky | >0.90 | State-of-the-art metódy, temporálne modelovanie |
