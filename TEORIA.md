# Teória — od základov po celý projekt

Tento dokument sa neodovzdáva. Slúži ako podrobný prehľad teórie pre potreby obhajoby.

---

## 1. Čo je obrázok pre počítač

Ľudské oko vidí obrázok ako scénu. Počítač vidí obrázok ako tabuľku čísel.

Každý pixel má tri hodnoty — červená (R), zelená (G), modrá (B). Každá hodnota je číslo od 0 do 255. Čierna = 0, biela = 255.

Príklad jedného pixela:
```
jasne červený pixel:  R=255, G=0,   B=0
čisto biely pixel:    R=255, G=255, B=255
čisto čierny pixel:   R=0,   G=0,   B=0
obloha (modrá):       R=100, G=149, B=237
```

### Čo je tenzor a prečo tvar `(3, 224, 224)`

**Tenzor** = n-rozmerná tabuľka čísel. Nie je to nič iné ako zovšeobecnenie pojmu „tabuľka":

- **1D tenzor** = zoznam: `[10, 20, 30]`
- **2D tenzor** = tabuľka (riadky × stĺpce), napr. tabuľka v Exceli
- **3D tenzor** = viac tabuliek položených na sebe — kocka čísel

Jeden obrázok 224×224 pixelov je **3D tenzor tvaru `(3, 224, 224)`**:
- prvý rozmer `3` = tri farebné kanály (R, G, B)
- druhý rozmer `224` = výška (224 riadkov pixelov)
- tretí rozmer `224` = šírka (224 stĺpcov pixelov)

Predstav si to ako tri vrstvy tabuliek položené na sebe:

```
Kanál 1 (Červená):         Kanál 2 (Zelená):          Kanál 3 (Modrá):
┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
│ 120  98  75  ...   │     │  30  45  60  ...   │     │ 200 180 170  ...   │
│  85  92  88  ...   │     │  40  38  55  ...   │     │ 190 175 165  ...   │
│  ...               │     │  ...               │     │  ...               │
└────────────────────┘     └────────────────────┘     └────────────────────┘
       224 × 224                  224 × 224                  224 × 224
```

Celkovo: `3 × 224 × 224 = 150 528 čísel` na jeden obrázok.

### Prečo dávka 32 — tvar `(32, 3, 224, 224)`

Sieť netrénujeme obrázok po obrázku — posielame 32 naraz (dávka / batch). Pridáme ďalší rozmer:

```
(32,      3,       224,    224  )
  ↑       ↑         ↑       ↑
počet  kanály    výška   šírka
obrázkov
```

Celkovo: `32 × 3 × 224 × 224 = 4 816 896 čísel` — toto pošleme cez sieť v jednom kroku.

Na výstupe dostaneme 32 predikcií — jedno číslo pre každý obrázok z dávky.

---

## 2. Čo je neurónová sieť

Neurónová sieť je matematická funkcia. Dostane čísla na vstupe, vydá čísla na výstupe.

### Jeden neurón

Základnou stavebnou jednotkou je **neurón**:
1. Dostane niekoľko čísel na vstupe ($x_1, x_2, ..., x_n$)
2. Každý vstup vynásobí svojou váhou ($w_1, w_2, ..., w_n$)
3. Všetko sčíta, pridá bias $b$
4. Výsledok pošle cez aktivačnú funkciu

Matematicky:
$$\text{výstup} = f\left(\sum_{i=1}^{n} w_i x_i + b\right)$$

**Konkrétny príklad** — neurón s dvoma vstupmi:
```
vstupy:   x₁ = 0.5,  x₂ = 0.8
váhy:     w₁ = 2.0,  w₂ = -1.0
bias:     b  = 0.1

súčet = (0.5 × 2.0) + (0.8 × -1.0) + 0.1
      =  1.0  +  (-0.8)  + 0.1
      =  0.3

po ReLU: max(0, 0.3) = 0.3   ← výstup neurónu
```

**Váhy a bias** sú parametre — čísla ktoré sieť mení počas učenia. Na začiatku sú náhodné, po trénovaní sú nastavené tak aby sieť dávala správne odpovede.

**ReLU** (Rectified Linear Unit) je najčastejšia aktivačná funkcia: $f(x) = \max(0, x)$. Jednoducho povedané — záporné čísla zmení na nulu, kladné nechá. Bez aktivačnej funkcie by celá sieť bola len jedna lineárna rovnica, čo by nestačilo na zachytenie zložitých vzorov.

### Vrstvy

Neuróny sú usporiadané do vrstiev. Výstup jednej vrstvy je vstupom ďalšej. Informácia tečie dopredu — od vstupu k výstupu. Tomu hovoríme **forward pass**.

```
vstup (obrázok) → vrstva 1 → vrstva 2 → ... → výstup (číslo)
```

### Konvolučné vrstvy (CNN)

Naše siete sú **konvolučné** (Convolutional Neural Network).

Naivný prístup by bol: každý neurón je prepojený s každým pixelom. Pri 224×224 obrázku by mal jeden neurón 150 528 vstupov — to je príliš veľa parametrov.

Konvolučné neuróny sa pozerajú len na malé **okienko** obrázka (napr. 3×3 pixely). Toto okienko sa posúva po celom obrázku — zľava doprava, zhora nadol — a na každej pozícii vypočíta jednu hodnotu. Výsledkom je nová tabuľka čísel, tzv. **feature mapa** — zachytáva, kde v obrázku sa nachádza konkrétny vzor (hrana, farba, textúra).

```
pôvodný obrázok          filter (3×3)          feature mapa
┌──────────────┐          ┌───────┐             ┌──────────┐
│ 1  2  3  4   │          │ 1  0  │             │ hodnota  │
│ 5  6  7  8   │  ×  ...  │-1  1  │   =   ...   │  každej  │
│ 9 10 11 12   │          │ 0  1  │             │ pozície  │
│13 14 15 16   │          └───────┘             └──────────┘
└──────────────┘
```

Hodnoty filtra sú váhy — sieť sa ich naučí sama. Prvé vrstvy sa naučia jednoduché filtre (detekcia hrán), hlbšie vrstvy zložitejšie (textúry, tvary, objekty).

---

## 3. Transfer learning — prečo nestačí trénovať od nuly

### Problém s malým datasetom

Sieť ResNet18 má **11,2 milióna parametrov** (váh). Každú váhu treba nastaviť na správnu hodnotu. Na to potrebujeme veľa príkladov — inak sieť „uhádne" správne odpovede pre trénovacie dáta, ale na nových zlyhá.

My máme 946 obrázkov. To je extrémne málo.

Keby sme trénovali od náhodných váh, sieť by sa buď vôbec nenaučila nič, alebo by sa naučila trénovacie dáta **naspamäť** — vrátane ich náhodného šumu — a na nových obrázkoch by zlyhala. Tomu hovoríme **overfitting** (preučenie).

### Riešenie: predtrénované váhy

Siete ResNet18, EfficientNet a MobileNet boli natrénované na **ImageNete** — datasete s 1,2 milióna obrázkov a 1000 kategóriami (mačky, psy, autá, huby...).

Po tomto trénovaní siete vedia rozpoznávať:
- v prvých vrstvách: hrany, farebné prechody
- v stredných vrstvách: textúry, tvary
- v hlbších vrstvách: časti objektov (koleso, ucho, krídlo)

Tieto znalosti sú **prenositeľné**. Schopnosť rozpoznávať hrany a textúry je užitočná aj pre fotky oblohy.

### Čo presne meníme

Pôvodná výstupná vrstva siete vydáva 1000 čísel (pravdepodobnosť každej z 1000 tried ImageNetu). My potrebujeme jedno číslo — hodnotu žiarenia. Preto:

```
[predtrénované vrstvy: rozpoznávajú vizuálne vzory]
                ↓
[nová výstupná vrstva: nn.Linear(512, 1)]
                ↓
        predikcia žiarenia (W/m²)
```

`nn.Linear(512, 1)` je lineárna vrstva: zoberie 512 čísel z predposlednej vrstvy a vypočíta z nich jedno číslo. Je to v podstate vážený súčet:

$$\hat{y} = w_1 x_1 + w_2 x_2 + \ldots + w_{512} x_{512} + b$$

kde $x_1 ... x_{512}$ sú výstupy predposlednej vrstvy, $w_1 ... w_{512}$ sú váhy ktoré sa naučíme, $b$ je bias.

Predtrénované váhy sa dolaďujú (**fine-tuning**) — nie sú zmrazené, ale začínajú z dobrého bodu namiesto z náhodného. Preto stačí oveľa menej dát.

---

## 4. Trénovací cyklus krok po kroku

### Krok 1: Forward pass (dopredný prechod)

Vezmeme dávku 32 obrázkov. Každý má tvar `(3, 224, 224)` — 3 kanály, 224×224 pixelov. Dávka má tvar `(32, 3, 224, 224)`.

Tieto čísla prechádzajú vrstvou za vrstvou celou sieťou. Každá vrstva transformuje dáta — filtruje, zmenšuje, kombinuje. Na konci dostaneme 32 čísel, každé je predikcia žiarenia pre jeden obrázok.

```
obrázok 1  (150 528 čísel) → [sieť] → 423.5 W/m²
obrázok 2  (150 528 čísel) → [sieť] → 187.2 W/m²
...
obrázok 32 (150 528 čísel) → [sieť] → 651.0 W/m²
```

### Krok 2: Výpočet straty (loss)

Porovnáme predikcie so skutočnými hodnotami. Potrebujeme jedno číslo ktoré hovorí „ako veľmi sme sa mýlili". Tomu hovoríme **strata** (loss).

Používame **MSE** (Mean Squared Error — stredná kvadratická chyba):

$$\mathcal{L} = \frac{1}{32} \sum_{i=1}^{32} (\hat{y}_i - y_i)^2$$

Rozmeníme si to na súčasti:
- $\hat{y}_i$ = predikcia siete pre obrázok $i$ (čo sieť povedala)
- $y_i$ = skutočná hodnota žiarenia pre obrázok $i$ (čo nameril pyranometer)
- $(\hat{y}_i - y_i)$ = rozdiel — o koľko sme sa mýlili
- $(\hat{y}_i - y_i)^2$ = rozdiel umocnený na druhú
- $\frac{1}{32} \sum$ = spriemerujeme cez všetkých 32 obrázkov v dávke

**Konkrétny príklad s číslami** pre 4 obrázky z dávky:

```
obrázok │ predikcia (ŷ) │ skutočnosť (y) │ rozdiel │ rozdiel²
────────┼───────────────┼────────────────┼─────────┼─────────
      1 │     423.5     │     500.0      │  -76.5  │  5852.3
      2 │     187.2     │     200.0      │  -12.8  │   163.8
      3 │     651.0     │     600.0      │  +51.0  │  2601.0
      4 │     300.0     │     150.0      │ +150.0  │ 22500.0
────────┴───────────────┴────────────────┴─────────┴─────────
                                             priemer: 7779.3  ← MSE
```

MSE pre túto dávku = 7779.3. Toto je strata — číslo ktoré chceme minimalizovať tréningom.

**Prečo umocňujeme na druhú a nie len berieme rozdiel?**

Problém s prostým rozdielom: obrázok 1 má rozdiel −76.5, obrázok 3 má +51.0. Keby sme len sčítali, záporné a kladné chyby by sa navzájom rušili a výsledok by vyzeral lepší ako je.

Umocnenie na druhú rieši oba problémy naraz:
1. **Záporné číslo umocnené na druhú je vždy kladné** → (−76.5)² = 5852.3, (+51.0)² = 2601.0. Chyby sa nerušia.
2. **Väčšie chyby sa penalizujú výrazne viac** → chyba 150 dáva 22 500, chyba 50 dáva 2 500. Chyba 3× väčšia → penalizácia 9× väčšia. Model je teda pod tlakom opravovať najmä veľké chyby.

**Prečo práve MSE a nie MAE?**

MAE (mean absolute error) — priemerná absolútna chyba — berieme len $|\hat{y}_i - y_i|$ bez umocňovania. Je to jednoduchšie a intuitívnejšie. Ale MSE sa lepšie hodí pre tréning, pretože derivácia MSE je plynulá (kvadratická krivka má v každom bode definovaný sklon), čo zjednodušuje výpočet gradientov v kroku 3. Derivácia MAE má v bode 0 nespojitosť (zlom), čo komplikuje optimalizáciu.

Preto: **MSE používame ako trénovaciu stratu** (v kóde: `criterion = nn.MSELoss()`), ale **MAE reportujeme ako výsledok** — lebo W/m² je zrozumiteľnejšia jednotka ako (W/m²)².

Strata je jedno číslo — miera celkovej chyby na tejto dávke. Čím nižšie, tým lepšie.

### Krok 3: Backpropagation (spätné šírenie)

Teraz vieme o koľko sme sa mýlili. Potrebujeme zistiť, **ktoré váhy za to môžu** a ako ich zmeniť aby sme sa nabudúce mýlili menej.

**Gradient** je kľúčový pojem. Pre každú váhu $w$ vypočítame:

$$\frac{\partial \mathcal{L}}{\partial w}$$

Toto čítame ako „o koľko sa zmení strata ak trošku zmením váhu $w$?" Ak je gradient kladný — zvýšenie váhy zväčší stratu, teda váhu treba znížiť. Ak záporný — váhu treba zvýšiť.

Backpropagation prechádza sieť odzadu (od výstupu k vstupu) a pre každú váhu tento gradient vypočíta pomocou pravidla reťazenia derivácií. Ide o čistú matematiku — v PyTorche stačí napísať `loss.backward()` a všetky gradienty sa vypočítajú automaticky.

### Krok 4: Aktualizácia váh (Adam optimizer)

Teraz vieme smer. Aktualizujeme každú váhu:

$$w \leftarrow w - \alpha \cdot \frac{\partial \mathcal{L}}{\partial w}$$

- $\alpha$ (learning rate) = veľkosť kroku. U nás 0.001.
- Ak gradient = 2.5 a $\alpha$ = 0.001, váha sa zmení o $-0.0025$

**Adam** je vylepšenie tohto základného postupu. Problém základného gradientného zostupu je, že každá váha dostane rovnako veľký krok. Niektoré váhy by potrebovali väčší krok, iné menší.

Adam si pre každú váhu pamätá históriu predchádzajúcich gradientov:
- Ak váha **dlhodobo ide jedným smerom** → Adam jej dá väčší efektívny krok
- Ak váha **osciluje hore-dole** → Adam jej dá menší efektívny krok

Vďaka tomu Adam konverguje rýchlejšie a stabilnejšie. Konkrétne vzorce:

$$m_t = 0.9 \cdot m_{t-1} + 0.1 \cdot g_t \quad \text{(kĺzavý priemer smeru)}$$
$$v_t = 0.999 \cdot v_{t-1} + 0.001 \cdot g_t^2 \quad \text{(kĺzavý priemer veľkosti)}$$
$$w \leftarrow w - \alpha \cdot \frac{m_t}{\sqrt{v_t} + 10^{-8}}$$

Kde $g_t$ je aktuálny gradient. Delenie $\sqrt{v_t}$ znormalizuje krok — váhy s veľkými gradientmi dostanú menší efektívny krok, váhy s malými dostanú väčší.

### Krok 5: Opakuj

Toto sa opakuje pre každú dávku v datasete. Keď sieť prejde raz cez celý trénovací dataset, hovoríme že prebehla jedna **epocha**. My trénujeme max 20 epoch.

Po každej epoche vyhodnotíme model na **validačnej množine** — dátach ktoré sieť počas trénovania nevidela — a sledujeme, či sa zlepšuje.

---

## 5. Rozdelenie datasetu a prečo

946 obrázkov rozdelíme na tri časti:

### Tréning (70 %) — 662 obrázkov

Tieto dáta sieť vidí počas trénovania. Na nich sa počítajú gradienty a aktualizujú váhy.

### Validácia (15 %) — 142 obrázkov

Tieto dáta sieť **nevidí** počas trénovania — nepoužívajú sa na aktualizáciu váh. Po každej epoche na nich vyhodnotíme model. Sledujeme, či sa zlepšuje na dátach ktoré nevidela — to je skutočná miera výkonu.

Validácia slúži aj na **early stopping** a výber najlepšej epochy.

### Test (15 %) — 142 obrázkov

Tieto dáta sa použijú **iba raz** — na záverečné vyhodnotenie najlepšieho modelu.

**Prečo ich nemiešame s validáciou?** Keby sme ladili model podľa testovacích výsledkov (napr. „toto LR dáva lepší test, skúsme iné"), test by sa stal súčasťou trénovania. Výsledky by pôsobili lepšie ako v skutočnosti. Test musí byť „nevidený" až do konca — simuluje reálne nasadenie kde nemáme odpovede vopred.

---

## 6. Overfitting a ako mu predchádzame

**Overfitting** = model sa naučí trénovacie dáta naspamäť namiesto všeobecných vzorov.

Predstav si študenta ktorý sa naučí odpovede na konkrétne otázky zo skúšky naspamäť, ale nerozumie látke. Na rovakých otázkach urobí 100 %, na iných otázkach z tej istej látky prepadne.

Vizuálne — čo sa deje s MAE počas trénovania:
```
epocha:           1    2    3    4    5    6    7    8    9   10
trénovacia MAE: 200  160  130  110   95   85   78   72   68   65  ← stále klesá
validačná MAE:  210  170  140  120  108  105  107  112  118  125  ← začala rásť od epochy 6
```

Model sa zlepšuje na trénovacích dátach, ale na nevidených sa zhoršuje — preučuje sa.

### Early stopping

Sledujeme validačné MAE. Ak sa nezlepší 7 epoch za sebou (`PATIENCE = 7`), zastavíme tréning a obnovíme váhy z epochy kde bola validačná MAE najnižšia.

V príklade vyššie by sme sa zastavili po epoche 12 (6+7=13, ale optimum bolo na 5 a od 6 do 12 = 7 epoch bez zlepšenia) a vrátili by sme váhy z epochy 5.

### Transfer learning sám o sebe pomáha

Predtrénované váhy sú dobrým štartovacím bodom. Model nemusí objaviť od nuly ako vyzerajú hrany a textúry — už to vie z ImageNetu. Preto sa rýchlejšie naučí zmysluplné veci a menej skĺzne do overfittingu.

---

## 7. Normalizácia — prečo a ako

Pred vstupom do siete upravíme hodnoty pixelov:

$$x'_c = \frac{x_c - \mu_c}{\sigma_c}$$

Pre každý farebný kanál zvlášť odčítame priemer a vydelíme štandardnou odchýlkou.

**Konkrétny príklad** — jeden pixel červeného kanála:
```
pôvodná hodnota:  x = 180  (číslo 0-255, po ToTensor je to 180/255 ≈ 0.706)
priemer ImageNet: μ = 0.485
štandardná odch.: σ = 0.229

normalizovaná hodnota: (0.706 - 0.485) / 0.229 ≈ 0.965
```

**Prečo tieto konkrétne čísla (0.485, 0.229 atď.)?**
Sú to priemerné hodnoty pixelov a štandardné odchýlky vypočítané z celého ImageNet datasetu. Siete boli predtrénované na obrázkoch normalizovaných týmito číslami. Ak by sme dali iné čísla, prvé vrstvy siete by dostali úplne iný rozsah hodnôt — váhy by fungovali nesprávne a transfer learning by nepomohol.

**Prečo normalizácia vôbec pomáha?**
Bez nej by rôzne kanály mali rôzne rozsahy. Červený kanál môže mať priemer 150, modrý 80. Optimizer by musel nastaviť inak veľké kroky pre rôzne časti siete, čo spomaluje učenie. Po normalizácii majú všetky vstupy podobný rozsah (okolo 0, väčšinou -2 až +2) a sieť sa učí stabilnejšie.

---

## 8. Augmentácia — teória a prax

**Augmentácia** = umelé zväčšovanie trénovacej množiny. Každý obrázok sa pri každej epoche náhodne transformuje — sieť teda vidí rôzne verzie toho istého obrázka.

**Logika:** Ak model vidí obrázok mačky otočenej o 90°, naučí sa že otočenie neovplyvňuje to čo je na obrázku. Trénovacia množina sa efektívne zväčší (každý obrázok je 4 rôznych otočení = 4× viac dát).

### Kedy augmentácia pomáha

Pri úlohách kde transformácia **nemení** výstup:
- rozpoznávanie buniek pod mikroskopom — bunka otočená o 90° je stále bunka
- röntgenové snímky — zápal je zápal v akejkoľvek orientácii
- satelitné snímky terénu — pole je pole

### Prečo pri oblohe nepomáha

Fotka oblohy má **fyzikálne pevnú orientáciu** — gravitácia určuje kde je hore:
- horizont je vždy dole
- slnko sa pohybuje v hornej časti neba — jeho výška nad horizontom priamo určuje intenzitu žiarenia
- oblaky nízko = iná situácia ako oblaky vysoko

Keď otočíme obrázok o 90°, horizont je zrazu na boku a slnko je zboku. Toto je fyzikálne nezmyselná situácia. Model musí trénovať na reálnych aj otočených obrázkoch — namiesto toho aby sa naučil „slnko vysoko = veľa žiarenia", musí sa naučiť ignorovať orientáciu. Tým stráca práve tú informáciu ktorá je kľúčová.

**Výsledok:** augmentácia zhoršila MAE o 5–20 W/m² vo všetkých modeloch. Nie chyba — zmysluplný fyzikálny záver.

---

## 9. Metriky — hlbší pohľad

Metriky sú čísla ktoré merajú kvalitu modelu. Vypočítame ich na testovacej množine po skončení trénovania.

### MAE — Mean Absolute Error (stredná absolútna chyba)

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

Postup výpočtu:
1. Pre každý testovací obrázok: vezmi rozdiel medzi skutočnou hodnotou a predikciou, vždy ako kladné číslo
2. Všetky tieto rozdiely spriemeruj

**Príklad:**
```
skutočné:   500, 300, 100
predikcie:  450, 340,  80
rozdiely:    50,  40,  20
MAE = (50 + 40 + 20) / 3 = 36.7 W/m²
```

MAE = 36.7 hovorí: priemerne sa mýlime o 36.7 W/m². Ľahko interpretovateľné.

### RMSE — Root Mean Squared Error

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

Podobné ako MAE, ale rozdiely sa najprv kvadrujú, potom spriemerujú, potom odmocnia.

**Kľúčový rozdiel oproti MAE:** kvadrát penalizuje väčšie chyby oveľa viac. Chyba 100 W/m² = 10 000 vo výpočte, chyba 10 W/m² = len 100. Preto RMSE zachytí, či máme niekoľko extrémne zlých predikcií (outlierov).

Ak RMSE >> MAE, model sa na niektorých obrázkoch mýli extrémne veľa.

### R² — koeficient determinácie

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

kde $\bar{y}$ je priemer všetkých skutočných hodnôt.

**Čo to znamená intuitívne:**

Menovateľ ($\sum(y_i - \bar{y})^2$) = celková variabilita v dátach — o koľko sa hodnoty žiarenia líšia navzájom.

Čitateľ ($\sum(y_i - \hat{y}_i)^2$) = chyba modelu — o koľko sa predikcie líšia od skutočnosti.

R² = 1 − (chyba modelu / celková variabilita)

**Príklady:**
- R² = 1.0 → model je dokonalý, žiadna chyba
- R² = 0.54 → model vysvetlil 54 % variability, 46 % ostalo nevysvetlených
- R² = 0 → model predpovedá vždy len priemer, rovnako neužitočný
- R² < 0 → model je **horší** ako vždy predpovedať priemer — toto nastalo pri 25 % datasetu

**Prečo R² < 0?** Ak je sieť úplne zmätená (napr. 175 obrázkov nestačilo), jej chyba je väčšia ako keby vždy tipla priemer. Čitateľ > menovateľ → R² záporné.

---

## 10. Tri porovnávané architektúry

### ResNet18

ResNet (Residual Network, 2015) zaviedol kľúčovú inováciu: **reziduálne spojenia** (skip connections).

Bežná vrstva: `výstup = F(vstup)` — vrstva transformuje vstup.

ResNet vrstva: `výstup = F(vstup) + vstup` — k transformácii sa pripočíta pôvodný vstup.

```
vstup ──────────────────────────┐
  │                             │
  ↓                             │
[konvolúcia + ReLU]             │  (toto je "skratka")
  │                             │
  ↓                             │
[konvolúcia]                    │
  │                             │
  └──── + ←───────────────── ───┘
         ↓
       výstup
```

**Prečo to pomáha?** Problém hlbokých sietí: gradient sa pri spätnom šírení postupne zmenšuje (vanishing gradient) a neskorú vrstvám sa váhy prestanú aktualizovať. Reziduálne spojenie vytvára „skratku" cez ktorú gradient prúdi priamo — sieť sa môže trénovať aj keď je veľmi hlboká.

ResNet18 má 18 vrstiev, 11.2M parametrov. Napriek tomu že je najväčší, vyhral — jeho architektonické vzory lepšie zachytávajú vizuálne prvky oblohy.

### EfficientNet-B0

EfficientNet (2019) navrhol systematický spôsob ako škálovať sieť — zväčšovať šírku (počet filtrov), hĺbku (počet vrstiev) aj rozlíšenie vstupu proporcionálne naraz. B0 je základná, najmenšia verzia.

Používa **depthwise separable convolutions** — konvolúcia rozdelená do dvoch krokov:
1. Aplikuj filter na každý kanál zvlášť (priestorová informácia)
2. Skombinuj kanály dohromady (kanálová informácia)

Výsledok: rovnaký efekt ako klasická konvolúcia, ale s výrazne menej operáciami — sieť je menšia a rýchlejšia.

Napriek menšiemu počtu parametrov (4M) bol EfficientNet paradoxne **pomalší na trénovaní** ako ResNet18 (11.2M). Dôvod: jeho operácie sú sekvenčné a menej paralelizovateľné — GPU ich nemôže robiť súčasne tak dobre ako ResNet.

### MobileNetV3-Small

Navrhnutý pre mobilné zariadenia — minimálny počet parametrov (1.5M) pri prijateľnom výkone.

Kľúčová inovácia: **squeeze-and-excitation bloky**. Po každej konvolúcii sieť:
1. Spriemeruje každý kanál na jedno číslo (squeeze = „stlačenie")
2. Naučí sa váhy pre každý kanál — niektoré kanály sú dôležitejšie ako iné
3. Vynásobí kanály týmito váhami (excitation = „vzrušenie")

Efekt: sieť sa naučí, **ktoré vizuálne vzory sú dôležité** pre daný vstup a zintenzívni ich.

Bol najrýchlejší (~29s/experiment) ale pri plnom datasete nedosiahol kvalitu ResNetu.

---

## 11. Veľkosť vstupu — prečo 128px vyhral

Sieť vyžaduje fixnú veľkosť vstupu. My testujeme 128×128, 224×224, 320×320.

Väčší vstup v princípe nesie viac informácie. Ale:

**Väčší vstup = väčší model.** Prvá konvolučná vrstva spracováva pixely — pri 320px je vstup 6.25× väčší ako pri 128px. To znamená viac výpočtov, a pri niektorých architektúrach aj viac parametrov v prvej vrstve.

**Väčší model potrebuje viac dát.** Pri 946 obrázkoch komplikovanejší model ľahšie preučí — namiesto naučenia všeobecných vzorov si zapamätá konkrétne trénovacie príklady.

**Fyzikálna úvaha:** Pre odhad žiarenia nepotrebujeme vidieť ostré hrany oblakov — stačí vedieť kde je slnko a koľko oblohy je pokrytých. 128px zachytí toto globálne rozloženie dostatočne. Väčší vstup pridáva zbytočné detaily.

---

## 12. Batch size — prečo 32

**Batch size** = počet obrázkov spracovaných naraz pred jednou aktualizáciou váh.

Prečo nie spracovávame jeden po druhom? Gradient vypočítaný z jedného obrázka je veľmi šumivý — jeden konkrétny obrázok môže byť atypický a gradient by nás viedol zlým smerom. Keď spriemerujeme gradienty cez 32 obrázkov, dostaneme stabilnejší smer.

Prečo nie celý dataset naraz? Musel by sa zmestiť do GPU pamäte. 946 obrázkov pri 224px = ~150MB tenzor, to je ešte OK, ale pri väčších datasetoch (státisíce obrázkov) to nie je možné.

**Malý batch (napr. 4):** šumivý gradient, GPU nevyťažené, pomalé.
**Veľký batch (napr. 512):** presný gradient, GPU plne vyťažené, rýchle — ale model môže konvergovať do horších miním.
**32:** štandardný kompromis.

---

## 13. Čo je learning rate a prečo záleží

Learning rate $\alpha$ určuje veľkosť kroku pri aktualizácii váhy.

Predstav si slepca ktorý hľadá najnižší bod v kopcovitej krajine. Každý krok urobí v smere kde terén klesá (gradient). Learning rate = dĺžka kroku.

```
príliš malý LR:  □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ ← minimum  (trvá večne)
správny LR:      □   □   □   □   □ ← minimum              (plynulo klesá)
príliš veľký LR: □           □           □           □     (preskakuje)
```

Pri príliš veľkom LR slepec skočí cez minimum na druhú stranu kopca, potom späť, a nikdy sa nezastaví.

My testujeme tri hodnoty (0.001, 0.0001, 0.01) a vyberáme najlepšiu. `0.001` je štandardné pre Adam a potvrdilo sa ako optimálne aj v našom experimente.

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

---

### `LEARNING_RATES = [0.001, 0.0001, 0.01]`

**Ak použijem príliš veľké LR (napr. 0.1):**
- Optimizer preskakuje cez minimu → val MAE osciluje alebo rastie
- Model nekonverguje vôbec

**Ak použijem príliš malé LR (napr. 0.000001):**
- Váhy sa menia len minimálne → za 20 epoch skoro žiadne zlepšenie

**Prečo testujeme tri:** Optimálne LR závisí od architektúry a datasetu. Bez experimentu to nevieme vopred.

---

### `EPOCHS = 20`

**Ak zvýšim na 100:**
- Early stopping zastaví tréning aj tak najneskôr `PATIENCE` epoch po najlepšej
- Zvýšenie nad ~30 epoch nemá pri tomto datasete takmer žiadny efekt

**Ak znížim na 5:**
- Model nemá čas konvergovať, výsledky výrazne horšie

---

### `PATIENCE = 7`

**Ak zvýšim na 15:**
- Tolerujeme dlhšiu stagnáciu, môže zachytiť neskorú konvergenciu
- Tréning trvá dlhšie, riziko overfittingu v čase čakania

**Ak znížim na 2–3:**
- Zastavíme príliš skoro, vhodné len pre rýchle orientačné experimenty

---

### `INPUT_SIZES = [128, 224, 320]`

**Ak pridám menší (64px):**
- Možná strata informácie o pozícii slnka, rýchlejší tréning

**Ak pridám väčší (512px):**
- Výrazne pomalší tréning, pri 946 obrázkoch takmer istý overfitting

**Prečo 128px vyhral:** Stačí na zachytenie globálneho rozloženia oblohy, väčší vstup pridáva komplexitu bez pridanej hodnoty.

---

### `TRAIN_FRACS = [0.25, 0.5, 0.75, 1.0]`

| Frakcia | Vzorky | Výsledok |
|---------|--------|----------|
| 25 % | ~175 | R² záporné — horší ako predpovedanie priemeru |
| 50 % | ~330 | Výrazné zlepšenie, model začína zachytávať vzory |
| 75 % | ~497 | Ďalšie zlepšenie, krivka sa vyrovnáva |
| 100 % | ~662 | Najlepší výsledok |

Minimum pre zmysluplné výsledky pri transfer learningu: ~300–400 vzoriek.

---

### Výber modelu

**ResNet50 (25M parametrov):** pri 946 obrázkoch silný overfitting, výsledky pravdepodobne horšie.

**ViT (Vision Transformer):** transformery potrebujú stovky tisíc obrázkov. Pri 946 by takmer určite zlyhali. CNN majú vstavanú vlastnosť „lokálne vzory sú dôležité" čo pomáha pri malom datasete; transformery to nemajú.

---

### Augmentácia — alternatívy

**Horizontálny flip:** obloha je symetrická ľavo-pravý flip (nenarúša horizont) → mohlo by pomôcť.

**Zmena jasu/kontrastu:** simuluje rôzne kamery na rôznych staniciach → mohlo by zlepšiť generalizáciu.

**Náhodné orezy:** riziko orezania slnka z obrázka → menej vhodné.

---

### Čo by sa stalo s väčším datasetom

| Dataset | Očakávané R² | Poznámka |
|---------|-------------|----------|
| 946 (naše) | ~0.54 | Transfer learning, malý dataset |
| ~5 000 | ~0.70–0.75 | Môžeme použiť väčšie modely |
| ~50 000 | ~0.80–0.85 | Tréning od nuly je možný |
| Kompletné Eye2Sky | >0.90 | State-of-the-art, temporálne modelovanie |
