import streamlit as st

def run(): st.title("❓ Tudományos Súgó – Neurolab AI") st.markdown(""" Üdvözlünk a Neurolab AI Scientific Playground felületen! Ez az alkalmazás különféle elméleti és gyakorlati idegrendszeri, fizikai és matematikai modellek interaktív vizsgálatát támogatja.

A következőkben bemutatjuk az egyes modulok matematikai alapjait, célját, alkalmazhatóságát és következtetéseit. """)

# Kuramoto modell
st.markdown("""


---

🕸️ Kuramoto Modell – Szinkronizációs Dinamika

Matematikai leírás:

\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^{N} A_{ij} \sin(\theta_j - \theta_i)

Kollektív viselkedés: Ez a modell az egyedi oszcillátorok (pl. neuronok, sejtciklusok) fázisszinkronizációját írja le. Az $R(t)$ order parameter:

R(t) = \left| \frac{1}{N} \sum_{j=1}^{N} e^{i\theta_j(t)} \right|

Célja az alkalmazásban: Modellezi a neuronális hálózatok dinamikus kollektív állapotait. A gráfstruktúra és kapcsolódási erősség változtatásával megfigyelhetők a szinkronizáció feltételei.

Következtetés és felhasználás:

Hálózati instabilitások detektálása

Szinkron idegi aktivitás jellemzése

Ritmikus zavarok szimulációja (pl. epilepszia) """)

XOR neurális hálózat

st.markdown("""



---

❌ XOR Predikció – Neurális hálózat tanulása

Matematikai modell:

\hat{y} = \sigma\left( W^{(2)} \cdot \sigma(W^{(1)} x + b^{(1)}) + b^{(2)} \right)

Cél: A nemlineárisan szeparálható problémák (mint az XOR) csak rejtett rétegekkel taníthatók. A klasszikus példa motiválta a mélytanulás fejlődését.

Tanulási kritérium:

\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2

Következtetés:

Szükség van nemlineáris aktivációkra és rejtett rétegekre

Megérthető, mikor buknak el lineáris modellek

Alapvető a modern klasszifikációs problémákhoz """)

Berry-görbület

st.markdown("""



---

🌐 Berry-görbület – Kvantum topológia

Meghatározás:

\Omega(\mathbf{k}) = \nabla_{\mathbf{k}} \times \mathbf{A}(\mathbf{k})
\quad \text{ahol} \quad
\mathbf{A}(\mathbf{k}) = -i \langle u(\mathbf{k}) | \nabla_{\mathbf{k}} | u(\mathbf{k}) \rangle

Fizikai értelmezés: A Berry-görbület topológiai fázisokat különböztet meg kvantumállapotok között. A görbület integrálja: Chern-szám, kvantált mennyiség.

Célja az appban: Lehetővé teszi topológiai invariánsok számítását diszkrét pontokon.

Következtetés:

Az adat térbeli szerkezetét írja le

Topológiai különbségek kimutatására alkalmas

Megjelenik kvantum Hall-effektusban, szigetelőkben """)

Hopfield-háló

st.markdown("""



---

🧠 Hopfield-háló – Asszociatív memória

Súlymátrix:

W_{ij} = \sum_{\mu=1}^{P} \xi_i^\mu \xi_j^\mu \quad \text{(Hebbian tanulás)}

Dinamikai szabály:

s_i^{t+1} = \mathrm{sign} \left( \sum_j W_{ij} s_j^t \right)

Cél: Betanított bináris minták visszakeresése – még zajos bemenetekből is.

Következtetés:

Energiafüggvény minimumaiba konvergál

Működik, mint neurális memória

Demonstrálja az energia-alapú számítást """)

Fraktál explorer

st.markdown("""



---

🌀 Fraktál Explorer – Kaotikus rendszerek

Iterációs képlet:

z_{n+1} = z_n^2 + c

Cél: Megfigyelni a Mandelbrot-halmaz és Julia-halmaz határeseteit és bifurkációit.

Következtetés:

Nemlineáris rendszerek érzékenysége

Numerikus vizsgálatok stabil és kaotikus zónákban

Vizualizációja a komplex dinamikának """)

Echo State Network

st.markdown("""



---

🔄 ESN – Idősoros előrejelzés

Architektúra:

\mathbf{x}(t+1) = \tanh(W_{res} \cdot \mathbf{x}(t) + W_{in} \cdot \mathbf{u}(t))

\hat{y}(t) = W_{out} \cdot \mathbf{x}(t)

**Cél:**
Dinamikus rendszerek előrejelzése kis tanítási költséggel. A belső súlyokat nem kell tanítani.

**Következtetés:**
- Hatékony idősoros predikció
- Jó teljesítmény komplex mintákon
- Dinamikus mintafelismerés
""")

    # Generative Kuramoto
    st.markdown("""
---
### 🧩 Generative Kuramoto – Struktúra és dinamika

**Funkció:**
Random gráf generálása (pl. Erdos-Renyi, Barabasi), majd Kuramoto szimuláció ráillesztése.

**Cél:**
Feltérképezni, hogy a gráf topológiája hogyan befolyásolja a szinkronizációs viselkedést.

**Következtetés:**
- Lokális kapcsolatokból globális dinamika
- Dinamikai robusztusság vizsgálata
""")

    # Graph Sync Analysis
    st.markdown("""
---
### 🧮 Graph Sync Analysis – Hálózati stabilitás

**Vizsgálja:**
- Kapcsolati mátrix
- Szinkronizáció erőssége
- Spektrális jellemzők (pl. Laplace sajátértékek)

**Következtetés:**
- Hálózat szerkezete és szinkron dinamikája között kapcsolat
- Stabilitás meghatározható spektrális rés alapján
""")

    # Persistent Homology
    st.markdown("""
---
### 🏔️ Persistent Homology – Topológiai adatértelmezés

**Algoritmus:**
- Vietoris-Rips komplexum
- Homológia osztályok számítása
- Perzisztencia diagram előállítása

**Cél:**
Adatok globális topológiai szerkezetének feltárása, robust minták felismerése.

**Következtetés:**
- Zaj és valódi struktúra szétválasztása
- Topológiai invariánsok kimutatása
- Gépi tanulásba integrálható jellemzők
""")

    st.markdown("""
---
Verzió: **2025.07**  
Készítette: *ReflectAI fejlesztői és tudományos tanácsadók*
""")

# Modul belépési pont
app = run

