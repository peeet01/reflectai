import streamlit as st

def run(): st.title("❓ Súgó – Neurolab AI Scientific Playground Sandbox")

st.markdown(r"""
## 🔍 Mi ez az alkalmazás?
A **Neurolab AI** egy nyílt kutatásorientált interaktív sandbox, amely lehetővé teszi különböző mesterséges intelligencia modellek, dinamikai rendszerek és hálózati szimulációk futtatását és megértését. A cél, hogy **kutatók, hallgatók, oktatók és fejlesztők** számára egy szemléletes, moduláris és bővíthető felület álljon rendelkezésre a gépi tanulás, idegrendszeri dinamika és szinkronizáció területein.

---

## 🧭 Modulismertető (Tudományos leírásokkal és képletekkel)

### 🔁 XOR predikció neurális hálóval
- **Cél:** Egy bináris logikai függvény (XOR) megtanítása egy több rétegű perceptron segítségével.
- **Tudományos háttér:** Az XOR nemlineáris problémát jelent, amit egyetlen rétegű háló nem tud megtanulni, de egy rejtett réteggel rendelkező MLP képes rá.
- **Képlet:**

y = \sigma(W_2 \cdot \tanh(W_1 x + b_1) + b_2)

### 🧭 Kuramoto szinkronizáció
- **Cél:** Az oszcillátorok kollektív szinkronizációs viselkedésének modellezése.
- **Tudományos háttér:** A Kuramoto-modell fázisoszcillátorok közötti szinkronizációt ír le.
- **Képlet:**

\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^N \sin(\theta_j - \theta_i)

### 🧠 Hebbian tanulás
- **Cél:** A Hebb-féle tanulási szabály szemléltetése.
- **Képlet:**

\Delta w_{ij} = \eta x_i x_j

### ⚡ Kuramoto–Hebbian hálózat
- **Cél:** Dinamikus szinkronizációs és adaptív súlytanulási folyamatok kombinációja.
- **Képlet:**

w_{ij}(t+1) = w_{ij}(t) + \eta \sin(\theta_i - \theta_j)

### 🔒 Topológiai szinkronizáció
- **Cél:** A hálózati struktúra hatása a szinkronizáció stabilitására.
- **Tudományos háttér:** A gráf Laplace-mátrixának spektruma befolyásolja a szinkronizáció feltételeit.

### 🌀 Lorenz rendszer (szimuláció)
- **Cél:** A determinisztikus káosz bemutatása.
- **Képletek:**

\begin{aligned}
      \dot{x} &= \sigma (y - x) \\
      \dot{y} &= x (\rho - z) - y \\
      \dot{z} &= x y - \beta z
      \end{aligned}

### 🔮 Lorenz predikció
- **Cél:** Kaotikus rendszer előrejelzése neurális hálóval.

### 🧬 Zajtűrés és szinkronizációs robusztusság
- **Cél:** A szinkronizáció érzékenységének mérése külső zajra.

### 🧩 Topológiai Chern–szám analízis
- **Cél:** Chern-szám meghatározása Berry-görbület alapján.
- **Képlet:**

C = \frac{1}{2\pi} \int_{BZ} \Omega(k) \ d^2k

### 🧠 Belátás alapú tanulás (Insight Learning)
- **Cél:** A hirtelen tanulási áttörés modellezése nem próbálgatás alapján.

### 📈 Echo State Network (ESN) predikció
- **Cél:** Dinamikus rendszerek predikciója visszacsatolt hálóval.
- **Képlet:**

x(t+1) = \tanh(W_{in} u(t) + W x(t))

### 🔄 Hebbian plaszticitás dinamikája
- **Cél:** Súlyváltozások vizsgálata időben.

### 🧮 Szinkronfraktál dimenzióanalízis
- **Cél:** A szinkronizáció mintázatainak fraktáldimenziójának vizsgálata.

### 🧠 Generatív Kuramoto hálózat
- **Cél:** Véletlen Kuramoto-hálók viselkedésének vizsgálata.

### 🧭 Memória tájkép (Memory Landscape)
- **Cél:** Hálózati memóriaállapotok topográfiai feltérképezése.

---

## 📦 Export és mentés
- CSV export
- Modellmentés
- Jegyzetmentés

---

## 👥 Célközönség
- Kutatók, oktatók, diákok, fejlesztők
""")

