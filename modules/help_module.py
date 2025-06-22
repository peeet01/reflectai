import streamlit as st

def run(): st.title("❓ Súgó – Neurolab AI Scientific Playground Sandbox")

st.markdown("""
## 🔍 Mi ez az alkalmazás?
A **Neurolab AI** egy nyílt kutatásorientált interaktív sandbox, amely lehetővé teszi különböző mesterséges intelligencia modellek, dinamikai rendszerek és hálózati szimulációk futtatását és megértését. A cél, hogy **kutatók, hallgatók, oktatók és fejlesztők** számára egy szemléletes, moduláris és bővíthető felület álljon rendelkezésre a gépi tanulás, idegrendszeri dinamika és szinkronizáció területein.

---

## 🧝‍♂️ Modulismertető (Tudományos + Matematikai leírásokkal)

### ⭮️ XOR predikció neurális hálóval
- **Cél:** Egy bináris logikai függvény (XOR) megtanítása MLP-vel.
- **Képlet:** y = \sigma(W_2 \cdot \tanh(W_1 x + b_1) + b_2)

### 🧽 Kuramoto szinkronizáció
- **Cél:** Oszcillátorok kollektív szinkronizációja.
- **Egyenlet:** \frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^{N} \sin(\theta_j - \theta_i)

### 🧠 Hebbian tanulás
- **Cél:** Hebb-elv vizualizációja.
- **Képlet:** \Delta w_{ij} = \eta x_i x_j

### ⚡ Kuramoto–Hebbian hálózat
- **Kombinált modell:** Időbeli fázis + adaptív kapcsolat
- **Képletek:** Kuramoto + Hebbian kombináció

### 🔒 Topológiai szinkronizáció
- **Cél:** Gráfstruktúrák hatásának vizsgálata
- **Képlet:** Szomszédsági mátrix alapú Kuramoto

### 🌀 Lorenz rendszer
- **Cél:** Kaotikus dinamika
- **Képletek:**
    - \frac{dx}{dt} = \sigma(y - x)
    - \frac{dy}{dt} = x(\rho - z) - y
    - \frac{dz}{dt} = xy - \beta z

### 🔮 Lorenz predikció
- **Cél:** Idősor becslés neurális hálóval
- **Módszer:** Regresszió tanulással

### 🧬 Zajtűrés
- **Cél:** Zaj hatásának mérése
- **Képlet:** x_{noisy} = x + \epsilon \cdot N(0,1)

### 🧩 Chern-szám analízis
- **Cél:** Topológiai invariancia
- **Képlet:** C = \frac{1}{2\pi} \int_{BZ} F_{xy}(k) \, d^2k

### 🧠 Insight Learning
- **Cél:** Belátás alapú tanulás modellezése
- **Elv:** Nemlineáris, diszkrét hirtelen váltás

### 📈 ESN predikció
- **Cél:** Idősorok memóriaalapú predikciója
- **Képlet:** y(t) = W_{out} x(t)

### 🔄 Hebbian plaszticitás
- **Cél:** Szinaptikus erősség időbeli változása
- **Képlet:** w_{ij}(t+1) = w_{ij}(t) + \eta x_i x_j

### 📊 Szinkronfraktál dimenzió
- **Cél:** Dimenziómérés fraktálszerkezetekben
- **Módszer:** Box-counting, korrelációs dimenzió

### 🧠 Generatív Kuramoto
- **Cél:** Dinamikus gráfgenerálás + Kuramoto
- **Alap:** Preferenciális csatlakozás + szinkronmodul

### 🧽 Memória tájkép
- **Cél:** Energiafelület vizualizáció memóriában
- **Elv:** Stabilitásminimumok topológiája

---

## 📦 Export és mentés
- CSV export
- Modellmentés `.pth` kiterjesztéssel
- Jegyzetelés környezetben

---

## 👥 Célközönség
- **Kutatók**: gyors modellezés és validáció
- **Oktatók**: interaktív demonstrációk
- **Hallgatók**: előtanulmányok és kísérletek
- **Fejlesztők**: moduláris architektúra
""")

