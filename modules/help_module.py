import streamlit as st

def run(): st.title("❓ Súgó – Neurolab AI Scientific Playground Sandbox")

st.markdown("""
## 🔍 Mi ez az alkalmazás?

A **Neurolab AI** egy kutatásorientált sandbox platform, amely lehetővé teszi neurális hálózatok, szinkronizációs rendszerek,
topológiai modellek és tanulási folyamatok gyors tesztelését és vizualizációját. Az alkalmazás célja, hogy intuitív felületet
biztosítson **kutatóknak, hallgatóknak és fejlesztőknek**, akik Python-alapú MI modellekből és elméletekből szeretnének többet látni
– vizuálisan és mérhetően.

---

## 🧭 Modulismertető

### 🔁 XOR predikció neurális hálóval
- **Cél:** Egy egyszerű MLP tanítása az XOR probléma megoldására.
- **Funkciók:** zaj hozzáadása, predikció, tanítási pontosság és idő mérése, CSV export, modellmentés, egyéni input vizsgálat,
  interaktív 3D felület, konfúziós mátrix.

### 🧭 Kuramoto szinkronizáció
- **Cél:** Oszcillátorok szinkronizációjának vizsgálata időben.
- **Funkciók:** kezdeti/végső fáziseloszlás, szinkronizációs index (r), szórásgörbe, dendrit-szerű 3D vizualizáció.

### 🧠 Hebbian tanulás
- **Cél:** Hebbian szabály szimulációja szinaptikus erősségek alapján.
- **Funkciók:** tanulási ráta és neuronháló méretének paraméterezése, mátrix alapú súlyvizualizáció.

### ⚡ Kuramoto–Hebbian hálózat
- **Cél:** Kuramoto dinamikák és Hebbian tanulás kombinálása.

### 🔒 Topológiai szinkronizáció
- **Cél:** Hálózat topológiájának hatása a szinkronizációra.

### 🌀 Lorenz rendszer (szimuláció)
- **Cél:** Kaotikus Lorenz-rendszer numerikus szimulációja.

### 🔮 Lorenz predikció
- **Cél:** Neurális hálóval történő Lorenz-rendszer előrejelzése.

### 🧬 Zajtűrés és szinkronizációs robusztusság
- **Cél:** A rendszer érzékenysége a bemeneti zajra.

### 🧩 Topológiai Chern–szám analízis
- **Cél:** Chern-számok numerikus becslése Berry-görbület alapján.

### 🧠 Belátás alapú tanulás (Insight Learning)
- **Cél:** A megértés-alapú hirtelen tanulás szimulációja.

### 📈 Echo State Network (ESN) predikció
- **Cél:** ESN-re alapozott dinamikus predikciók időfüggő adatokon.

### 🔄 Hebbian plaszticitás dinamikája
- **Cél:** A tanulási súlyok időbeli alakulása Hebbian elvek mentén.

### 🧮 Szinkronfraktál dimenzióanalízis
- **Cél:** Fraktáldimenzió becslése szinkronizációs hálókon.

### 🧠 Generatív Kuramoto hálózat
- **Cél:** Automatikusan generált oszcillátorhálók viselkedésének vizsgálata.

### 🧭 Memória tájkép (Memory Landscape)
- **Cél:** Hálózatok memóriaállapotainak feltérképezése.

---

## 📦 Export és mentés
- CSV export a predikciós eredményekhez
- Modellek mentése újrabetöltéshez
- Jegyzetek mentése vizsgálatokhoz

---

## 👥 Célközönség
- **Kutatók:** elméletek gyors verifikálása
- **Oktatók:** szemléltető példák, interaktív demonstrációk
- **Diákok:** tanulási kísérletek, saját modellek megértése
- **Fejlesztők:** bővíthető moduláris rendszer

""")

