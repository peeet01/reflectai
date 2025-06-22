import streamlit as st

def run():
    st.title("📘 Súgó – Neurolab AI Scientific Playground Sandbox")

    st.markdown("""
## 🔍 Mi ez az alkalmazás?

A **Neurolab AI** egy nyílt tudományos sandbox környezet, amely a mesterséges intelligencia és nemlineáris dinamikai rendszerek vizsgálatára szolgál. A cél egy **vizuális, interaktív és bővíthető** felület biztosítása kutatási és oktatási célokra.

---

## 🧭 Modulismertető (tudományos háttérrel)

### 🔁 XOR predikció neurális hálóval

**Cél:** A kizáró vagy (XOR) logikai kapu tanítása mesterséges neurális hálóval.

**Tudományos háttér:**

Az XOR probléma nemlineárisan szeparálható, ezért szükséges egy rejtett réteg az alábbi többrétegű perceptronban (MLP):

$$
y = \\sigma(W_2 \\cdot \\tanh(W_1 x + b_1) + b_2)
$$

ahol:

- $x ∈ ℝ^2$ a bemenet,
- $\\tanh$ az aktivációs függvény,
- $\\sigma$ a kimeneti sigmoid függvény.

**Funkciók:** zajgenerálás, tanítás, predikció, CSV export, tanulási idő, 3D felület, konfúziós mátrix.

---

### 🧭 Kuramoto szinkronizáció

**Cél:** Szinkronizációs viselkedés vizsgálata egy oszcillátorhálóban.

**Matematikai modell (Kuramoto-egyenlet):**

$$
\\frac{d\\theta_i}{dt} = \\omega_i + \\frac{K}{N} \\sum_{j=1}^N \\sin(\\theta_j - \\theta_i)
$$

ahol:

- $\\theta_i$ az $i$-edik oszcillátor fázisa,
- $\\omega_i$ a sajátfrekvencia,
- $K$ a kapcsolódási erősség.

**Szinkronizációs mérték (order parameter):**

$$
r(t) = \\left| \\frac{1}{N} \\sum_{j=1}^N e^{i\\theta_j(t)} \\right|
$$

**Funkciók:** fáziseloszlás, szórás, szinkronindex, dendritikus 3D vizualizáció.

---

### 🧠 Hebbian tanulás

**Cél:** A Hebbian tanulás modellezése, amely szerint „az együtt tüzelő neuronok erősítik egymást”.

**Tanulási szabály:**

$$
\\Delta w_{ij} = \\eta x_i x_j
$$

ahol:

- $\\eta$ a tanulási ráta,
- $x_i$, $x_j$ a bemenetek aktivitása.

**Funkciók:** súlymátrix vizualizáció, paraméterezhető tanulás.

---

### ⚡ Kuramoto–Hebbian hálózat

**Cél:** A szinkronizáció és plaszticitás kombinálása dinamikus tanulási hálózatban.

**Kombinált szabály (időfüggő):**

$$
\\frac{d\\theta_i}{dt} = \\omega_i + \\sum_j w_{ij}(t) \\sin(\\theta_j - \\theta_i)
$$

$$
\\frac{dw_{ij}}{dt} = \\eta \\cos(\\theta_j - \\theta_i)
$$

Ez a rendszer képes **tanulni** a szinkronizációból.

---

### 🔒 Topológiai szinkronizáció

**Cél:** A gráf topológiájának hatása a szinkronizációra.

A szinkronizáció mértéke függ a **Laplacián mátrix** spektrumától:

$$
L = D - A
$$

ahol $D$ a fokszám mátrix, $A$ az adjancencia mátrix.

---

### 🌀 Lorenz rendszer

**Cél:** A híres kaotikus Lorenz-rendszer szimulációja.

$$
\\begin{aligned}
\\frac{dx}{dt} &= \\sigma(y - x) \\\\
\\frac{dy}{dt} &= x(\\rho - z) - y \\\\
\\frac{dz}{dt} &= xy - \\beta z
\\end{aligned}
$$

---

### 🔮 Lorenz predikció

**Cél:** Idősoros adatok előrejelzése neurális hálóval, pl. RNN vagy ESN.

---

### 🧬 Zajtűrés és robusztusság

**Cél:** Az MI-modulok érzékenységének vizsgálata bemeneti zajra és szinkronizációs instabilitásra.

---

### 🧩 Topológiai Chern–szám analízis

**Cél:** A topológiai fázisok numerikus becslése.

**Berry-görbületből számolt Chern-szám:**

$$
C = \\frac{1}{2\\pi} \\int_{BZ} \\mathcal{F}(k) \\, dk
$$

---

### 🧠 Belátás alapú tanulás

**Cél:** Az "aha" élmény modellezése tanulás közben. Nem fokozatos tanulás, hanem hirtelen áttörés (gestalt switching).

---

### 📈 Echo State Network (ESN)

**Cél:** Nemlineáris dinamikus rendszerek előrejelzése "reservoir computing" módszerrel.

**Képlet:**

$$
x(t+1) = \\tanh(W_{res} x(t) + W_{in} u(t))
$$

---

### 🔄 Hebbian plaszticitás dinamikája

**Cél:** A Hebbian súlyok időbeli evolúciója. Vizsgálható stabilitás, konvergencia.

---

### 🧮 Szinkronfraktál dimenzióanalízis

**Cél:** Fraktáldimenzió becslése a Kuramoto háló fázisain.

---

### 🧠 Generatív Kuramoto hálózat

**Cél:** Új szinkronizációs gráfok automatikus generálása, szimulációja.

---

### 🧭 Memória tájkép

**Cél:** Neurális memóriaállapotok vizsgálata és stabilitásuk ábrázolása energiagörbületként.

---

## 👩‍🔬 Célközönség

- **Kutatók:** új modellek, elméletek gyors prototípusai
- **Oktatók:** oktatási szemléltetés
- **Diákok:** gyakorlati MI- és fizikai rendszerek tanulmányozása
- **Fejlesztők:** moduláris bővítés, kutatás-alapú kísérletek

---
""")
