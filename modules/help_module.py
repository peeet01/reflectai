import streamlit as st

def run():
    st.title("📘 Tudományos Súgó – Neurolab AI")

    st.markdown("""
Üdvözlünk a **Neurolab AI Scientific Playground** felületén! Ez az alkalmazás különféle **idegtudományi, fizikai és matematikai modellek** interaktív vizsgálatát támogatja.  
Az alábbi modulok **matematikai érvényességgel**, **vizualizációval** és **prediktív szimulációkkal** segítik a tudományos megértést.

---

### 🕸️ Kuramoto Modell – Szinkronizációs Dinamika

**Matematikai képlet:**

$$
\\frac{d\\theta_i}{dt} = \\omega_i + \\frac{K}{N} \\sum_{j=1}^{N} A_{ij} \\sin(\\theta_j - \\theta_i)
$$

**Order parameter (globális szinkronizáció):**

$$
R(t) = \\left| \\frac{1}{N} \\sum_{j=1}^{N} e^{i\\theta_j(t)} \\right|
$$

**Cél:** Dinamikus hálózati szinkronizáció vizsgálata.  
**Vizualizáció:** Fázisállapotok időbeli alakulása és szinkronizáció mértéke.  
**Következtetés:** Detektálható szinkron idegi aktivitás, epilepsziás minták vagy ritmikus zavarok.

---

### ❌ XOR Predikció – Neurális hálózat tanulása

**Modellképlet:**

$$
\\hat{y} = \\sigma(W^{(2)} \\cdot \\sigma(W^{(1)}x + b^{(1)}) + b^{(2)})
$$

**Tanulási veszteség:**

$$
\\mathcal{L} = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2
$$

**Cél:** Nemlineáris minták felismerése rejtett rétegekkel.  
**Vizualizáció:** Pontfelhő és döntési határ vizuális összevetése.  
**Következtetés:** Megérthető a mélytanulás jelentősége.

---

### 🌐 Berry-görbület – Kvantum topológia

**Berry-görbület definíció:**

$$
\\Omega(k) = \\nabla_k \\times A(k), \\quad A(k) = i \\langle u_k | \\nabla_k u_k \\rangle
$$

**Berry-fázis:**

$$
\\gamma = \\oint_C A(k) \\cdot dk = \\int_S \\Omega(k) \\, d^2k
$$

**Cél:** Kvantumállapotok topológiai invariánsainak számítása.  
**Vizualizáció:** Kvantált értékek (pl. Chern-szám) színtérképes megjelenítése.  
**Következtetés:** Megjelenik kvantum Hall-effektusban, topológiai szigetelőkben.

---

### 🧠 Hopfield-háló – Asszociatív memória

**Súlymátrix Hebb szabály szerint:**

$$
W_{ij} = \\sum_{\\mu=1}^{P} \\xi_i^\\mu \\xi_j^\\mu
$$

**Dinamikai frissítés:**

$$
s_i^{(t+1)} = \\text{sign}\\left(\\sum_j W_{ij} s_j^{(t)} \\right)
$$

**Cél:** Zajos minták alapján tanult bináris memóriák visszanyerése.  
**Vizualizáció:** Minták energiafelszínre történő konvergenciája.  
**Következtetés:** Energiaalapú számítás és stabil mintatárolás.

---

### 🌀 Fraktál Explorer – Kaotikus rendszerek

**Mandelbrot-halmaz képlete:**

$$
z_{n+1} = z_n^2 + c
$$

**Cél:** Kaotikus és stabil viselkedés határainak vizsgálata.  
**Vizualizáció:** Színes komplex sík fraktálok, Julia-halmazok.  
**Következtetés:** Nemlineáris rendszerek bifurkációi, érzékenysége.

---

### 🔄 Echo State Network (ESN) – Idősoros előrejelzés

**Belső dinamika:**

$$
x(t+1) = \\tanh(W_{res} \\cdot x(t) + W_{in} \\cdot u(t))
$$

**Kimenet:**

$$
\\hat{y}(t) = W_{out} \\cdot x(t)
$$

**Cél:** Idősorok előrejelzése kis tanítási költséggel.  
**Vizualizáció:** Predikciók összevetése a valós mintákkal.  
**Következtetés:** Jó teljesítmény bonyolult mintákon is.

---

### 🧩 Generative Kuramoto – Struktúra és dinamika

**Funkció:** Véletlenszerű gráf generálás (pl. Erdős–Rényi, Barabási) és Kuramoto-dinamika futtatása.

**Cél:** Struktúra-dinamika összefüggés megértése.  
**Vizualizáció:** Hálózati gráf + szinkronizációs minta.  
**Következtetés:** Topológiai robusztusság és kritikus átmenetek.

---

### 🧮 Graph Sync Analysis – Hálózati stabilitás

**Spektrális tulajdonságok:**

- Laplace-mátrix sajátértékek
- Szinkron stabilitásvizsgálat

**Cél:** Szinkron dinamikák stabilitásának becslése gráf alapján.  
**Következtetés:** Meghatározható stabilitási határ a spektrális rés alapján.

---

### 🏔️ Persistent Homology – Topológiai adatértelmezés

**Lépések:**

- Vietoris–Rips komplexum
- Perzisztencia diagram generálás

**Cél:** Adatok globális szerkezetének feltérképezése.  
**Vizualizáció:** Diagram időbeli topológiai változásokkal.  
**Következtetés:** Gépi tanulásba integrálható robusztus jellemzők.

---

Verzió: **2025.07**  
Készítette: *ReflectAI fejlesztői és tudományos tanácsadók*
    """)

# Modul belépési pont
app = run
