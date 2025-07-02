import streamlit as st

def run(): st.set_page_config(page_title="Tudományos Súgó – Neurolab AI", layout="centered") st.title("❓ Tudományos Súgó – Neurolab AI")

st.markdown("""
Üdvözlünk a **Neurolab AI Scientific Playground** felületen!  
Ez az alkalmazás különféle elméleti és gyakorlati idegrendszeri, fizikai és matematikai modellek interaktív vizsgálatát támogatja.

---

## 🕸️ Kuramoto Modell – Szinkronizációs Dinamika
A Kuramoto-modell az oszcillátorok fázisszinkronizációját írja le:

$$
\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^{N} A_{ij} \sin(\theta_j - \theta_i)
$$

Ahol:
- $\theta_i$: az *i*-edik oszcillátor fázisa
- $\omega_i$: természetes frekvencia
- $K$: csatolási erősség
- $A_{ij}$: kapcsolódási mátrix

A globális szinkronizáció mértéke:
$$
R(t) = \left| \frac{1}{N} \sum_{j=1}^{N} e^{i\theta_j(t)} \right|
$$

Alkalmazás: agyhullámok, szívritmus, biológiai oszcilláció.

---

## ❌ XOR Predikció – Neurális hálózat tanítása
A klasszikus XOR probléma bemutatja, hogy lineáris modellek nem alkalmasak nemlineáris döntési határok kezelésére.

$$
\hat{y} = \sigma\left( W^{(2)} \cdot \sigma(W^{(1)} \cdot x + b^{(1)}) + b^{(2)} \right)
$$

- Aktiváció: $\sigma(x) = \frac{1}{1 + e^{-x}}$
- Veszteségfüggvény:
$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
$$

Modul célja: tanítási algoritmusok kipróbálása, nemlinearitás vizsgálata.

---

## 🌐 Berry-görbület – Kvantum topológia
$$
\Omega(\mathbf{k}) = \nabla_{\mathbf{k}} \times \mathbf{A}(\mathbf{k})
$$

Ahol:
$$
\mathbf{A}(\mathbf{k}) = -i \langle u(\mathbf{k}) | \nabla_{\mathbf{k}} | u(\mathbf{k}) \rangle
$$

Jelentősége: topológiai fázisok, kvantum Hall-effektus, Chern-szám vizsgálata.

---

## 🧠 Hopfield-háló – Asszociatív memória
$$
W_{ij} = \sum_{\mu=1}^P \xi_i^\mu \xi_j^\mu, \quad W_{ii} = 0
$$
$$
s_i^{t+1} = \text{sign} \left( \sum_j W_{ij} s_j^t \right)
$$

Funkciója: betanult minták visszanyerése, energiafüggvény alapú konvergencia.

---

## 🌀 Fraktál Explorer – Kaotikus dinamika
$$
z_{n+1} = z_n^2 + c
$$

Mandelbrot-halmaz iteratív számítása.  
Vizualizáció célja: komplex síkbeli viselkedés, káosz és határmintázatok.

---

## 🧠 ESN Prediction – Echo State Network
Lineáris olvasóval rendelkező rekurrens háló:
$$
x(t+1) = \tanh(Wx(t) + Win u(t))
$$
$$
y(t) = W_{out} x(t)
$$

Cél: nemlineáris rendszerek időbeli előrejelzése.  
Az **"állapot-rezervoir"** koncepció robusztus és gyors tanítást tesz lehetővé.

---

## 🔁 Generative Kuramoto – Gráfgeneráció szinkronizációhoz
Modul gráfokat generál oszcillátor dinamikákhoz.  
Alkalmazható szinkronizációs feltételek és gráfstruktúra kapcsolatának kutatására.

---

## 📊 Graph Sync Analysis – Hálózati szinkron analízis
Vizsgálja a gráf topológiák hatását a Kuramoto-modell szinkronizációjára.  
Felhasználó által definiált gráfokon futtatható.  
Következtetési lehetőség: gráf metrikák és szinkronizáció mértékének kapcsolata.

---

## 🧬 Hebbian Learning – Szinaptikus plaszticitás
$$
\Delta w_{ij} = \eta \cdot x_i x_j
$$

A Hebb-szabály alapján: "Neurons that fire together wire together".  
Modellezhető vele adaptív kapcsolatok és tanulás gráfstruktúrák mentén.

---

## 🧩 Persistent Homology – Topológiai adatelemzés
Homológia: topológiai jellemzők (lyukak, komponensek) számítása különböző skálákon.  
Felhasználása: gráfstruktúrák, időfüggő hálózatok stabil mintázatainak azonosítása.

---

## 🪞 Reflection Modul
Lehetőség saját hipotézisek, megfigyelések, levezetett konklúziók dokumentálására.

---

## 🧪 Következtetések és bizonyítási ötletek
- A Kuramoto-modell gráfstruktúra függő szinkronizációs képessége vizsgálható
- Az XOR megoldása mutatja a mélytanulás szükségességét nemlineáris rendszerekhez
- A Hopfield-háló energiaalapú tanulása stabil mintatárolót biztosít
- Az ESN segítségével időfüggő viselkedések előrejelzése lehetséges komplex rendszerekben

---

Verzió: **2025.07**  
Készítette: *ReflectAI közösség – sandbox kutatói célokra*
""")

ReflectAI modul belépési pontja

app = run

