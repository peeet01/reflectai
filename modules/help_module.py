import streamlit as st

def run(): st.title("❓ Súgó és Dokumentáció – Neurolab AI") st.markdown(""" Üdvözlünk a Neurolab AI Scientific Playground alkalmazásban!
Ez a sandbox környezet lehetőséget ad különféle idegtudományi, hálózati és tanulási modellek vizsgálatára.

---
## 🔢 Alapvető matematikai képletek
""")

# Kuramoto modell
st.markdown("### 🕸️ Kuramoto Modell – Szinkronizációs Dinamika")
st.latex(r"\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^{N} A_{ij} \sin(\theta_j - \theta_i)")
st.markdown("""
**Jelölések**:  
- $\theta_i$: az *i*-edik oszcillátor fázisa  
- $\omega_i$: természetes frekvencia  
- $K$: kapcsolódási erősség  
- $A_{ij}$: kapcsolódási mátrix  
- $N$: oszcillátorok száma
""")
st.latex(r"R(t) = \left| \frac{1}{N} \sum_{j=1}^{N} e^{i\theta_j(t)} \right|")
st.markdown("""
A Kuramoto-modell klasszikus példa a kollektív viselkedés vizsgálatára komplex rendszerekben.  
Alkalmazásai: agyhullámok, hálózati áramkörök, biológiai ritmusok, szociodinamikai rendszerek.
""")

# XOR modell
st.markdown("---")
st.markdown("### ❌ XOR Predikció – Neurális Hálózat")
st.markdown("""
| x₁ | x₂ | XOR |
|----|----|-----|
| 0  | 0  |  0  |
| 0  | 1  |  1  |
| 1  | 0  |  1  |
| 1  | 1  |  0  |
""")
st.latex(r"\hat{y} = \sigma\left( W^{(2)} \cdot \sigma(W^{(1)} \cdot x + b^{(1)}) + b^{(2)} \right)")
st.markdown("Ahol $\sigma(x) = \frac{1}{1 + e^{-x}}$ a szigmoid aktiváció.")
st.latex(r"\mathcal{L} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2")
st.markdown("""
Az XOR probléma klasszikus példája a nemlineárisan szeparálható problémáknak.  
Megoldásához több rétegű perceptron (MLP) szükséges – ez vezette el a mélytanulás kialakulásához.
""")

# Berry curvature
st.markdown("---")
st.markdown("### 🌐 Berry-görbület – Topológiai Kvantumfizika")
st.latex(r"\Omega(\mathbf{k}) = \nabla_{\mathbf{k}} \times \mathbf{A}(\mathbf{k})")
st.markdown("A Berry-kapcsolat:")
st.latex(r"\mathbf{A}(\mathbf{k}) = -i \langle u(\mathbf{k}) | \nabla_{\mathbf{k}} | u(\mathbf{k}) \rangle")
st.markdown("""
A Berry-görbület a kvantummechanika topológiai aspektusait tárja fel.  
Kiemelten fontos a topológiai szigetelők és kvantum Hall-effektus megértésében.  
A Chern-szám kvantált válaszokat jelez – topológiai invariáns.
""")

# Hopfield háló
st.markdown("---")
st.markdown("### 🧠 Hopfield-háló – Asszociatív Memória")
st.latex(r"W_{ij} = \sum_{\mu=1}^P \xi_i^\mu \xi_j^\mu,\quad W_{ii} = 0")
st.latex(r"s_i^{t+1} = \text{sign} \left( \sum_j W_{ij} s_j^t \right)")
st.markdown("""
A Hopfield-háló olyan neurális rendszer, amely betanított mintákra képes visszaemlékezni.  
Az energiafüggvénye alapján konvergál stabil állapotokba, akár zajos bemenetből is.

Ez demonstrálható a **Memória Tájkép Pro** modulban vizuálisan.
""")

# Fraktál modul
st.markdown("---")
st.markdown("### 🌀 Fraktál Explorer – Geometria és Kaotikus Rendszerek")
st.latex(r"z_{n+1} = z_n^2 + c")
st.markdown("""
A komplex síkon értelmezett iterációs folyamat Mandelbrot-halmazt eredményez.  
A fraktálok peremén kaotikusan viselkedő rendszerek, amelyeket numerikus módszerekkel lehet vizsgálni.
""")

# Hebbian Learning Viz
st.markdown("---")
st.markdown("### 🧠 Hebbian Learning Viz – Szinaptikus tanulás vizualizációja")
st.latex(r"\Delta w_{ij} = \eta \, x_i \, y_j")
st.markdown("""
A **Hebbian tanulás** klasszikus szabálya szerint az együtt tüzelő neuronok kapcsolata erősödik.  
A modul ezt vizualizálja különböző gráfszerkezeteken.
""")

# Insight Learning
st.markdown("---")
st.markdown("### 💡 Insight Learning – Hirtelen megértés neurális modellezése")
st.markdown("""
Modell, ahol a tanulás nem fokozatos, hanem hirtelen ""megértési"" áttörésen keresztül történik.  
A tanulási ráta küszöb alapján aktiválódik.
""")

# Plasticity Dynamics
st.markdown("---")
st.markdown("### 🔁 Plasticity Dynamics – Időfüggő szinaptikus súlyváltozások")
st.latex(r"\tau \frac{dw}{dt} = -w + f(x, y)")
st.markdown("""
A modul a szinaptikus súlyok változásának dinamikáját mutatja. A különböző $f(x, y)$ függvények különböző biológiai modelleket követnek.
""")

# Persistent Homology
st.markdown("---")
st.markdown("### 🧮 Persistent Homology – Topológiai adatértelmezés")
st.latex(r"\text{PH}_k = \text{Homology}_k(K^\epsilon)")
st.markdown("""
Topológiai módszer, amely a különböző méretskálákon megjelenő lyukakat és kapcsolatokat elemzi.
""")

# Graph Sync Analysis
st.markdown("---")
st.markdown("### 🌐 Graph Sync Analysis – Hálózati szinkronanalízis")
st.latex(r"R(t) = \left| \frac{1}{N} \sum_{j=1}^{N} e^{i\theta_j(t)} \right|")
st.markdown("""
A modul vizsgálja, hogy különböző gráfstruktúrák milyen hatással vannak az oszcillátorok szinkronizációjára.
""")

# További modulok
st.markdown("---")
st.markdown("### 📁 További modulok áttekintése")
st.markdown("""
- **ESN Prediction**: időbeli mintázatok előrejelzése Echo State Network segítségével
- **Generative Kuramoto**: gráfalapú oszcillátor-rendszerek generálása és szinkronizációs jellemzők
- **Lorenz Sim**: a híres kaotikus Lorenz-rendszer vizsgálata
- **Noise Robustness**: zajtűrés vizsgálata különböző neurális modelleken
- **MLP Predict Lorenz**: mély neurális hálózat predikciója a Lorenz-rendszerre
- **Data Upload**: saját adatok feltöltése és elemzése
- **Neural Entropy**: entrópia becslés neurális válaszok alapján
- **Reflection Modul**: saját hipotézisek, jegyzetek és elméletek dokumentálása
- **Help**: ez a jelenlegi dokumentációs felület
""")

st.markdown("---")
st.markdown("## ✍️ Javaslat")
st.markdown("""
Használd a képleteket referencia vagy bemutató célra – vagy a `Reflection Modul` segítségével fűzd hozzá saját értelmezésedet és megfigyelésedet.
""")

st.markdown("""

Verzió: 2025.06
Készítette: ReflectAI fejlesztői és közösség
""")

ReflectAI belépési pont

app = run

