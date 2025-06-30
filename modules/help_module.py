import streamlit as st

def run():
    st.title("❓ Súgó és Dokumentáció – Neurolab AI")
    st.markdown("""
    Üdvözlünk a **Neurolab AI Scientific Playground** alkalmazásban!  
    Ez a sandbox környezet lehetőséget ad különféle idegtudományi, hálózati és tanulási modellek vizsgálatára.

    ---
    ## 🔢 Alapvető matematikai képletek
    """)

    # Kuramoto modell
    st.markdown("### 🕸️ Kuramoto Modell – Szinkronizációs Dinamika")
    st.latex(r"\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^{N} A_{ij} \sin(\theta_j - \theta_i)")
    st.markdown("""
    **Jelölések**:  
    - $\\theta_i$: az *i*-edik oszcillátor fázisa  
    - $\\omega_i$: természetes frekvencia  
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
    st.markdown("Ahol $\\sigma(x) = \\frac{1}{1 + e^{-x}}$ a szigmoid aktiváció.")
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

    # További modulok listája
    st.markdown("---")
    st.markdown("## ✅ Egyéb modulok áttekintése")
    st.markdown("""
    - **ESN Prediction**: időbeli mintázatok előrejelzése Echo State Network segítségével  
    - **Generative Kuramoto**: gráfalapú oszcillátor-rendszerek generálása  
    - **Graph Sync Analysis**: gráfstruktúra és szinkronizáció kapcsolatának vizsgálata  
    - **Hebbian Learning / Plasticity Dynamics**: szinaptikus tanulás és dinamikus súlyváltozások  
    - **Persistent Homology**: topológiai adatértelmezés  
    - **Reflection Modul**: saját megfigyelések és levezetett hipotézisek dokumentálása  
    """)

    st.markdown("---")
    st.markdown("## 🧪 Bizonyítási ötletek")
    st.markdown("""
    - A Kuramoto-modell fázisszinkronizációja gráfelméleti módszerekkel is igazolható  
    - Az XOR taníthatósága bemutatja a nemlinearitás szerepét a neurális hálókban  
    - A Berry-görbület invariánsai segítik a topológiai állapotok megkülönböztetését  
    - A Hopfield-háló minimum energiára törekszik, így stabil mintatárolóként működik  
    """)

    st.markdown("---")
    st.markdown("## ✍️ Javaslat")
    st.markdown("""
    Használd a képleteket referencia vagy bemutató célra – vagy a `Reflection Modul` segítségével fűzd hozzá saját értelmezésedet és megfigyelésedet.
    """)

    st.markdown("---")
    st.markdown("Verzió: **2025.06**  
    Készítette: *ReflectAI fejlesztői és közösség*")

# ReflectAI belépési pont
app = run
