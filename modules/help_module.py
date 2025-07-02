# -*- coding: utf-8 -*-
import streamlit as st

def run():
    st.set_page_config(page_title="Súgó – Neurolab", layout="wide")
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
    st.latex(r"R(t) = \left| \frac{1}{N} \sum_{j=1}^{N} e^{i\theta_j(t)} \right|")
    st.markdown("""
    A Kuramoto-modell kollektív fázisszinkronizáció vizsgálatára alkalmas.  
    Használat: gráf alapú oszcillátor-hálózatok dinamikájának szimulációja.  
    [Tudományos forrás](https://doi.org/10.1016/0370-1573(84)90022-1)
    """)

    # XOR
    st.markdown("### ❌ XOR – Neurális hálózat klasszikus példája")
    st.markdown("""
    | x₁ | x₂ | XOR |
    |----|----|-----|
    | 0  | 0  |  0  |
    | 0  | 1  |  1  |
    | 1  | 0  |  1  |
    | 1  | 1  |  0  |
    """)
    st.latex(r"\hat{y} = \sigma\left( W^{(2)} \cdot \sigma(W^{(1)} x + b^{(1)}) + b^{(2)} \right)")
    st.markdown("""
    A megoldáshoz nemlineáris rejtett réteg szükséges (MLP).  
    [Tudományos forrás](https://cs231n.github.io/neural-networks-1/)
    """)

    # Berry curvature
    st.markdown("### 🌐 Berry-görbület – Kvantumtopológia")
    st.latex(r"\Omega(\mathbf{k}) = \nabla_{\mathbf{k}} \times \mathbf{A}(\mathbf{k})")
    st.latex(r"\mathbf{A}(\mathbf{k}) = -i \langle u(\mathbf{k}) | \nabla_{\mathbf{k}} | u(\mathbf{k}) \rangle")
    st.markdown("""
    A Berry-görbület topológiai fázisok feltárására alkalmas.  
    [Tudományos forrás](https://doi.org/10.1103/RevModPhys.82.1959)
    """)

    # Hopfield háló
    st.markdown("### 🧠 Hopfield-háló – Memória Dinamika")
    st.latex(r"W_{ij} = \sum_{\mu=1}^P \xi_i^\mu \xi_j^\mu,\quad W_{ii} = 0")
    st.latex(r"s_i^{t+1} = \text{sign} \left( \sum_j W_{ij} s_j^t \right)")
    st.markdown("""
    Egy asszociatív memóriaháló, amely stabil mintákat tárol.  
    [Tudományos forrás](https://www.nature.com/articles/nn.4401)
    """)

    # Fraktál Explorer
    st.markdown("### 🌀 Fraktál Explorer – Mandelbrot és káosz")
    st.latex(r"z_{n+1} = z_n^2 + c")
    st.markdown("""
    Mandelbrot-halmaz vizualizációja komplex síkon.  
    [Tudományos forrás](https://mathworld.wolfram.com/MandelbrotSet.html)
    """)

    # ESN
    st.markdown("### 🧠 ESN Prediction – Echo State Network")
    st.markdown("""
    Idősoros előrejelzés belső dinamikus reprezentációval.  
    Használ: rekurrens neuronháló fix rejtett súlyokkal.  
    [Tudományos forrás](https://www.sciencedirect.com/science/article/abs/pii/S0893608005001603)
    """)

    # Generative Kuramoto
    st.markdown("### 🔁 Generative Kuramoto – Oszcillátor gráfok")
    st.markdown("""
    Paraméterezhető gráfgenerálás és szinkronizációs tesztelés.  
    Tetszőleges topológiákhoz alkalmazható.
    """)

    # Graph Sync Analysis
    st.markdown("### 📈 Graph Sync Analysis")
    st.markdown("""
    Szinkronizáció vizsgálata gráfok spektrális tulajdonságai alapján.  
    Kiemelten a Laplace-mátrix és kapcsolási mintázatok elemzése.  
    [Tudományos forrás](https://arxiv.org/abs/1205.5709)
    """)

    # Hebbian / Plasticity
    st.markdown("### 🔗 Hebbian tanulás és szinaptikus plaszticitás")
    st.latex(r"\Delta w_{ij} \propto x_i x_j")
    st.markdown("""
    "Fire together, wire together" – klasszikus Hebbian-elv.  
    Alkalmazása: tanulási szabályok és hálózati adaptációk modellezése.
    """)

    # Persistent Homology
    st.markdown("### 🏞️ Persistent Homology – Topológiai Adatanalízis")
    st.markdown("""
    Részhalmazokon keresztüli geometriai jellemzők fennmaradása.  
    Alkalmazása: időbeli vagy gráfalapú adatstruktúrák feltárása.  
    [Tudományos forrás](https://www.ams.org/journals/notices/201101/rtx110100014p.pdf)
    """)

    # Reflection modul
    st.markdown("### 🪞 Reflection modul – Hipotézisek és saját elemzések")
    st.markdown("""
    Jegyzetek, saját levezetett képletek, megfigyelések rögzítése.  
    Segíti a személyre szabott kutatási napló vezetését.
    """)

    # Ajánlások
    st.markdown("---")
    st.markdown("## ✅ Ajánlott használat")
    st.markdown("""
    - Kombinálj több modult, például Kuramoto + Fraktál  
    - Tölts fel saját adatot, és futtass tanulást ESN-nel  
    - Használd a `Reflection` modult önálló megfigyelésekhez  
    - Próbáld ki az animált gráf- és fázistér vizualizációkat  
    """)

    # Zárás
    st.markdown("---")
    st.markdown("""
Verzió: **2025.07.02**  
Készítette: *ReflectAI és közösség*  
GitHub: [NeurolabAI Sandbox](https://github.com/your-repo)
    """)

# Belépési pont
app = run
