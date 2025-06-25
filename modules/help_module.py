import streamlit as st

def run():
    st.title("❓ Súgó és Dokumentáció – Neurolab AI")
    st.markdown("""
    Üdvözlünk a **Neurolab AI Scientific Playground** alkalmazásban!  
    Ez a sandbox környezet lehetőséget ad különféle idegtudományi, hálózati és tanulási modellek vizsgálatára.

    ---
    ## 🔢 Alapvető matematikai képletek

    ### 🕸️ Kuramoto Modell
    A Kuramoto-modell oszcillátorok szinkronizációját írja le:
    """)
    st.latex(r"""
    \frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^{N} A_{ij} \sin(\theta_j - \theta_i)
    """)
    st.markdown("""
    **Jelölések**:  
    - $\\theta_i$: az *i*-edik oszcillátor fázisa  
    - $\\omega_i$: természetes frekvencia  
    - $K$: kapcsolódási erősség  
    - $A_{ij}$: kapcsolódási mátrix  
    - $N$: oszcillátorok száma  
    """)
    st.markdown("A szinkronizációs mérték:")
    st.latex(r"""
    R(t) = \left| \frac{1}{N} \sum_{j=1}^{N} e^{i\theta_j(t)} \right|
    """)

    st.markdown("---\n### ❌ XOR predikció – neurális hálózat")
    st.markdown("""
    A cél, hogy megtanítsuk egy hálózatnak az XOR logikai műveletet:

    | x₁ | x₂ | XOR |
    |----|----|-----|
    | 0  | 0  |  0  |
    | 0  | 1  |  1  |
    | 1  | 0  |  1  |
    | 1  | 1  |  0  |

    Egy egyszerű MLP esetén:
    """)
    st.latex(r"""
    \hat{y} = \sigma\left( W^{(2)} \cdot \sigma(W^{(1)} \cdot x + b^{(1)}) + b^{(2)} \right)
    """)
    st.markdown("""
    Ahol $\\sigma(x) = \\frac{1}{1 + e^{-x}}$ a szigmoid aktiváció.  
    A tanulás célja: minimalizálni az átlagos kvadratikus hibát:
    """)
    st.latex(r"""
    \mathcal{L} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
    """)

    st.markdown("---\n### 🌐 Berry-görbület")
    st.markdown("A topológiai védelem szimulációjához Berry-görbületet számítunk:")
    st.latex(r"""
    \Omega(\mathbf{k}) = \nabla_{\mathbf{k}} \times \mathbf{A}(\mathbf{k})
    """)
    st.markdown("Ahol az **Berry-kapcsolat**:")
    st.latex(r"""
    \mathbf{A}(\mathbf{k}) = -i \langle u(\mathbf{k}) | \nabla_{\mathbf{k}} | u(\mathbf{k}) \rangle
    """)

    st.markdown("---\n## 🧪 Bizonyítási ötletek")
    st.markdown("""
    - A Kuramoto-modell globális szinkronizációja analitikus módon is igazolható a kapcsolati gráf és $K$ értéke alapján (*Strogatz-féle stabilitási feltétel*).
    - Az XOR taníthatósága nem lineáris, tehát **legalább egy rejtett réteg** szükséges az univerzális approximátor tétel alapján.
    - A Berry-görbület invariánsai (Chern-szám) **topológiai kvantumállapotokat** különböztetnek meg – numerikus szimulációval is kimutathatók.
    """)

    st.markdown("---\n## ✍️ Javaslat")
    st.markdown("""
    Használd a képleteket referencia vagy bemutató célra –  
    vagy a `Reflection Modul` segítségével fűzd hozzá saját értelmezésedet és megfigyelésedet.
    """)

    st.markdown("---\nVerzió: **2025.06**  
    Készítette: *ReflectAI fejlesztői és közösség*")

# ReflectAI integrációhoz szükséges
app = run
