import streamlit as st

def run():
    st.title("❓ Súgó – Neurolab AI Scientific Playground Sandbox")

    st.markdown("""
    ## 🔍 Mi ez az alkalmazás?
    A **Neurolab AI** egy interaktív kutatási platform, amely lehetővé teszi különféle mesterséges intelligencia, hálózati dinamika és matematikai modellek vizsgálatát. A cél egy nyitott, vizualizáció-alapú, moduláris sandbox létrehozása kutatók, hallgatók és oktatók számára.

    ---

    ## 🧭 Modulismertető – Tudományos és történeti háttérrel
    """)

    st.markdown("### 🔁 XOR predikció neurális hálóval")
    st.markdown("""
    **Cél:** A klasszikus XOR logikai függvény megtanítása mesterséges neurális hálóval.  
    **Tudományos háttér:** Az XOR probléma a 80-as években kulcsszerepet játszott a mély tanulás fejlődésében. Egyetlen lineáris réteg nem tudja megoldani, így legalább egy rejtett rétegre van szükség.
    
    **Alkalmazott képlet:**  
    A kimenet:  
    $$
    \hat{y} = \sigma\left(W_2 \cdot \tanh(W_1 \cdot x + b_1) + b_2\right)
    $$
    ahol $\\sigma$ a szigmoid aktiváció, $W_i$, $b_i$ a hálózat súlyai és biasai.
    """)

    st.markdown("### 🧭 Kuramoto szinkronizáció")
    st.markdown("""
    **Cél:** Oszcillátorok kollektív szinkronizációjának vizsgálata.  
    **Tudományos háttér:** A Kuramoto-modellt Yoshiki Kuramoto japán fizikus vezette be 1975-ben. A modell bemutatja, hogyan képesek egymással kapcsolatban lévő oszcillátorok szinkronizálódni.

    **Kuramoto-egyenlet:**
    $$
    \\frac{d\\theta_i}{dt} = \\omega_i + \\frac{K}{N} \\sum_{j=1}^{N} \\sin(\\theta_j - \\theta_i)
    $$

    A szinkronizáció mértéke:
    $$
    r(t) = \\left|\\frac{1}{N} \\sum_{j=1}^N e^{i \\theta_j(t)} \\right|
    $$
    """)

    st.markdown("### 🧠 Hebbian tanulás")
    st.markdown("""
    **Cél:** A tanulás biológiai modellje – ha két neuron egyidejűleg aktiválódik, akkor kapcsolatuk erősödik.  
    **Történeti háttér:** Donald Hebb 1949-ben fogalmazta meg ezt az elvet, ami az egyik első formális tanulási szabály volt az agyban.

    **Hebb-szabály:**  
    $$
    \\Delta w_{ij} = \\eta \\, x_i \\, y_j
    $$
    ahol $x_i$ a bemenet, $y_j$ a kimenet, $\\eta$ a tanulási ráta.
    """)

    st.markdown("### ⚡ Kuramoto–Hebbian hálózat")
    st.markdown("""
    **Cél:** Dinamikus oszcillátorhálózatok és adaptív tanulási szabály összekapcsolása.  
    **Magyarázat:** A Kuramoto-dinamika hat a kapcsolat súlyaira, miközben a Hebbian-szabály az összekötések erősségét adaptálja a szinkronizáció függvényében.

    $$
    \\Delta w_{ij}(t) \\propto \\cos(\\theta_i(t) - \\theta_j(t))
    $$
    """)

    st.markdown("### 🌀 Lorenz rendszer")
    st.markdown("""
    **Cél:** A kaotikus viselkedés vizsgálata determinisztikus rendszerben.  
    **Történeti háttér:** Edward Lorenz 1963-as meteorológiai modellje volt az egyik első példája a káoszelméletnek.

    **Lorenz-egyenletek:**
    $$
    \\begin{aligned}
    \\frac{dx}{dt} &= \\sigma(y - x) \\\\
    \\frac{dy}{dt} &= x(\\rho - z) - y \\\\
    \\frac{dz}{dt} &= xy - \\beta z
    \\end{aligned}
    $$
    """)

    st.markdown("### 🔮 Lorenz predikció")
    st.markdown("""
    **Cél:** Mély neurális hálózat segítségével előrejelezni a Lorenz-rendszer jövőbeli állapotait.  
    **Magyarázat:** A modell a korábbi állapotokból tanulva jósolja meg a következő lépéseket.

    $$ \hat{x}_{t+1} = f(x_t, x_{t-1}, ...) $$
    """)

    st.markdown("### 🧬 Zajtűrés és robusztusság")
    st.markdown("""
    **Cél:** A modellek érzékenységének mérése véletlenszerű zajra.  
    **Motiváció:** A biológiai rendszerek gyakran robusztusak a hibák ellen, ezt modellezzük mesterséges rendszerekben.
    """)

    st.markdown("### 🧩 Chern–szám analízis")
    st.markdown("""
    **Cél:** Kvantumtopológiai jellemzők numerikus vizsgálata (pl. Berry-görbület).  
    **Képlet:**
    $$
    C = \\frac{1}{2\\pi} \\int_{BZ} F(k) \\, d^2k
    $$
    ahol $F(k)$ a Berry-görbület, $BZ$ a Brillouin-zóna.
    """)

    st.markdown("### 📈 Echo State Network")
    st.markdown("""
    **Cél:** Dinamikus idősorok tanulása visszacsatolt hálóval.  
    **Képlet:**
    $$
    x(t+1) = \\tanh(W_{res} \\, x(t) + W_{in} \\, u(t))
    $$
    """)

    st.markdown("### 🧠 Insight Learning")
    st.markdown("""
    **Cél:** Tanulás, amely hirtelen felismerésből következik, nem fokozatos fejlődésből.  
    **Pszichológiai alap:** Köhler és a Gestalt-pszichológia elveiből származik.
    """)

    st.markdown("### 🧠 Generatív Kuramoto hálózat")
    st.markdown("""
    **Cél:** Új gráfstruktúrák generálása és szinkronizációjuk vizsgálata Kuramoto-modell segítségével.
    """)

    st.markdown("### 🧭 Memória tájkép")
    st.markdown("""
    **Cél:** Neurális hálók stabil állapotainak (memória pontok) feltérképezése.  
    **Elmélet:** A tájkép lokális minimumai stabil állapotokként viselkednek.
    """)

    st.markdown("---")
    st.markdown("## 👥 Célcsoport")
    st.markdown("""
    - **Kutatók:** Topológiai, tanulási vagy dinamikai modellek gyors tesztelése  
    - **Oktatók:** Vizualizációk és oktatási segédanyagok  
    - **Hallgatók:** Modellkísérletezés és tanulás  
    - **Fejlesztők:** Rugalmas és bővíthető Python/Streamlit sandbox
    """)
