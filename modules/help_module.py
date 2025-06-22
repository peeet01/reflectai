import streamlit as st

def run():
    st.title("❓ Súgó – Neurolab AI Scientific Playground Sandbox")

    st.markdown("""
    ## 🔍 Mi ez az alkalmazás?
    A **Neurolab AI** egy interaktív kutatási platform, amely lehetővé teszi különféle mesterséges intelligencia, hálózati dinamika és matematikai modellek vizsgálatát. A cél egy nyitott, vizualizáció-alapú, moduláris sandbox létrehozása kutatók, hallgatók és oktatók számára.
    """)

    st.markdown("## 🧭 Modulismertető – Tudományos és történeti háttérrel")

    st.markdown("### 🔁 XOR predikció neurális hálóval")
    st.markdown("""
    **Cél:** A klasszikus XOR logikai függvény megtanítása mesterséges neurális hálóval.  
    **Tudományos háttér:** Az XOR probléma a 80-as években kulcsszerepet játszott a mély tanulás fejlődésében. Egyetlen lineáris réteg nem tudja megoldani, így legalább egy rejtett rétegre van szükség.

    **Alkalmazott képlet:**  
    $$
    \\hat{y} = \\sigma\\left(W_2 \\cdot \\tanh(W_1 \\cdot x + b_1) + b_2\\right)
    $$

    **Paraméterek:**  
    - Rejtett réteg mérete  
    - Tanulási ráta  
    - Epochok száma

    **Alkalmazások:**  
    - Nemlineáris problémák tanítása  
    - Aktivációs függvények hatása  
    - Gépi tanulás alapjai
    """)

    st.markdown("### 🧭 Kuramoto szinkronizáció")
    st.markdown("""
    **Cél:** Oszcillátorok kollektív szinkronizációjának vizsgálata.  
    **Tudományos háttér:** Yoshiki Kuramoto japán fizikus 1975-ben írta le a modellt.

    **Kuramoto-egyenlet:**  
    $$
    \\frac{d\\theta_i}{dt} = \\omega_i + \\frac{K}{N} \\sum_{j=1}^{N} \\sin(\\theta_j - \\theta_i)
    $$

    **Order parameter:**  
    $$
    r(t) = \\left|\\frac{1}{N} \\sum_{j=1}^N e^{i \\theta_j(t)} \\right|
    $$

    **Paraméterek:**  
    - Kapcsolási erősség  
    - Oszcillátorok száma

    **Alkalmazások:**  
    - Idegi ritmusok  
    - Biológiai oszcillációk  
    - Szinkronizációs zavarok elemzése
    """)

    st.markdown("### 🧠 Hebbian tanulás")
    st.markdown("""
    **Cél:** Biológiai ihletésű tanulási szabály vizsgálata.  
    **Háttér:** Donald Hebb (1949) – „ami együtt tüzel, összekapcsolódik”.

    **Hebb-szabály:**  
    $$
    \\Delta w_{ij} = \\eta \\, x_i \\, y_j
    $$

    **Paraméterek:**  
    - Tanulási ráta  
    - Neuronok száma

    **Alkalmazások:**  
    - Szinaptikus plaszticitás modellezése  
    - Egyszerű memóriahálók
    """)

    st.markdown("### ⚡ Kuramoto–Hebbian hálózat")
    st.markdown("""
    **Cél:** Szinkronizáció és tanulás egyesítése.  
    **Elv:** Dinamikus gráf súlyainak változása Hebbian szabály szerint, Kuramoto fáziskülönbségek alapján.

    $$
    \\Delta w_{ij}(t) \\propto \\cos(\\theta_i(t) - \\theta_j(t))
    $$

    **Alkalmazások:**  
    - Adaptív szinkronizáció  
    - Bioinspirált hálózatok
    """)

    st.markdown("### 🌀 Lorenz rendszer")
    st.markdown("""
    **Cél:** Káosz és determinisztikus dinamika vizsgálata.  
    **Háttér:** Edward Lorenz (1963) – meteorológiai modellekből kiindulva.

    **Lorenz-egyenletek:**  
    $$
    \\begin{aligned}
    \\frac{dx}{dt} &= \\sigma(y - x) \\\\
    \\frac{dy}{dt} &= x(\\rho - z) - y \\\\
    \\frac{dz}{dt} &= xy - \\beta z
    \\end{aligned}
    $$

    **Alkalmazások:**  
    - Idősor szimuláció  
    - Káoszelmélet oktatása
    """)

    st.markdown("### 🔮 Lorenz predikció")
    st.markdown("""
    **Cél:** Mély hálózatokkal előrejelezni a Lorenz dinamikát.  
    **Elv:** Korábbi állapotok alapján tanulható nemlineáris viselkedés.

    $$
    \\hat{x}_{t+1} = f(x_t, x_{t-1}, ...)
    $$

    **Alkalmazások:**  
    - Idősor-előrejelzés  
    - Káoszdetekció gépi tanulással
    """)

    st.markdown("### 🧬 Zajtűrés és robusztusság")
    st.markdown("""
    **Cél:** Mesterséges rendszerek érzékenysége a zajra.  
    **Motiváció:** Biológiai rendszerek gyakran robusztusak hibák ellen.

    **Alkalmazások:**  
    - Hibatűrő rendszerek  
    - Szimulációk érzékenységi vizsgálata
    """)

    st.markdown("### 🧩 Chern–szám analízis")
    st.markdown("""
    **Cél:** Topológiai invariánsok számítása kvantumrácsokon.

    **Képlet:**  
    $$
    C = \\frac{1}{2\\pi} \\int_{BZ} F(k) \\, d^2k
    $$  
    $F(k)$: Berry-görbület

    **Alkalmazások:**  
    - Topológiai szigetelők modellezése  
    - Kvantum számítási struktúrák
    """)

    st.markdown("### 📈 Echo State Network (ESN)")
    st.markdown("""
    **Cél:** Időfüggő mintázatok megtanulása visszacsatolt hálókkal.

    **Képlet:**  
    $$
    x(t+1) = \\tanh(W_{res}x(t) + W_{in}u(t))
    $$

    **Alkalmazások:**  
    - Idősorok előrejelzése  
    - Viselkedésminták tanulása
    """)

    st.markdown("### 🧠 Insight Learning")
    st.markdown("""
    **Cél:** Belátás-alapú tanulás szimulációja.  
    **Háttér:** Köhler és Gestalt pszichológia elvein alapul.

    **Alkalmazások:**  
    - Hirtelen felismerések modellezése  
    - Tanulási áttörések
    """)

    st.markdown("### 🧠 Generatív Kuramoto hálózat")
    st.markdown("""
    **Cél:** Dinamikus gráfok generálása és szinkronizációs tulajdonságaik vizsgálata.

    **Alkalmazások:**  
    - Véletlen hálók dinamika szerinti evolúciója  
    - Gráfelméleti szinkronizáció
    """)

    st.markdown("### 🧭 Memória tájkép")
    st.markdown("""
    **Cél:** Neurális hálók stabil állapotainak feltérképezése.  
    **Elv:** A háló energiatájképének minimumai jelzik a memóriapontokat.

    **Alkalmazások:**  
    - Hopfield-hálók vizsgálata  
    - Attractor-alapú tanulás
    """)

    st.markdown("### 📊 Lyapunov spektrum")
    st.markdown("""
    **Cél:** Egy rendszer kaotikusságának számszerűsítése.

    **Definíció:**  
    $$
    \\lambda = \\lim_{t \\to \\infty} \\frac{1}{t} \\ln \\frac{\\delta(t)}{\\delta(0)}
    $$

    **Alkalmazások:**  
    - Káoszdetekció  
    - Stabilitásvizsgálat
    """)

    st.markdown("---")
    st.markdown("## 👥 Célcsoport")
    st.markdown("""
    - **Kutatók:** Gyors modelltesztelés, vizualizációk, elméleti kísérletek  
    - **Oktatók:** Oktatási szemléltető eszköz, matematikai és gépi tanulási példák  
    - **Hallgatók:** Interaktív tanulás, paraméterkísérletezés, önálló kutatási projektek  
    - **Fejlesztők:** Moduláris és nyílt rendszer új ötletek prototipizálására
    """)
