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

    st.markdown("#### 📘 Napi önreflexió")
    st.markdown("""
    **Cél:** A napi tanulási és érzelmi állapotok tudatosítása, kognitív metareflexió fejlesztése.  
    **Tudományos háttér:** Metakogníció, önszabályozott tanulás, pszichológiai naplózás.

    **Koncepció:** A kérdések különböző kognitív és érzelmi szinteken aktiválják a tanulót.

    **Adat:** A modul a `questions.json` fájlból dolgozik.

    **Alkalmazások:**  
    - Mentálhigiénés gyakorlatok  
    - Oktatási önértékelések  
    - AI-támogatott coaching modulok
    """)

    st.markdown("#### 🧮 Perzisztens homológia")
    st.markdown("""
    **Cél:** Topológiai Data Analysis (TDA) eszközeként a rejtett adatstruktúrák vizsgálata.  
    **Tudományos háttér:** Algebrai topológia, Vietoris–Rips komplexumok, Betti-számok.

    **Módszer:** A szintetikus pontfelhők topológiai jellemzőinek analízise.

    **Alkalmazások:**  
    - Orvosi képalkotás  
    - Hálózatelemzés  
    - Gépi tanulási előfeldolgozás
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

    st.caption("Frissítve: 2025-06-23 16:46")
