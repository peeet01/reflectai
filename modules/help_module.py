import streamlit as st

def run():
    st.title("❓ Tudományos Súgó – Neurolab AI")
    st.markdown("""
    Üdvözlünk a **Neurolab AI Scientific Playground** felületen!  
    Ez az alkalmazás különféle elméleti és gyakorlati idegrendszeri, fizikai és matematikai modellek interaktív vizsgálatát támogatja.

    Az alábbiakban bemutatjuk az egyes modulok **matematikai alapjait**, **célját**, és **következtetéseit**.
    """)

    with st.expander("🕸️ Kuramoto Modell – Szinkronizációs Dinamika"):
        st.latex(r"\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^{N} A_{ij} \sin(\theta_j - \theta_i)")
        st.markdown("""
        **Kollektív viselkedés:** Egyedi oszcillátorok fázisszinkronizációját írja le.  
        Order parameter:
        """)
        st.latex(r"R(t) = \left| \frac{1}{N} \sum_{j=1}^{N} e^{i\theta_j(t)} \right|")
        st.markdown("""
        **Cél:** Modellezi a neuronhálózatok kollektív állapotait.  
        **Felhasználás:**
        - Hálózati instabilitások detektálása  
        - Szinkron idegi aktivitás jellemzése  
        - Ritmikus zavarok (pl. epilepszia) szimulációja  
        """)

    with st.expander("❌ XOR Predikció – Neurális hálózat"):
        st.latex(r"\hat{y} = \sigma(W^{(2)} \cdot \sigma(W^{(1)}x + b^{(1)}) + b^{(2)})")
        st.latex(r"\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2")
        st.markdown("""
        **Cél:** A nemlineárisan szeparálható problémák (pl. XOR) rejtett rétegekkel oldhatók meg.  
        **Felhasználás:**
        - Mély tanulási architektúrák motivációja  
        - Lineáris modellek korlátainak bemutatása  
        """)

    with st.expander("🌐 Berry-görbület – Kvantum topológia"):
        st.latex(r"\Omega(\mathbf{k}) = \nabla_{\mathbf{k}} \times \mathbf{A}(\mathbf{k}) \quad \text{ahol} \quad \mathbf{A}(\mathbf{k}) = -i \langle u(\mathbf{k}) | \nabla_{\mathbf{k}} | u(\mathbf{k}) \rangle")
        st.markdown("""
        **Cél:** Kvantált topológiai mennyiségek megjelenítése.  
        **Felhasználás:**
        - Kvantum Hall-effektus modellezése  
        - Topológiai különbségek azonosítása  
        """)

    with st.expander("🧠 Hopfield-háló – Asszociatív memória"):
        st.latex(r"W_{ij} = \sum_{\mu=1}^{P} \xi_i^\mu \xi_j^\mu")
        st.latex(r"s_i^{t+1} = \mathrm{sign} \left( \sum_j W_{ij} s_j^t \right)")
        st.markdown("""
        **Cél:** Minták visszakeresése zajos állapotból.  
        **Felhasználás:**
        - Memóriaállapotok visszanyerése  
        - Energia-alapú tanulási dinamika  
        """)

    with st.expander("🧮 Fractal Dimension – Ön-szimiláris szerkezetek"):
        st.latex(r"D = \lim_{\epsilon \to 0} \frac{\log N(\epsilon)}{\log (1/\epsilon)}")
        st.markdown("""
        **Cél:** A fraktáldimenzió a klasszikus dimenzió általánosítása, amely azt méri, hogy egy objektum **mennyire tölti ki a teret** különböző skálákon.  
        A fenti képlet a **box-counting dimenzió** képlete, ahol:
        - $N(\\epsilon)$: az objektum lefedéséhez szükséges $\epsilon$ méretű dobozok száma  
        - $\epsilon$: a doboz mérete

        **Alkalmazás:**
        - Kaotikus attraktorok és természetes mintázatok (pl. felhők, erezetek) komplexitásának mérése  
        - Neurális aktivitás komplexitásának jellemzése  
        - MRI és EEG adatok skálafüggetlen szerkezeteinek feltárása  

        **Jellemzők:**
        - A fraktáldimenzió lehet **nem egész szám**, pl. Mandelbrot-halmaz: $D \\approx 1.26$  
        - A dimenzió nő, ha az objektum egyre inkább „kitölti” a teret.

        **Tudományos jelentőség:**  
        A fraktáldimenzió segítségével **rendszerek komplexitása** jellemezhető kvantitatív módon, különösen ott, ahol klasszikus mértékek (pl. topológiai dimenzió) csődöt mondanak.
        """)

    with st.expander("🌀 Fraktál Explorer – Kaotikus rendszerek"):
        st.latex(r"z_{n+1} = z_n^2 + c")
        st.markdown("""
        **Cél:** Mandelbrot- és Julia-halmazok megjelenítése.  
        **Felhasználás:**
        - Stabil és kaotikus zónák feltárása  
        - Nemlineáris dinamika vizualizálása  
        """)

    with st.expander("🔄 Echo State Network – Idősoros előrejelzés"):
        st.latex(r"\mathbf{x}(t+1) = \tanh(W_{res} \cdot \mathbf{x}(t) + W_{in} \cdot \mathbf{u}(t))")
        st.latex(r"\hat{y}(t) = W_{out} \cdot \mathbf{x}(t)")
        st.markdown("""
        **Cél:** Idősoros előrejelzés kis tanítási költséggel.  
        **Felhasználás:**
        - Komplex rendszerek predikciója  
        - Dinamikus mintafelismerés  
        """)

    with st.expander("🧩 Generative Kuramoto – Struktúra és dinamika"):
        st.markdown("""
        **Cél:** Random gráf generálása és annak dinamikai szimulációja.  
        **Felhasználás:**
        - Gráf topológia és szinkronizáció kapcsolatának feltárása  
        """)

    with st.expander("🧮 Graph Sync Analysis – Hálózati stabilitás"):
        st.markdown("""
        **Cél:** Szinkronizáció erőssége és Laplace spektrum elemzése.  
        **Felhasználás:**
        - Stabilitás és hálózatszerkezet összefüggéseinek feltárása  
        """)

    with st.expander("🏔️ Persistent Homology – Topológiai adatértelmezés"):
        st.markdown("""
        **Cél:** Perzisztens topológiai struktúrák kiszűrése.  
        **Felhasználás:**
        - Zaj és valódi szerkezet megkülönböztetése  
        - Gépi tanulási jellemzők generálása  
        """)

    st.markdown("""---  
    Verzió: **2025.07**  
    Készítette: *ReflectAI fejlesztői és tudományos tanácsadók*  
    """)

# Kötelező belépési pont
app = run
