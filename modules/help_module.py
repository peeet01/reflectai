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
        
    with st.expander("🧠 Hebbian Learning Viz – Szinaptikus erősödés szemléltetése"):
        st.latex(r"\Delta w = \eta \cdot x \cdot y")
        st.markdown("""
        **Cél:** A Hebbian tanulás bemutatása vizuálisan, ahol a bemenet ($x$) és a kimenet ($y$) együttes aktivitása megerősíti a szinaptikus kapcsolatot ($w$).  

        **Magyarázat:**
        - $x$: bemeneti neuron aktivitása  
        - $y$: kimeneti neuron aktivitása  
        - $\eta$: tanulási ráta  
        - $\Delta w$: szinaptikus súlyváltozás

        Ez a szabály a híres "Cells that fire together, wire together" elvét követi.  
        A modul vizuálisan mutatja be, hogy a gyakori együttes aktiváció hogyan növeli a kapcsolatok erősségét.

        **Alkalmazás:**
        - Neurális adaptációk vizsgálata  
        - Biológiai tanulási folyamatok megértése  
        - Dinamikus hálózati súlymódosulások elemzése

        **Tudományos háttér:**  
        A Hebbian tanulás egy alapvető **unsupervised learning** mechanizmus, amely a korrelált aktivitást preferálja, és a **neurális reprezentációk kialakulását** modellezi.
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

    with st.expander("🧠 Memory Landscape – Asszociatív tárolási térkép"):
        st.latex(r"E(\mathbf{s}) = -\frac{1}{2} \sum_{i \neq j} W_{ij} s_i s_j")
        st.markdown("""
        **Cél:** A neurális hálózat memóriastruktúráinak vizuális feltérképezése az energiafüggvény alapján.  

        **Magyarázat:**
        - $E(\\mathbf{s})$: az adott állapothoz tartozó hálózati energia  
        - $W_{ij}$: szinaptikus súlymátrix elemei  
        - $s_i$: az $i$-edik neuron állapota ($\\pm1$)  

        **Felhasználás:**
        - Minták stabilitásának és robusztusságának vizsgálata  
        - Lokális minimumok detektálása az energiafelületen  
        - Asszociatív memória térképezése (pl. Hopfield-háló)

        **Tudományos háttér:**  
        A memóriatárolás úgy történik, hogy a mintákhoz **energia-minimumok** rendelődnek. A hálózat dinamika szerint ezekbe a minimumokba **konvergál**:
        """)
        st.latex(r"s_i^{(t+1)} = \mathrm{sign} \left( \sum_j W_{ij} s_j^{(t)} \right)")
        
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

    with st.expander("🌋 Criticality Explorer – Neurális rendszerek kritikus állapota"):
        st.latex(r"P(s) \propto s^{-\tau}")
        st.markdown("""
        **Cél:** A neurális rendszerek **önszerveződő kritikus viselkedésének** vizsgálata.  
        A kritikus pontokon megfigyelhető, hogy az aktivitáseloszlás **skálafüggetlen**, azaz **hatványfüggvény** szerint alakul.

        **Magyarázat:**
        - $P(s)$: egy adott $s$ méretű aktivitási esemény valószínűsége  
        - $\\tau$: hatványkitevő (tipikusan $\\sim 1.5$ körül)
 
        **Jellemzők:**
        - Nincs jellegzetes méret: **kis és nagy aktivitások** is előfordulnak  
        - **Kritikus lejtő** jelenik meg log-log skálán  
        - A rendszer érzékenyen reagál a bemenetekre

        **Felhasználás:**
        - Agyi aktivitás hullámzásainak (avalanches) modellezése  
        - Kritikus állapotok keresése és jellemzése  
        - Komplex rendszerek stabilitásának és tanulékonyságának optimalizálása

        **Megjegyzés:**  
        A kritikusitás közelében a hálózat **maximális információfeldolgozási kapacitással** működhet.
        """)

    with st.expander("📉 Lyapunov Spectrum – Kaotikus rendszerek stabilitása"):
        st.latex(r"\lambda_i = \lim_{t \to \infty} \frac{1}{t} \ln \frac{||\delta x_i(t)||}{||\delta x_i(0)||}")
        st.markdown("""
        **Cél:** A dinamikus rendszer stabilitásának vizsgálata a Lyapunov-exponenseken keresztül.  

        **Magyarázat:**
        - $\lambda_i$: az $i$-edik Lyapunov-exponens  
        - $\delta x_i$: perturbáció az állapottérben  
        - A pozitív $\lambda$ értékek a rendszer **kaotikusságára** utalnak  
        - A negatív értékek stabilitást jeleznek, míg a nulla semleges viselkedést

        **Felhasználás:**
        - Kaotikus rendszerek detektálása  
        - Stabil/instabil viselkedés feltérképezése  
        - Lorenz-rendszer, Rössler-attraktor, Kuramoto-hálózatok vizsgálata  

        **Tudományos háttér:**  
        A Lyapunov-spektrum a **nemlineáris dinamika** egyik alapvető eszköze. A teljes spektrum jellemzi a rendszer entrópiáját és prediktálhatóságát:
        """)
        st.latex(r"h_{KS} = \sum_{\lambda_i > 0} \lambda_i \quad \text{(Kolmogorov–Sinai entrópia)}")

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
