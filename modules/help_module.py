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
        
    with st.expander("🧠 Hebbian Learning – Szinaptikus erősítés elve"):
        st.latex(r"\Delta w_{ij} = \eta \cdot x_i \cdot y_j")
        st.markdown("""
        **Cél:** A tanulás során megerősíteni azokat a kapcsolatokat, amelyek gyakran aktiválódnak együtt.

        **Magyarázat:**
        - $x_i$: preszinaptikus neuron aktivitása  
        - $y_j$: posztszinaptikus neuron aktivitása  
        - $\\eta$: tanulási ráta  
        - $\\Delta w_{ij}$: a szinaptikus súly módosulása az $i \\to j$ kapcsolaton

        **Alapelve:**  
        „**Neurons that fire together, wire together**” – az együtt aktiválódó neuronok közötti kapcsolat erősödik.

        **Jellemzők:**
        - Egyszerű, biológiailag inspirált szabály  
        - Nincs külső tanári jel (nem felügyelt tanulás)  
        - A tanulás **pozitív visszacsatolást** eredményez

        **Felhasználás:**
        - Minták tanulása és reprezentációja  
        - Alapja a későbbi komplex tanulási szabályoknak (pl. STDP, Oja, BCM)

        **Megjegyzés:**  
        A tiszta Hebbian tanulás instabil lehet – gyakran **normalizációval** egészítik ki (pl. Oja szabály).
        """)

    with st.expander("💡 Insight Learning – Belátásos tanulás szimuláció"):
        st.markdown(r"""
        **Cél:** Egy probléma hirtelen, strukturált megoldásának megtalálása kísérleti vagy vizuális minták alapján.

        **Elméleti háttér:**
        - Az **insight** nem fokozatos tanuláson alapul, hanem hirtelen megértésen.
        - Nem szükséges folyamatos megerősítés vagy hibajel – gyakran **strukturális reprezentációk átszervezéséből** születik a megoldás.
        - A viselkedés ugrásszerűen javul, nem fokozatosan.

        **Analógia – problématér átrendezése:**
        $$ \text{Megértés: } \quad S_0 \xrightarrow{\text{transzformáció}} S^* $$
        ahol:
        - $S_0$: kezdeti mentális reprezentáció  
        - $S^*$: új struktúra, amely lehetővé teszi a megoldást

        **Felhasználás:**
        - Viselkedéspszichológia (Köhler, Gestalt-pszichológia)  
        - Gépi tanulásban: stratégiai újratervezés, problémamegoldás mintázatból  
        - Robotikában: **hirtelen útvonaltervezés** vizuális minták alapján

        **Példa:** Egy algoritmus nem a próbálkozások számával tanul, hanem a bemenet struktúrájának elemzésével képes felismerni az optimális lépést.

        """)

    with st.expander("🌪️ Lorenz-rendszer – MLP predikció"):
        st.markdown("""
        **Cél:** A Lorenz-rendszer egyik komponensének (pl. \( x(t) \)) előrejelzése **többrétegű perceptron (MLP)** segítségével, kizárólag múltbeli adatok alapján.
        """)

        st.markdown("**Lorenz-egyenletek:**")
        st.latex(r"\frac{dx}{dt} = \sigma(y - x)")
        st.latex(r"\frac{dy}{dt} = x(\rho - z) - y")
        st.latex(r"\frac{dz}{dt} = xy - \beta z")
        st.markdown(r"Ahol \( \sigma, \rho, \beta \) a rendszer paraméterei.")

        st.markdown("**MLP célfüggvény:**")
        st.latex(r"\hat{x}_{t+1} = f(x_t, x_{t-1}, \dots, x_{t-w})")

        st.markdown("""
        A modell megtanulja az idősor **nemlineáris dinamikáját** egy csúszó ablakos megközelítéssel.

        **Tanulás:**
        - Bemenet: \( w \) hosszú múltbeli szakasz  
        - Kimenet: a következő időlépés komponense (pl. \( x_{t+1} \))  
        - Loss: átlagos négyzetes hiba (MSE)

        **Teljesítménymutatók:**
        - \( R^2 \): a predikció magyarázati ereje  
        - MSE: a hibák átlagos négyzete

        **Felhasználás:**
        - Determinisztikus káosz előrejelzése  
        - Idősoros predikció és nemlineáris rendszerek elemzése  
        - Gépi tanulási modellek tesztelése ismeretlen dinamikán
        """)
        
    with st.expander("🧠 Hebbian Learning Viz – Szinaptikus erősödés szemléltetése"):
        st.latex(r"\Delta w = \eta \cdot x \cdot y")
        st.latex(r"x: \text{ bemeneti neuron aktivitása}")
        st.latex(r"y: \text{ kimeneti neuron aktivitása}")
        st.latex(r"\eta: \text{ tanulási ráta}")
        st.latex(r"\Delta w: \text{ szinaptikus súlyváltozás}")

        st.markdown("""
        **Cél:** A Hebbian tanulás bemutatása vizuálisan, ahol a bemenet ($x$) és a kimenet ($y$) együttes aktivitása megerősíti a szinaptikus kapcsolatot ($w$).  

        **Magyarázat:**  
        Ez a szabály a híres *"Cells that fire together, wire together"* elvét követi.  
        A modul vizuálisan mutatja be, hogy a gyakori együttes aktiváció hogyan növeli a kapcsolatok erősségét.

        **Alkalmazás:**
        - Neurális adaptációk vizsgálata  
        - Biológiai tanulási folyamatok megértése  
        - Dinamikus hálózati súlymódosulások elemzése

        **Tudományos háttér:**  
        A Hebbian tanulás egy alapvető **unsupervised learning** mechanizmus, amely a korrelált aktivitást preferálja, és a **neurális reprezentációk kialakulását** modellezi.
        """)

    with st.expander("🧠 Oja Learning – Főkomponens tanulása"):
        st.markdown(r"""
        **Cél:** A modell megtanulja a bemenet legfontosabb irányát, azaz a **főkomponenst** (PCA-hasonló tanulás).

        **Tanulási szabály:**
        """)
        st.latex(r"\Delta \mathbf{w} = \eta \cdot y \cdot (\mathbf{x} - y \cdot \mathbf{w})")
        st.markdown(r"""
        Ahol:  
        - $\mathbf{x}$: bemeneti vektor  
        - $\mathbf{w}$: súlyvektor  
        - $y = \mathbf{w}^T \mathbf{x}$: neuron kimenete  
        - $\eta$: tanulási ráta  
        - $\Delta \mathbf{w}$: súlyváltozás

        **Mechanizmus:**  
        Az Oja-szabály a **Hebbian tanulást** egészíti ki egy normalizáló taggal, így stabilizálja a súlyok növekedését.

        **Viselkedés:**
        - A súlyvektor konvergál a legnagyobb sajátértékhez tartozó sajátvektor irányába  
        - Hasonlóan viselkedik, mint a PCA első komponense

        **Felhasználás:**  
        - Dimenziócsökkentés neurális úton  
        - Főkomponens detekció tanulás útján  
        - Nem felügyelt tanulási folyamatok modellezése  
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

    with st.expander("⏱️ STDP – Spike-Timing Dependent Plasticity"):
        st.markdown("**Cél:** A szinaptikus súlyok időzítésalapú módosítása – biológiailag inspirált tanulási szabály.")

        st.markdown("**Tanulási szabály:**")
        st.latex(r"""
        \Delta w(\Delta t) =
        \begin{cases}
        A_+ \cdot e^{-\Delta t / \tau_+}, & \text{ha } \Delta t > 0 \ (\text{LTP}) \\\\
        -A_- \cdot e^{\Delta t / \tau_-}, & \text{ha } \Delta t < 0 \ (\text{LTD})
        \end{cases}
        """)

        st.markdown("**Ahol:**")
        st.markdown("""
        - \( \Delta t = t_{\text{post}} - t_{\text{pre}} \): a poszt- és preszinaptikus spike-ok közötti időeltolódás  
        - \( A_+, A_- \): a súlyváltozás maximális amplitúdói  
        - \( \tau_+, \tau_- \): időállandók LTP-re és LTD-re külön-külön  
        - \( \Delta w \): a szinaptikus súlyváltozás
        """)

        st.markdown("**Magyarázat:**")
        st.markdown("""
        - Ha a **preszinaptikus** neuron tüzel *a posztszinaptikus előtt* ( \( \Delta t > 0 \) ), akkor **Long-Term Potentiation** (LTP) történik → erősödik a kapcsolat  
        - Ha a **preszinaptikus** neuron később tüzel ( \( \Delta t < 0 \) ), akkor **Long-Term Depression** (LTD) történik → gyengül a kapcsolat  
        - A változás mértéke exponenciálisan csökken az időeltolódás nagyságával
        """)

        st.markdown("**Alkalmazás:**")
        st.markdown("""
        - Időzítésalapú mintafelismerés tanulása  
        - Biológiailag hiteles neurális modellek fejlesztése  
        - Nem felügyelt tanulás ideghálózatokban  
        - Hebbian-elv továbbfejlesztett, időzített változata
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
