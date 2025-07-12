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

    with st.expander("🕸️ Kuramoto–Hebbian Szimuláció – Kollektív tanulás dinamikája"):
        st.markdown("""
        **Cél:** A Kuramoto-oszcillátorok és a Hebbian tanulás kombinált modelljének szimulációja.  
        A kapcsolaterősségek időbeli módosulása az **együttes szinkronizáció** függvénye.

        **Kuramoto-egyenletek a dinamikára:**
        """)
        st.latex(r"\frac{d\theta_i}{dt} = \omega_i + \sum_{j=1}^{N} K_{ij} \sin(\theta_j - \theta_i)")

        st.markdown("**Hebbian tanulási szabály az élsúlyokra:**")
        st.latex(r"\Delta K_{ij} = \eta \cdot \cos(\theta_i - \theta_j)")

        st.markdown("**Ahol:**")
        st.latex(r"\theta_i: \text{ az } i\text{-edik oszcillátor fázisa}")
        st.latex(r"\omega_i: \text{ természetes frekvencia}")
        st.latex(r"K_{ij}: \text{ kapcsolat erőssége } (i \leftrightarrow j)")
        st.latex(r"\eta: \text{ tanulási ráta}")
        st.latex(r"\Delta K_{ij}: \text{ a kapcsolat módosulása a Hebbian szabály szerint}")

        st.markdown("""
        **Magyarázat:**  
        - A tanulási szabály megerősíti a szinkron fázisban lévő párok közötti kapcsolatot.  
        - Az antiszinkron (ellentétes fázisú) párok között a kapcsolat gyengülhet.  
        - Így a hálózat képes adaptívan módosítani a topológiáját az aktivitás alapján.

        **Felhasználás:**  
        - Biológiai és mesterséges hálózatok **önszervező** szinkronizációs mechanizmusainak modellezése  
        - Neurodinamikai **plaszticitás** és **topológiai adaptáció** vizsgálata  
        - Szinkronitás-alapú tanulás szimulációja

        **Megjegyzés:**  
        A modell kapcsolódik az agyi hálózatok azon hipotéziséhez, hogy a **szinkron tűzelés** hosszútávú kapcsolaterősödéshez vezet.
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

    with st.expander("🧠 BCM tanulás – Dinamikus küszöb-alapú tanulás"):
        st.markdown(r"""
        **Cél:** A neuronális aktivitás és tanulás összefüggéseinek modellezése egy dinamikus küszöb segítségével.  
        A BCM-szabály (Bienenstock–Cooper–Munro) leírja, mikor erősödik vagy gyengül egy szinaptikus kapcsolat a kimeneti aktivitás függvényében.

        **BCM-egyenletek:**
        """)
    
        st.latex(r"\frac{dw}{dt} = \eta \cdot x \cdot y \cdot (y - \theta)")
        st.latex(r"\frac{d\theta}{dt} = \frac{1}{\tau} (y^2 - \theta)")

        st.markdown(r"""
        **Paraméterek:**
        - \( x \): bemeneti neuron aktivitása  
        - \( y \): kimeneti neuron aktivitása  
        - \( \eta \): tanulási ráta  
        - \( \theta \): aktivitásalapú küszöb, amely időben tanulódik  
        - \( \tau \): időállandó a küszöb változására

        **Magyarázat:**  
        - Ha \( y > \theta \), akkor a szinapszis **erősödik** (LTP)  
        - Ha \( y < \theta \), akkor a szinapszis **gyengül** (LTD)  
        - A küszöb \( \theta \) maga is tanul az aktivitás négyzetének időbeli átlagaként

        **Alkalmazás:**  
        - Homeosztatikus plaszticitás modellezése  
        - Neuronális stabilitás biztosítása tanulás során  
        - Biológiailag realisztikus tanulási szabályok elemzése
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

    with st.expander("🌪️ Lorenz-rendszer – Determinisztikus káosz szimulációja"):
        st.markdown("""
        **Cél:** A Lorenz-rendszer nemlineáris, kaotikus dinamikájának szemléltetése és vizsgálata.

        **Dinamika:** A Lorenz-egyenletek három változóra ható, nemlineáris differenciálegyenletek:
        """)
        st.latex(r"\frac{dx}{dt} = \sigma(y - x)")
        st.latex(r"\frac{dy}{dt} = x(\rho - z) - y")
        st.latex(r"\frac{dz}{dt} = x y - \beta z")

        st.markdown("**Ahol:**")
        st.latex(r"x, y, z: \text{ a rendszer állapotváltozói}")
        st.latex(r"\sigma: \text{ Prandtl-szám (tipikusan } \sigma = 10 \text{)}")
        st.latex(r"\rho: \text{ Rayleigh-szám (tipikusan } \rho = 28 \text{)}")
        st.latex(r"\beta: \text{ geometriai paraméter (tipikusan } \beta = 8/3 \text{)}")

        st.markdown("""
        **Tulajdonságok:**  
        - A rendszer **érzékeny a kezdeti feltételekre**  
        - Nemlineáris visszacsatolások miatt **determinista káosz** alakul ki  
        - Jellemzője az ún. **Lorenz-attraktor**, amely egy fraktál geometriájú pályatér

        **Tudományos jelentőség:**  
        - A Lorenz-modell eredetileg a **légkör konvekciós áramlásait** írta le  
        - Később a **kaotikus rendszerek ikonikus példájává** vált  
        - Alapvető szerepet játszik a komplex dinamikák és nemlineáris viselkedések megértésében

        **Felhasználás:**  
        - Fizikai rendszerek, pl. klímamodellek, áramlások szimulációja  
        - Gépi tanulási modellek tesztelése kaotikus dinamikán  
        - Nemlineáris predikciós algoritmusok benchmarkolása
        """)

    with st.expander("🔮 ESN Prediction – Echo State Network előrejelzés"):
        st.markdown("""
        **Cél:** Idősoros adatok előrejelzése egy **rezervoár alapú** neurális háló segítségével, minimális tanulással.

        **Dinamikai egyenlet:**  
        A belső állapot frissítése:
        """)
        st.latex(r"\mathbf{x}(t+1) = \tanh(W_{\text{res}} \cdot \mathbf{x}(t) + W_{\text{in}} \cdot \mathbf{u}(t))")

        st.markdown("**Kimenet számítása:**")
        st.latex(r"\hat{y}(t) = W_{\text{out}} \cdot \mathbf{x}(t)")

        st.markdown("**Ahol:**")
        st.latex(r"\mathbf{x}(t): \text{ rezervoár állapota}")
        st.latex(r"\mathbf{u}(t): \text{ bemeneti vektor az idő } t \text{-ben}")
        st.latex(r"W_{\text{res}}: \text{ rezervoár súlymátrix (nem tanulódik)}")
        st.latex(r"W_{\text{in}}: \text{ bemeneti súlyok}")
        st.latex(r"W_{\text{out}}: \text{ kimeneti súlyok (tanulhatók)}")
        st.latex(r"\hat{y}(t): \text{ predikált érték}")

        st.markdown("""
        **Jellemzők:**  
        - Csak a kimeneti réteg tanul  
        - A belső dinamika komplex és **nemlineáris**, de nem kell optimalizálni  
        - Hatékony időbeli minták felismerésére

        **Alkalmazás:**  
        - Idősoros előrejelzés (pl. klíma, gazdaság, neuronjelek)  
        - Dinamikus rendszerek modellezése  
        - Online adaptív tanulás rezervoáralapú hálókkal
        """)

    with st.expander("🧠 Neural Entropy – Információelméleti aktivitásmérés"):
        st.markdown("""
        **Cél:** Az idegi aktivitás entrópiájának mérése, mint a **komplexitás** és **információtartalom** kvantitatív mutatója.

        **Shannon-entrópia definíciója:**
        """)
        st.latex(r"H(X) = - \sum_{i} p(x_i) \log_2 p(x_i)")

        st.markdown("""
        Ahol:  
        """)
        st.latex(r"X: \text{ diszkrét valószínűségi változó (pl. spike aktivitás)}")
        st.latex(r"p(x_i): \text{ az } x_i \text{ állapot előfordulási valószínűsége}")

        st.markdown("""
        **Magyarázat:**  
        - Magas entrópia → nagy **variabilitás** és **információtartalom**  
        - Alacsony entrópia → **rendezettség**, vagy determinisztikus viselkedés  
        - Használható időbeli vagy térbeli aktivitásminták elemzésére

        **Alkalmazás:**  
        - Neuronális hálózatok komplexitásának vizsgálata  
        - Kritikus állapotok detektálása  
        - EEG/MEG/fMRI adatok információtartalmának becslése  
        - Tanulási folyamatok során bekövetkező entrópiaváltozások követése

        **Kapcsolódó fogalom:**  
        **Differenciális entrópia** folytonos eloszlásokra:
        """)
        st.latex(r"h(X) = - \int p(x) \log p(x) \, dx")

        st.markdown("""
        ahol \( p(x) \) a valószínűségi sűrűségfüggvény.

        **Tudományos jelentőség:**  
        Az entrópia alkalmazása lehetővé teszi az agyi rendszerek **adaptív dinamikájának** és **információfeldolgozó kapacitásának** objektív értékelését.
        """)

    with st.expander("🔐 Information Bottleneck – Releváns reprezentációk kivonása"):
        st.markdown("""
        **Cél:**  
        Az információelmélet alapján olyan **releváns reprezentációk** megtalálása, amelyek a bemenet ($X$) és a célváltozó ($Y$) közötti **információtartalmat** megőrzik, miközben a fölösleges zajt eltávolítják.

        Ez a kompromisszum a **tömörítés** és az **informatív predikció** között zajlik.

        #### 🧠 Alapelv – Információs célfüggvény:
        Az Information Bottleneck formalizmusa a következő optimalizálási problémát definiálja:
        """)
        st.latex(r"\min_{p(t|x)} \ \mathcal{L} = I(X;T) - \beta \cdot I(T;Y)")
        st.markdown("""
        Ahol:  
        - $T$: kódolt (tömörített) reprezentáció  
        - $I(X;T)$: mennyi információ marad meg $X$-ből a reprezentációban  
        - $I(T;Y)$: mennyi releváns információ van $T$-ben $Y$-ra vonatkozóan  
        - $\\beta$: szabályozó paraméter a tömörítés és predikció között

        #### 🔍 Magyarázat:
        - A cél az, hogy $T$ **minél kevesebb információt** tartalmazzon $X$-ről  
        - Ugyanakkor $T$ tartalmazza a $Y$-ra nézve **maximálisan prediktív információt**  
        - A $\\beta$ paraméter egyensúlyt teremt a két cél között (nagy $\\beta$ → több releváns információ)

        #### 📊 Vizuális kimenet:
        A modul 3D térben ábrázolja a kódolt reprezentációkat ($T$), különféle beágyazásokkal:
        - **Raw Grid**: alap térbeli eloszlás  
        - **PCA**: főkomponens-analízis  
        - **t-SNE**: nemlineáris beágyazás a rejtett szerkezetek kiemelésére

        A pontok színkódolása a célt változó (pl. címkék) szerint történik, így jól látható, hogy mennyire szeparálhatóak a reprezentációk.

        #### 📁 Kimenet:
        A 3D-ben megjelenített kódolt reprezentáció exportálható `.csv` formátumban is további elemzéshez.

        #### 🧪 Tudományos jelentőség:
        - A modell **információmegőrző tömörítést** végez, amely elengedhetetlen a hatékony reprezentációk kialakításához  
        - Alapja a **deep variational bottleneck** modelleknek  
        - Fontos eszköz a neurális hálózatok **rejtett rétegeinek értelmezésében**

        #### 📚 Alkalmazások:
        - Mély tanulási modellek belső reprezentációinak vizsgálata  
        - Feature kiválasztás és dimenziócsökkentés  
        - Adattömörítés, rejtett klaszterstruktúrák feltárása  
        - Információelméleti regularizálás neurális hálókban  
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

    with st.expander("🧠 Boltzmann-gép – Energián alapuló generatív tanulás"):
        st.markdown(r"""
        **Cél:** A Boltzmann-gép egy **statisztikai mechanikán** alapuló, generatív neurális hálózat,  
        amely képes tanulni a bemenetek valószínűségi szerkezetét az **energia** és a **hőmérséklet** elvével.

        A modell a **neurális komponensek közötti szimmetrikus súlykapcsolatokkal** dolgozik, és a tanulás célja  
        az energiagörbe minimalizálása a tanuló minták alapján.

        **Boltzmann-energiafüggvény:**
        """)

        st.latex(r"E(v, h) = - \sum_i b_i v_i - \sum_j c_j h_j - \sum_{i,j} v_i W_{ij} h_j")

        st.markdown(r"""
        **Paraméterek:**
        - \( v_i \): látható réteg (bemenet)
        - \( h_j \): rejtett réteg (neurális reprezentációk)
        - \( b_i, c_j \): bias paraméterek a rétegekhez
        - \( W_{ij} \): súlykapcsolatok a két réteg között

        **Magyarázat:**  
        A hálózat célja, hogy minimalizálja az \( E(v, h) \) energiafüggvényt a tanult eloszlás szerint.

        **Tanulási szabály (Contrastive Divergence – CD):**
        """)

        st.latex(r"\Delta W_{ij} = \eta (\langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}})")

        st.markdown(r"""
        - Az első tag a **pozitív fázis** (valós adatokon számolt korreláció)  
        - A második tag a **negatív fázis** (a modell által generált adatokon)  
        - \( \eta \): tanulási ráta
        """)

        st.markdown(r"""
        **Alkalmazás:**  
        - Mintafelismerés és jellemzőtanulás  
        - Generatív modellek (pl. Restricted Boltzmann Machines - RBM)  
        - Dimenziócsökkentés és előfeldolgozás  
        - Gépi látás és jelfeldolgozás

        **Megjegyzés:**  
        A Boltzmann-gépek **nem determinisztikus** modellek, a neurális egységek aktivációja valószínűségi alapon történik:

        """)

        st.latex(r"P(h_j = 1 | v) = \sigma(\sum_i W_{ij} v_i + c_j)")

        st.markdown("""
        Ahol \( \sigma \) a logisztikus aktivációs függvény.

        Ez a modell hidat képez a **fizikai rendszerek** és a **mély tanulási algoritmusok** között.
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

    with st.expander("🧠 Critical Hebbian (3D) – Kritikus tanulási dinamika vizualizációja"):
        st.markdown("""
        **Cél:**  
        A Hebbian tanulási szabály által vezérelt **dinamikus súlymódosulás** interaktív, 3D vizualizációja.  
        A rendszer célja a **kritikus állapot** közelébe való konvergálás – ahol a hálózat **maximális komplexitást és adaptivitást** mutat.

        **Tanulási szabály:**
        """)
        st.latex(r"\Delta W_{ij}(t) = \eta \cdot x_i(t) \cdot x_j(t)")

        st.markdown("""
        **Ahol:**
        """)
        st.latex(r"x_i(t): \text{ az } i\text{-edik neuron aktivációja időben}")
        st.latex(r"\eta: \text{ tanulási ráta}")
        st.latex(r"W_{ij}: \text{ szinaptikus súly a } i \rightarrow j \text{ kapcsolaton}")

        st.markdown("""
        A tanulás során a súlymátrix fejlődik, és a hálózat konfigurációja **önszerveződő módon** alakul ki.

        #### 🔬 Kritikusitás jellemzése:
        A rendszer viselkedése a **kritikus pont** közelében:
        """)
        st.latex(r"P(s) \propto s^{-\tau}")
        st.markdown("""
        Ahol:
        """)
        st.latex(r"s: \text{ aktivitási esemény mérete}, \quad \tau \approx 1.5")
        st.markdown("""
        Ez az eloszlás azt mutatja, hogy a rendszer **skálafüggetlen fluktuációkat** mutat – a kritikus rendszerek jellemzője.

        #### Vizualizáció:
        A súlymátrix időbeli fejlődése különböző **3D beágyazási módszerekkel** tekinthető meg:
        - **Raw Grid** – egyszerű térbeli elhelyezés  
        - **PCA** – főkomponens-alapú vetítés  
        - **t-SNE** – nemlineáris szerkezetek kiemelése

        #### Kimenet:
        A vizualizációhoz tartozó súlymátrix exportálható `.csv` formában.

        #### Tudományos jelentőség:
        - A kritikus dinamikák növelhetik a **rendszer válaszkészségét**  
        - Optimalizálhatják az **információfeldolgozást**  
        - **Önszerveződő komplexitás** jelenik meg

        **Kapcsolódás:** A modul a Hebbian tanulás és a **kritikus fázisátmenetek** egyesítésével vizsgálja a tanulás **nemlineáris viselkedését** – ideális oktatási és kutatási célokra.
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

    with st.expander("🧲 Ising-modell – Fázisátmenet és rendezettség vizsgálata"):
        st.markdown("""
        **Cél:** A klasszikus 2D **Ising-modell** dinamikájának bemutatása hőmérsékletfüggő viselkedéssel és rendezettségi mintázatokkal.
    
        **Rácsmodell:** A modell egy kétdimenziós négyzetrács, ahol minden ponton egy **spin** ($s_{i,j} = \pm 1$) található.

        **Hamilton-függvény:** A rendszer energiáját az alábbi alak határozza meg:
        """)
        st.latex(r"H = -J \sum_{\langle i,j \rangle} s_i s_j")
        st.markdown("**Ahol:**")
        st.latex(r"J > 0 \quad \text{ferromágneses csatolás}")
        st.latex(r"\langle i,j \rangle \quad \text{szomszédos spinpárok}")

        st.markdown("""
        **Mágnesezettség:** A globális rendezettség mértéke:
        """)
        st.latex(r"M = \frac{1}{N^2} \sum_{i,j} s_{i,j}")

        st.markdown("""
        **Energia:** Az átlagos belső energia:
        """)
        st.latex(r"E = -\frac{1}{N^2} \sum_{\langle i,j \rangle} s_i s_j")

        st.markdown("""
        **Sztochasztikus dinamika:**  
        A szimuláció a **Metropolis–Monte Carlo** algoritmust használja a hőmérsékletfüggő konfigurációk generálására.  
        A spinek frissítése a következő szabály szerint történik:
        """)

        st.latex(r"\Delta E = 2 s_{i,j} \sum_{\text{szomszédok}} s_k")
        st.markdown("Az új állapot elfogadási valószínűsége:")
        st.latex(r"P = \min \left( 1, \, e^{-\beta \Delta E} \right)")

        st.markdown("""
        **Fázisátmenet:**  
        Az Ising-modell képes szemléltetni a **másodfajú fázisátmenetet**, ahol a rendszer viselkedése gyökeresen megváltozik a kritikus hőmérséklet környékén:
        """)
        st.latex(r"T_c \approx \frac{2}{\ln(1 + \sqrt{2})} \approx 2.27")
        st.latex(r"\beta_c \approx 0.44")

        st.markdown("""
        **Jellemzők:**  
        - Alacsony hőmérsékleten: rendezettség ($M \ne 0$)  
        - Magas hőmérsékleten: rendezetlenség ($M \approx 0$)  
        - Kritikus pontban: skálafüggetlen fluktuációk
        """)
        st.markdown("""
        **Tudományos jelentőség:**  
        - Egyszerű, de pontos modell **rendezettség** és **kritikusság** vizsgálatára  
        - Sztochasztikus folyamatok, szinkronizáció és neurális mintázatok értelmezésére  
        - Kapcsolat mezőelmélettel, hálózatokkal, döntésdinamikákkal  

        **Felhasználás:**  
        - Komplex rendszerek viselkedésének modellezése  
        - Gépi tanulási hálók inspirálása (pl. Hopfield)  
        - Kvantum Ising-modell kvantuminformációhoz  
        - Véleménydinamika, társadalmi hálók modellezése
        """)

    with st.expander("📶 Noise Robustness – Tanulási rendszerek zajtűrése"):
        st.markdown("""
        **Cél:** Annak vizsgálata, hogy a neurális hálózatok mennyire képesek megőrizni teljesítményüket bemeneti zaj vagy belső fluktuációk jelenlétében.

        **Zajos bemenet modellezése:**
        """)
        st.latex(r"x(t) = x_{\text{clean}}(t) + \xi(t)")
        st.markdown("ahol:")
        st.latex(r"x_{\text{clean}}(t): \text{ a zajmentes bemeneti jel}")
        st.latex(r"\xi(t) \sim \mathcal{N}(0, \sigma^2): \text{ Gauss-zaj nulla várható értékkel és } \sigma^2 \text{ varianciával}")

        st.markdown("**Predikció hibája zajos környezetben:**")
        st.latex(r"\text{MSE} = \frac{1}{T} \sum_{t=1}^{T} (y(t) - \hat{y}(t))^2")

        st.markdown("**Jel–zaj arány (SNR):**")
        st.latex(r"\text{SNR (dB)} = 10 \cdot \log_{10} \left( \frac{\mathbb{E}[x_{\text{clean}}^2]}{\mathbb{E}[\xi^2]} \right)")

        st.markdown("**Zajrobusztus tanulási célfüggvény:**")
        st.latex(r"\min_{W} \ \mathbb{E}_{\xi} \left[ \| f_W(x + \xi) - y \|^2 \right]")

        st.markdown("""
        **Tudományos háttér:**  
        A zajrobusztusság kulcsfontosságú jellemző a **biológiai idegrendszerekben**, ahol a bemenetek és kimenetek mindig tartalmaznak fluktuációt.  
        A mesterséges hálózatokban ezt **regularizációval**, **adataugmentációval** vagy **dropout technikával** lehet fokozni.

        **Alkalmazás:**  
        - Valós idejű predikciós rendszerek stabilitása  
        - Szenzoros feldolgozás hibás vagy hiányos jelek esetén  
        - Biológiai inspirált hálózatok tesztelése természetes zaj környezetben
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

    with st.expander("⚡ Spiking Neural Network – LIF neuron és STDP"):
        st.markdown(r"""
        **Cél:** Egy biológiailag inspirált neuronmodell (**Leaky Integrate-and-Fire**, LIF) szimulációja, amely **STDP** (Spike-Timing Dependent Plasticity) tanulást alkalmaz.

        #### 🧠 LIF neuronmodell:
        """)
        st.latex(r"\tau_m \frac{dV(t)}{dt} = -V(t) + R_m \cdot I_{ext}")
        st.markdown(r"""
        Ahol:
        - \( V(t) \): membránpotenciál  
        - \( R_m \): membránellenállás  
        - \( I_{ext} \): külső áram  
        - \( \tau_m \): időállandó  

        A neuron tüzel, ha \( V(t) \geq V_{th} \), majd a potenciál visszaáll \( V_{reset} \)-re.

        #### 🔁 STDP tanulási szabály:
        """)
        st.latex(r"""
        \Delta w =
        \begin{cases}
        A_+ \cdot e^{-\Delta t / \tau_+}, & \text{ha } \Delta t > 0 \ (\text{LTP}) \\\\
        -A_- \cdot e^{\Delta t / \tau_-}, & \text{ha } \Delta t < 0 \ (\text{LTD})
        \end{cases}
        """)
        st.markdown(r"""
        - \( \Delta t = t_{\text{post}} - t_{\text{pre}} \): spike időzítéskülönbség  
        - \( A_+, A_- \): a súlymódosítás maximumai  
        - \( \tau_+, \tau_- \): időállandók  

        A szabály lehetővé teszi a szinaptikus súlyok **időzítésalapú módosítását** – alapvető a temporális mintázatok tanulásában.

        #### 📊 Felhasználás:
        - Neuromorf architektúrák (Loihi, TrueNorth)
        - Szenzoros időjelek feldolgozása  
        - Időfüggő minták felismerése (beszéd, mozgás)  
        - Energiahatékony gépi tanulás biológiai inspirációval
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
        
    with st.expander("🌐 Berry-görbület – Topológiai jellemzők"):
        st.markdown(r"""
        A Berry-görbület egy kvantumos állapottér geometriai tulajdonságát méri, amelyből topológiai invariánsok (pl. **Chern-szám**) számolhatók.

        ### 🧮 Matematikai háttér

        A Berry-kapcsolat:

        $$
        \mathbf{A}(\mathbf{k}) = i \langle u(\mathbf{k}) | \nabla_{\mathbf{k}} u(\mathbf{k}) \rangle
        $$

        A Berry-görbület a kapcsolat rotációja:

        $$
        \Omega(\mathbf{k}) = \nabla_{\mathbf{k}} \times \mathbf{A}(\mathbf{k})
        $$

        A Berry-fázis zárt pálya mentén integrált geometriai fázis:

        $$
        \gamma = \oint_C \mathbf{A}(\mathbf{k}) \cdot d\mathbf{k} = \int_S \Omega(\mathbf{k}) \, d^2k
        $$

        ### 🔬 Fizikai jelentés
        - A Berry-görbület lokálisan méri a **sávstruktúra topológiai torzulását**
        - A teljes térre integrálva a **Chern-számot** adja, ami:
        - Kvantált vezetőképességhez vezet (pl. Kvantum Hall-effektus)
        - Topológiai szigetelők peremállapotainak számát határozza meg

        ### 📈 Mit csinál ez a modul?

        - **Kontúrplot**: Berry-görbület a Brillouin-zónában
        - **3D vizualizáció**: Felületként ábrázolt görbület
        - **Berry-fázis**: Zárt kör mentén numerikus fázisszámítás
        - **d-vektor vizualizáció**: Állapottér iránygörbe a Bloch-gömbön

        A **delta** paraméter változtatásával a szimuláció topológiai átmeneteket is képes jelezni (pl. ha a Chern-szám 0 → ±1-re vált).

        ### 📥 Kimenet
        - Interaktív 2D és 3D görbületképek
        - Berry-fázis értéke adott kör mentén
        - CSV exportálható görbületadatokkal
        """)

    with st.expander("🔬 Plasticity Dynamics – Szinaptikus plaszticitás dinamikája"):
        st.markdown("""
        **Cél:** A szinaptikus súlyok időbeli változásának modellezése különböző biológiai tanulási szabályok mentén.

        **Általános Hebbian-plaszticitás egyenlete:**
        """)
        st.latex(r"\Delta w_{ij} = \eta \cdot x_i \cdot y_j")

        st.markdown("**Normalizált Hebbian (Oja-szabály):**")
        st.latex(r"\Delta w_{ij} = \eta \cdot y_j \cdot (x_i - y_j \cdot w_{ij})")

        st.markdown("**BCM szabály dinamikus küszöbbel:**")
        st.latex(r"\frac{dw}{dt} = \eta \cdot x \cdot y \cdot (y - \theta)")
        st.latex(r"\frac{d\theta}{dt} = \frac{1}{\tau} (y^2 - \theta)")

        st.markdown("**STDP – időzítésfüggő plaszticitás:**")
        st.latex(r"""
        \Delta w(\Delta t) =
        \begin{cases}
        A_+ \cdot e^{-\Delta t / \tau_+}, & \text{ha } \Delta t > 0 \\
        -A_- \cdot e^{\Delta t / \tau_-}, & \text{ha } \Delta t < 0
        \end{cases}
        """)

        st.markdown("**Magyarázat a változókhoz:**")
        st.latex(r"x_i: \text{ preszinaptikus neuron aktivitása}")
        st.latex(r"y_j: \text{ posztszinaptikus neuron aktivitása}")
        st.latex(r"\eta: \text{ tanulási ráta}")
        st.latex(r"\theta: \text{ aktivitásfüggő tanulási küszöb}")
        st.latex(r"\tau: \text{ időállandó}")
        st.latex(r"\Delta t = t_{\text{post}} - t_{\text{pre}}: \text{ spike időeltérés}")
        st.latex(r"A_+, A_-: \text{ maximális súlyváltozási amplitúdók}")

        st.markdown("""
        **Felhasználás:**  
        - Neurális hálózatok adaptív viselkedésének szimulációja  
        - Tanulás és memóriafolyamatok dinamikus modellezése  
        - Időzítésalapú szabályok biológiai validálása  

        **Tudományos jelentőség:**  
        A szinaptikus plaszticitás a **tanulás sejtbiológiai alapja**, amelynek pontos modellezése lehetővé teszi a **realisztikus neurális hálók** létrehozását.  
        A különböző szabályok eltérő stabilitási és adaptációs viselkedést mutatnak.
        """)

    with st.expander("🧠 Hopfield-háló – Asszociatív memória és dinamikus visszakeresés"):
        st.markdown(r"""
        A **Hopfield-háló** egy visszacsatolt bináris neuronhálózat, mely képes tanult minták **tárolására** és **visszakeresésére** részleges vagy zajos bemenetből.

        ### 🧮 Matematikai háttér

        A tanulási szabály:

        $$
        W_{ij} = \sum_{\mu=1}^{P} \xi_i^\mu \, \xi_j^\mu \quad (\text{Hebbian tanulás})
        $$

        A frissítési dinamika:

        $$
        s_i^{(t+1)} = \mathrm{sign} \left( \sum_j W_{ij} \, s_j^{(t)} \right)
        $$

        - $W_{ij}$: szinaptikus súly mátrix
        - $\xi_i^\mu$: a $\mu$-edik tanult minta $i$-edik komponense
        - $s_i^{(t)}$: az $i$-edik neuron állapota az $t$-edik időlépésben

        ### 📈 Mit csinál a modul?

        - **Minták betanítása** bináris (±1) képként
        - **Zajos bemenetből indulva** rekonstruálja a korábban tárolt mintát
        - **Energiafüggvény** alapján lejt mozgás – konvergens, stabil háló

        ### 🧩 Alkalmazási területek

        - Asszociatív memória (mint az emberi emlékezés modellezése)
        - Mintafelismerés zajos környezetben
        - Neurális dinamika és stabilitás modellezése

        ### ⚙️ Topológia

        - Teljesen összekapcsolt szimmetrikus háló
        - Nincs önkapcsolat: \( W_{ii} = 0 \)

        Az energiafüggvény garantálja, hogy a háló **stabil fixpontokra** konvergál:

        $$
        E = -\frac{1}{2} \sum_{i,j} W_{ij} s_i s_j
        $$

        A Hopfield-háló egy egyszerű, de erőteljes modell a **nemlineáris dinamikus rendszerek** és **memóriastruktúrák** tanulmányozására.
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
