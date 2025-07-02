import streamlit as st

def run(): st.title("❓ Súgó és Dokumentáció – Neurolab AI") st.markdown(""" Üdvözlünk a Neurolab AI alkalmazásban!
Ez a sandbox környezet lehetőséget ad különféle idegtudományi, hálózati és tanulási modellek vizsgálatára.
A cél a felfedezés, szimuláció és mélyebb megértés komplex rendszerek dinamikájáról.

A modulok tematikus csoportokba vannak rendezve:
- 🧠 **Tanulási algoritmusok**
- 📈 **Vizualizációk**
- ⚗️ **Szimulációk és dinamikák**
- 🧪 **Adatfeltöltés és predikciók**
- 📚 **Segédmodulok**

Minden modul célja, hogy **interaktív, tudományosan megalapozott** környezetet biztosítson kutatáshoz vagy oktatáshoz.
""")

st.markdown("---")
st.header("🧠 Tanulási algoritmusok")

st.subheader("Hebbian Learning")
st.latex(r"\Delta w_{ij} = \eta x_i x_j")
st.markdown("""
Az idegi tanulás klasszikus szabálya. A szinapszis erősödik, ha az előtte és utána lévő neuron egyszerre aktiválódik.
Alkalmazás: **Hebbian Learning Viz** vizualizációja mutatja be az időbeli tanulást.
""")

st.subheader("Insight Learning")
st.markdown("""
Problémamegoldás hirtelen felismeréssel – nem folyamatos megerősítés, hanem belátás. 
A modul az **emlékezeti állapotok ugrásszerű átrendeződését** demonstrálja.
""")

st.subheader("XOR Prediction & MLP Predict Lorenz")
st.latex(r"\hat{y} = \sigma(W^{(2)} \cdot \sigma(W^{(1)} x + b^{(1)}) + b^{(2)})")
st.markdown("""
Klasszikus nemlineáris osztályozási probléma, több rétegű perceptronnal megoldható. 
A Lorenz-modell predikciója időbeli sorozatokon alkalmazható.
""")

st.markdown("---")
st.header("⚗️ Szimulációk és dinamikák")

st.subheader("Kuramoto modellek")
st.latex(r"\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^{N} A_{ij} \sin(\theta_j - \theta_i)")
st.markdown("""
A **szinkronizációs dinamika** vizsgálatára szolgál. Különböző változatokban:
- **Kuramoto Sim**: alapmodell
- **Kuramoto Hebbian Sim**: tanulási szabályokkal
- **Generative Kuramoto**: gráfstruktúra generálása oszcillátorokhoz
""")

st.subheader("Lorenz rendszer")
st.latex(r"\begin{cases} \dot{x} = \sigma(y - x) \\ \dot{y} = x(\rho - z) - y \\ \dot{z} = xy - \beta z \end{cases}")
st.markdown("""
Kaotikus rendszer, amely szenzitív a kezdeti feltételekre. 
A **Lorenz Sim** modul vizualizálja a trajektóriákat.
""")

st.subheader("Plasticity Dynamics & Noise Robustness")
st.markdown("""
A tanulási szabályok és a zaj hatása az idegrendszeri hálókra. 
Használható a tanulás stabilitásának és robusztusságának tesztelésére.
""")

st.markdown("---")
st.header("📈 Vizualizációk")

st.subheader("Fractal Dimension & Explorer")
st.markdown("""
A fraktáldimenzió numerikus becslése és a Mandelbrot-halmaz iteratív generálása.
""")

st.subheader("Lyapunov Spectrum")
st.markdown("""
A rendszer érzékenysége a kezdeti feltételekre. 
A pozitív legnagyobb Lyapunov-exponens kaotikus viselkedésre utal.
""")

st.subheader("Persistent Homology")
st.markdown("""
Topológiai adatértelmezés. A topológiai jellemzők perzisztenciájának megfigyelése különböző skálákon.
""")

st.subheader("Memory Landscape")
st.markdown("""
A Hopfield-háló energia tájképe. Milyen stabil állapotok alakulnak ki?
""")

st.markdown("---")
st.header("🧪 Adatfeltöltés és predikciók")

st.subheader("Data Upload & ESN Prediction")
st.markdown("""
Betölthető saját CSV-adat, és Echo State Network-re (ESN) alapozott predikció végezhető.
""")

st.subheader("Berry Curvature")
st.latex(r"\Omega(\mathbf{k}) = \nabla_{\mathbf{k}} \times \mathbf{A}(\mathbf{k})")
st.markdown("""
A Berry-görbület a kvantummechanikai hullámfüggvény geometriai fázisát mutatja meg.
Fontos a topológiai szigetelők megértésében.
""")

st.subheader("Neural Entropy")
st.markdown("""
Egyensúly és entrópia vizsgálata neurális dinamika során.
Az információ mennyisége és a rendezettség közötti kapcsolat feltérképezése.
""")

st.markdown("---")
st.header("📚 Segédmodulok")

st.subheader("Graph Sync Analysis")
st.markdown("""
A gráf topológiája és a szinkronizációs viselkedés közötti összefüggést elemzi.
""")

st.subheader("Reflection Modul")
st.markdown("""
Szabad megfigyelések, hipotézisek rögzítésére. 
Tudományos naplóként is használható a saját eredményekhez.
""")

st.subheader("Questions")
st.markdown("""
Ötletelésre, saját kérdések és problémafelvetések gyűjtésére szolgál.
""")

st.subheader("Help")
st.markdown("""
Ez az aktuális oldal. Részletes leírás az összes modul tudományos hátteréről.
""")

st.markdown("---")
st.header("🧪 Bizonyítási ötletek és kutatási célok")
st.markdown("""
- A Kuramoto-modellek gráfelméleti interpretációi
- Hebbián tanulás stabilitása dinamikus gráfokon
- Topológiai invariánsok szerepe kvantumrendszerekben
- Fraktálhatárok és Lyapunov-spektrum kapcsolat
- Az entrópia, memória és generalizáció viszonya
""")

st.markdown("---")
st.header("🧠 Ajánlás a használathoz")
st.markdown("""
- Indulj a **Kezdőlapon**, majd válassz egy modult
- Tanulmányozd a képleteket, figyeld meg a szimuláció viselkedését
- Vezesd saját észrevételeidet a **Reflection Modulban**
- Kísérletezz új paraméterekkel és konfigurációkkal
""")

st.markdown("---")
st.markdown("""
Verzió: **2025.07.02**  
Készítette: *Kovacs Peter*
""")

ReflectAI belépési pont

app = run

