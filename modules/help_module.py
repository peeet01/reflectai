import streamlit as st

def run(): st.title("📘 Tudományos Súgó – Neurolab AI") st.markdown(""" Üdvözlünk a Neurolab AI Scientific Playground felületen! Ez az alkalmazás interaktív vizsgálatot nyújt idegtudományi, fizikai és matematikai modellekre alapozva.

Alább részletesen bemutatjuk a modulok tudományos hátterét, történetét, matematikai validációját, valamint az alkalmazásban betöltött szerepét és felhasználási lehetőségeit.
""")

######################
# 📈 Vizualizációk
######################
with st.expander("🧮 Fractal Dimension"):
    st.markdown("""
    **Történeti háttér:** A fraktálok elméletét Benoît Mandelbrot vezette be a 20. század közepén. A fraktáldimenzió a struktúrák önhasonlóságának kvantitatív mértéke.

    **Matematikai definíció:** Box-counting dimenzió:
    """)
    st.latex(r"D = \lim_{\epsilon \to 0} \frac{\log N(\epsilon)}{\log(1/\epsilon)}")
    st.markdown("""
    ahol N(\epsilon) az \epsilon méretű dobozok száma, amely lefedi az alakzatot.

    **Célja az alkalmazásban:** Mintázatok komplexitásának mérése (pl. neurális kimeneteknél).

    **Következtetések:**
    - Neurális dinamika rendezettségének becslése
    - Rendszerek dimenziós komplexitása
    - Minták jellemzése nemlineáris rendszerekben
    """)

with st.expander("🌀 Fractal Explorer"):
    st.markdown("""
    **Történet:** Mandelbrot- és Julia-halmaz vizsgálata a komplex síkban.

    **Iteráció:**
    """)
    st.latex(r"z_{n+1} = z_n^2 + c")
    st.markdown("""
    **Cél:** Kaotikus és stabil zónák felfedezése.

    **Következtetések:**
    - Stabilitási és bifurkációs analízis
    - Vizuális reprezentáció komplex dinamikákról
    """)

######################
# 🧠 Tanulási algoritmusok
######################
with st.expander("🧠 Hebbian Learning"):
    st.markdown("""
    **Történet:** Donald Hebb (1949) fogalmazta meg. "Neurons that fire together, wire together."

    **Szabály:**
    """)
    st.latex(r"w_i \leftarrow w_i + \eta \cdot x_i \cdot t")
    st.markdown("""
    **Cél:** Nem felügyelt tanulás modellezése.

    **Következtetések:**
    - Szinkron aktivációs minták tanulása
    - Biológiai plaszticitás modellezése
    """)

with st.expander("❌ XOR Prediction"):
    st.markdown("""
    **Történet:** Az XOR-probléma megoldhatatlansága lineáris modellekkel motiválta a mélytanulás fejlődését.

    **Modell:**
    """)
    st.latex(r"\hat{y} = \sigma\left(W^{(2)} \cdot \sigma(W^{(1)}x + b^{(1)}) + b^{(2)}\right)")
    st.markdown("""
    **Cél:** A nemlineáris szeparálhatóság szemléltetése.

    **Következtetések:**
    - Többrétegű hálók szükségessége
    - Nemlinearitás szerepe a tanulásban
    """)

######################
# ⚗️ Szimulációk és dinamikák
######################
with st.expander("🕸️ Kuramoto Sim"):
    st.markdown("""
    **Történet:** Yoshiki Kuramoto dolgozta ki az 1970-es években.

    **Egyenlet:**
    """)
    st.latex(r"\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^{N} \sin(\theta_j - \theta_i)")
    st.markdown("""
    **Cél:** Oszcillátorhálózatok szinkronizációjának vizsgálata.

    **Következtetések:**
    - Szinkron állapot detektálása
    - Szinkronküszöb meghatározása
    """)

######################
# 🧪 Predikciók
######################
with st.expander("🔄 ESN Prediction"):
    st.markdown("""
    **Történet:** Jaeger (2001) mutatta be az Echo State Network modellt.

    **Rekurzió:**
    """)
    st.latex(r"x(t+1) = \tanh(W_{res} x(t) + W_{in} u(t))")
    st.latex(r"\hat{y}(t) = W_{out} x(t)")
    st.markdown("""
    **Cél:** Idősoros előrejelzés tanítás nélkül a rejtett rétegre.

    **Következtetések:**
    - Dinamikus mintázatok felismerése
    - Hatékony idősoros predikció
    """)

with st.expander("🌐 Berry Curvature"):
    st.markdown("""
    **Történet:** Michael Berry (1984) vezette be a geometriai fázis fogalmát kvantummechanikában.

    **Formula:**
    """)
    st.latex(r"\Omega(k) = \nabla_k \times A(k), \quad A(k) = -i \langle u(k) | \nabla_k | u(k) \rangle")
    st.markdown("""
    **Cél:** Topológiai fázisinvariánsok számítása kvantált rendszerekben.

    **Következtetések:**
    - Topológiai különbségek azonosítása
    - Kvantált Chern-számok számítása
    """)

st.markdown("""


---

Verzió: 2025.07
Készítette: ReflectAI fejlesztői és tudományos tanácsadók
""")

Kötelező modul belépési pont

app = run

