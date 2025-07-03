import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from ripser import ripser
from persim import plot_diagrams

def run():
    st.title("🔷 Perzisztens homológia – Topológiai adatelemzés")

    st.markdown("""
A **perzisztens homológia** a topológiai adatfeldolgozás egyik módszere, amely  
a geometriai struktúrák stabilitását vizsgálja különböző skálák mentén.  
Ez az eszköz képes kimutatni klasztereket, ciklusokat és más rejtett topológiai jegyeket a mintákból.

A következő példákban szintetikus adatokat elemzünk a **Ripser** és **Persim** könyvtárak segítségével.
    """)

    dataset = st.selectbox("🧩 Adatkészlet", ["Két félhold", "Véletlen pontok", "Körök"])
    n_samples = st.slider("📊 Minták száma", 20, 1000, 300)

    if dataset == "Két félhold":
        from sklearn.datasets import make_moons
        X, _ = make_moons(n_samples=n_samples, noise=0.05)
    elif dataset == "Véletlen pontok":
        X = np.random.rand(n_samples, 2)
    elif dataset == "Körök":
        t = np.linspace(0, 2 * np.pi, n_samples)
        r = 1 + 0.1 * np.random.randn(n_samples)
        X = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)

    st.subheader("🔘 Pontfelhő")
    fig1, ax1 = plt.subplots()
    ax1.scatter(X[:, 0], X[:, 1], s=10)
    ax1.set_aspect("equal")
    ax1.set_title("Adatpontok")
    st.pyplot(fig1)

    st.subheader("📊 Perzisztencia diagram")
    result = ripser(X)['dgms']
    fig2, ax2 = plt.subplots()
    plot_diagrams(result, ax=ax2)
    st.pyplot(fig2)

    st.markdown("### 📚 Tudományos háttér")
    st.markdown("""
A **perzisztens homológia** a **topológiai adatanalízis (TDA)** része,  
amely az adatokban rejlő alakzatokat és mintázatokat elemzi különböző skálákon.  
Ez lehetővé teszi, hogy a zajtól eltekintve a valóban tartós geometriai jellemzők megmaradjanak.

#### ℹ️ Fogalmak:
- **H₀ komponensek** – diszjunkt klaszterek száma
- **H₁ komponensek** – ciklusok (pl. kör, lyuk) jelenléte
- **Perzisztencia** – az az intervallum, amíg egy topológiai jegy létezik

#### 📈 Diagram:
- Az X tengely a megjelenés skáláját,  
- Az Y tengely az eltűnés skáláját mutatja.
- A főátlótól való távolság a jellemző "fontosságát" jelzi.

#### 🧠 Alkalmazási területek:
- Képfeldolgozás és alakfelismerés
- Idősorok elemzése
- Neurális adatok topológiai elemzése
- Adatok strukturális összehasonlítása

A topológiai jellemzők "élettartamán" keresztül stabil és jelentős mintázatok emelhetők ki,  
amelyek gépi tanulási modellek számára robusztus bemenetként szolgálhatnak.
    """)

# Kötelező ReflectAI-kompatibilitás
app = run
