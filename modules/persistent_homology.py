import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from ripser import ripser
from persim import plot_diagrams
import plotly.graph_objects as go

def generate_data(dataset, n_samples):
    if dataset == "Két félhold":
        X, _ = make_moons(n_samples=n_samples, noise=0.05)
    elif dataset == "Véletlen pontok":
        X = np.random.rand(n_samples, 2)
    elif dataset == "Körök":
        t = np.linspace(0, 2 * np.pi, n_samples)
        r = 1 + 0.1 * np.random.randn(n_samples)
        X = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    return X

def plot_3d_colored_pointcloud(X):
    z = np.sin(X[:,0]*3) * np.cos(X[:,1]*3)  # szintetikus mélység
    fig = go.Figure(data=[go.Scatter3d(
        x=X[:,0], y=X[:,1], z=z,
        mode='markers',
        marker=dict(size=3, color=z, colorscale='Viridis', opacity=0.8)
    )])
    fig.update_layout(
        title="🌐 3D színes pontfelhő",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z (mesterséges)"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def run():
    st.title("🔷 Perzisztens homológia – Topológiai mintázatok felfedezése")

    st.markdown("""
A **perzisztens homológia** egy algebrai topológiai eszköz, amely különféle méretű mintázatokat (komponenseket, lyukakat)  
képes detektálni adatokban. A következő példák szintetikus adatkészleteken keresztül mutatják meg a módszer erejét.
""")

    dataset = st.selectbox("📊 Adatkészlet kiválasztása", ["Két félhold", "Véletlen pontok", "Körök"])
    n_samples = st.slider("🔢 Minták száma", 20, 1000, 300)

    # Adatok generálása
    X = generate_data(dataset, n_samples)

    st.subheader("🔘 2D Pontfelhő")
    fig1, ax1 = plt.subplots()
    ax1.scatter(X[:, 0], X[:, 1], s=10)
    ax1.set_aspect("equal")
    st.pyplot(fig1)

    st.subheader("🌐 3D Forgatható pontfelhő (színezett)")
    fig3d = plot_3d_colored_pointcloud(X)
    st.plotly_chart(fig3d, use_container_width=True)

    st.subheader("📊 Perzisztencia diagram")
    result = ripser(X)['dgms']
    fig2, ax2 = plt.subplots()
    plot_diagrams(result, ax=ax2)
    st.pyplot(fig2)

    st.markdown("### 📘 Tudományos háttér")
    st.markdown("""
A **perzisztens homológia** segítségével a geometriai és topológiai struktúrákat  
(kapcsolódó komponensek, lyukak) vizsgálhatjuk különböző léptékeken keresztül.

**Jelentés:**
- A **H₀ komponensek** a klasztereket (összefüggő részeket) reprezentálják  
- A **H₁ komponensek** a zárt hurkokat, ciklusokat jelölik  

**Alkalmazások:**
- Mintázatfelismerés zajos adatokban  
- Biológiai, orvosi, fizikai adatelemzés  
- Gépi tanulásban jellemzők kinyerése

**Diagram értelmezése:**
- Minden pont egy topológiai struktúrát jelöl (születés–halál intervallum)
- Minél tovább él, annál robusztusabb a mintázat
""")

# ReflectAI-kompatibilitás
app = run
