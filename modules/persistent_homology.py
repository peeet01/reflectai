import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from ripser import ripser
from persim import plot_diagrams

def run():
    st.header("🔷 Perzisztens homológia")
    st.write("Vizualizáció Ripser és persim csomagokkal szintetikus adatokon.")

    dataset = st.selectbox("Adatkészlet", ["Két félhold", "Véletlen pontok", "Körök"])
    n_samples = st.slider("Minták száma", 20, 1000, 300)

    if dataset == "Két félhold":
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
    st.pyplot(fig1)

    st.subheader("📊 Perzisztencia diagram")
    result = ripser(X)['dgms']
    fig2, ax2 = plt.subplots()
    plot_diagrams(result, ax=ax2)
    st.pyplot(fig2)

    st.info("A H0 komponensek a kapcsolódó klasztereket, a H1 komponensek a ciklusokat reprezentálják.")
