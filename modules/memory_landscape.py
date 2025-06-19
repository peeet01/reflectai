import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def generate_energy_landscape(N=100, seed=42):
    np.random.seed(seed)
    x = np.linspace(-2, 2, N)
    y = np.linspace(-2, 2, N)
    X, Y = np.meshgrid(x, y)
    Z = np.cos(3 * X) * np.sin(3 * Y) + 0.2 * np.random.randn(*X.shape)
    return X, Y, Z

def run():
    st.title("🧠 Memóriakonfigurációs Tájkép")
    st.markdown("Ez a modul a memóriarendszerek energia- vagy stabilitási tájképét vizualizálja.")

    N = st.slider("Pontsűrűség (N×N)", 50, 300, 100, step=10)
    seed = st.number_input("Véletlenszerűség magja (seed)", value=42)

    X, Y, Z = generate_energy_landscape(N=N, seed=seed)

    fig, ax = plt.subplots()
    c = ax.contourf(X, Y, Z, cmap='viridis')
    fig.colorbar(c, ax=ax)
    ax.set_title("Memóriatájkép (szintvonalas megjelenítés)")
    st.pyplot(fig)