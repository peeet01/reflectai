import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_data(show_spinner=False)
def kuramoto_sim_fast(N, K, T, noise_level, dt=0.05):
    omega = np.random.normal(0, 1, N)
    theta = np.random.uniform(0, 2*np.pi, N)
    r_vals = np.zeros(T)

    for t in range(T):
        sin_diff = np.sin(theta[:, None] - theta)
        coupling = np.sum(sin_diff, axis=1)
        theta += (omega + (K / N) * coupling) * dt
        theta += noise_level * np.random.normal(0, 1, N) * dt
        r_vals[t] = np.abs(np.sum(np.exp(1j * theta)) / N)

    return r_vals

def run():
    st.subheader("🔉 Zajtűrés és Szinkronizáció Vizualizáció (Gyorsított Pro változat)")

    N = st.slider("Oszcillátorok száma", 10, 100, 50)
    K = st.slider("Kapcsolási erősség (K)", 0.0, 10.0, 2.0)
    T = st.slider("Szimuláció hossza (lépések)", 100, 1000, 300)

    noise_levels = [0.0, 0.2, 0.5]
    colors = ['green', 'orange', 'red']
    labels = ['Zajmentes', 'Közepes zaj', 'Erős zaj']

    fig, ax = plt.subplots()
    for noise, color, label in zip(noise_levels, colors, labels):
        r_vals = kuramoto_sim_fast(N, K, T, noise)
        ax.plot(r_vals, color=color, label=f"{label} (zaj = {noise})", linewidth=2)

    ax.set_title("📉 Szinkronizációs index – Zajhatás vizsgálata")
    ax.set_xlabel("Időlépés")
    ax.set_ylabel("r (szinkronizációs index)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
