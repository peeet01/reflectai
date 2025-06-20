import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def kuramoto_sim(N, K, T, noise_level, dt=0.05):
    omega = np.random.normal(0, 1, N)
    theta = np.random.uniform(0, 2*np.pi, N)
    order_params = []

    for _ in range(T):
        theta_matrix = np.subtract.outer(theta, theta)
        coupling = np.sum(np.sin(theta_matrix), axis=1)
        theta += (omega + (K / N) * coupling) * dt
        theta += noise_level * np.random.normal(0, 1, N) * dt
        r = np.abs(np.sum(np.exp(1j * theta)) / N)
        order_params.append(r)

    return order_params

def run():
    st.subheader("🔉 Zajtűrés és Szinkronizáció Vizualizáció (Pro)")

    N = st.slider("Oszcillátorok száma", 10, 100, 50)
    K = st.slider("Kapcsolási erősség (K)", 0.0, 10.0, 2.0)
    T = st.slider("Szimuláció hossza (lépések)", 100, 1000, 500)
    dt = 0.05

    noise_levels = [0.0, 0.2, 0.5]
    colors = ['green', 'orange', 'red']
    labels = ['Zajmentes', 'Közepes zaj', 'Erős zaj']

    st.markdown("A szinkronizációs index (r) alakulása különböző zajszintek mellett:")

    fig, ax = plt.subplots()
    for noise, color, label in zip(noise_levels, colors, labels):
        r_vals = kuramoto_sim(N, K, T, noise, dt)
        ax.plot(r_vals, color=color, label=f"{label} (zaj = {noise})", linewidth=2)

    ax.set_title("📉 Szinkronizációs index alakulása zajfüggvényében")
    ax.set_xlabel("Időlépés")
    ax.set_ylabel("Szinkronizációs index (r)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
