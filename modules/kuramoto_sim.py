import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def run():
    st.header("🌐 Kuramoto szinkronizáció")

    # Paraméterek
    N = st.slider("Oszcillátorok száma", min_value=3, max_value=50, value=10)
    T = st.slider("Iterációk száma", min_value=100, max_value=1000, value=200)
    K = st.slider("Kapcsolási erősség (K)", min_value=0.0, max_value=5.0, value=1.0)
    dt = 0.1

    np.random.seed(42)
    theta = np.random.rand(N) * 2 * np.pi  # Kezdeti fázisok
    omega = np.random.normal(0, 1, N)       # Sajátfrekvenciák
    phase_history = []

    for _ in range(T):
        interaction = np.sum(np.sin(np.subtract.outer(theta, theta)), axis=1)
        dtheta = omega + (K / N) * interaction
        theta += dt * dtheta
        phase_history.append(np.copy(theta))

    # Eredmények ábrázolása
    phase_history = np.array(phase_history)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(N):
        ax.plot(phase_history[:, i], label=f"Oszc. {i+1}")
    ax.set_title("Kuramoto fázisszinkronizáció")
    ax.set_xlabel("Időlépések")
    ax.set_ylabel("Fázis (radián)")
    ax.legend(loc="upper right", fontsize="small", ncol=2)

    st.pyplot(fig)
    st.success(f"Szinkronizációs idő: {T} iteráció")
