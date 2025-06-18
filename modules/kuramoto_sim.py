
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.header("🌐 Kuramoto szinkronizáció")

    n = st.slider("Oszcillátorok száma", 3, 50, 10)
    steps = st.slider("Iterációk száma", 100, 1000, 200)
    k = st.slider("Kapcsolási erősség (K)", 0.0, 5.0, 1.0)

    phases = np.random.uniform(0, 2*np.pi, (n,))
    natural_frequencies = np.random.normal(0, 1, n)

    phase_history = np.zeros((steps, n))
    dt = 0.05

    for t in range(steps):
        for i in range(n):
            coupling = np.sum(np.sin(phases - phases[i]))
            phases[i] += dt * (natural_frequencies[i] + k * coupling / n)
        phase_history[t] = phases

    fig, ax = plt.subplots()
    ax.plot(phase_history)
    ax.set_title("Kuramoto fázisszinkronizáció")
    ax.set_xlabel("Időlépések")
    ax.set_ylabel("Fázis (radian)")
    st.pyplot(fig)
    st.success(f"Szinkronizációs idő: {steps} iteráció")
