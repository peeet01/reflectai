import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.subheader("🌐 Kuramoto szinkronizáció")
    st.write("Egyszerű Kuramoto-háló fázisszinkronizációs vizsgálata.")

    N = 10
    T = 200
    dt = 0.05
    K = 1.0

    theta = np.random.rand(N) * 2 * np.pi
    omega = np.random.normal(1.0, 0.1, N)
    history = [theta.copy()]

    for _ in range(T):
        dtheta = omega + (K / N) * np.sum(np.sin(theta[:, None] - theta), axis=1)
        theta += dt * dtheta
        history.append(theta.copy())

    history = np.array(history)

    fig, ax = plt.subplots()
    for i in range(N):
        ax.plot(history[:, i], label=f'Oszc. {i}')
    ax.set_title("Fázisok időbeli alakulása")
    ax.set_xlabel("Időlépés")
    ax.set_ylabel("Fázis")
    st.pyplot(fig)
