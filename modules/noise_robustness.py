import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.subheader("🔊 Zajtűrés és szinkronizációs robusztusság")
    st.write("A Kuramoto-hálózat viselkedése különböző zajszinteken.")

    N = 20
    steps = 300
    dt = 0.05
    K = 1.0
    D_vals = [0.0, 0.1, 0.2, 0.3, 0.4]
    repetitions = 10

    def kuramoto_sim(D):
        R_vals = []
        for _ in range(repetitions):
            theta = np.random.rand(N) * 2 * np.pi
            omega = np.random.normal(1.0, 0.1, N)
            Rs = []
            for _ in range(steps):
                dtheta = omega + (K / N) * np.sum(np.sin(theta[:, None] - theta), axis=1)
                dtheta += D * np.random.randn(N)  # zaj
                theta += dt * dtheta
                R = np.abs(np.sum(np.exp(1j * theta))) / N
                Rs.append(R)
            R_vals.append(np.mean(Rs[-50:]))
        return np.mean(R_vals), np.std(R_vals)

    results = [kuramoto_sim(D) for D in D_vals]
    means, stds = zip(*results)

    fig, ax = plt.subplots()
    ax.errorbar(D_vals, means, yerr=stds, fmt='-o', capsize=5)
    ax.set_xlabel("Zaj intenzitás (D)")
    ax.set_ylabel("Átlagos koherenciaindex R")
    ax.set_title("Zajtűrési görbe")
    st.pyplot(fig)