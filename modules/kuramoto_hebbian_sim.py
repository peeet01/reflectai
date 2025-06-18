import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run_kuramoto_hebbian():
    st.subheader("🔁 Kuramoto–Hebbian háló")
    st.write("Adaptív Kuramoto-modell Hebbian tanulással.")

    N = 10
    T = 10
    dt = 0.05
    steps = int(T / dt)

    theta = np.random.uniform(0, 2 * np.pi, N)
    omega = np.random.normal(1.0, 0.1, N)
    weights = np.ones((N, N)) / N

    results = np.zeros((steps, N))
    for t in range(steps):
        results[t] = theta
        for i in range(N):
            influence = np.sum(weights[i] * np.sin(theta - theta[i]))
            theta[i] += dt * (omega[i] + influence)

            # Hebbian frissítés
            for j in range(N):
                if i != j:
                    weights[i, j] += 0.01 * np.cos(theta[i] - theta[j])

    fig, ax = plt.subplots()
    ax.plot(results)
    st.pyplot(fig)
