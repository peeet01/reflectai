import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.subheader("🔁 Kuramoto–Hebbian hálózat")
    st.write("Szinkronizáció Hebbian frissítéssel kombinálva.")

    n = 6
    timesteps = 150
    coupling = 0.4
    theta = np.random.rand(n) * 2 * np.pi
    omega = np.random.rand(n)
    weights = np.random.rand(n, n)
    history = [theta.copy()]

    for _ in range(timesteps):
        for i in range(n):
            for j in range(n):
                if i != j:
                    weights[i, j] += 0.001 * np.cos(theta[i] - theta[j])
            theta[i] += omega[i] + (coupling / n) * np.sum(weights[i] * np.sin(theta - theta[i]))
        history.append(theta.copy())

    data = np.array(history)
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_title("Kuramoto–Hebbian szinkronizáció")
    st.pyplot(fig)
