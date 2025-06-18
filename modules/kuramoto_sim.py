
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.subheader("Kuramoto szinkronizáció")
    N, steps = 10, 200
    theta = np.random.rand(N) * 2 * np.pi
    omega = np.random.rand(N)
    coupling = 0.6
    history = [theta.copy()]
    for _ in range(steps):
        for i in range(N):
            interaction = np.sum(np.sin(theta - theta[i]))
            theta[i] += omega[i] + (coupling / N) * interaction
        history.append(theta.copy())
    history = np.array(history)
    fig, ax = plt.subplots()
    for i in range(N):
        ax.plot(history[:, i])
    st.pyplot(fig)
