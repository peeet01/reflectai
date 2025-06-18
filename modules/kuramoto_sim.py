
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.subheader("Kuramoto szinkronizáció")
    st.write("Egyszerű Kuramoto szimuláció fut...")
    n = 10
    theta = np.random.rand(n) * 2 * np.pi
    omega = np.random.rand(n)
    coupling = 0.5
    steps = 200
    history = [theta.copy()]
    for _ in range(steps):
        for i in range(n):
            interaction = np.sum(np.sin(theta - theta[i]))
            theta[i] += omega[i] + (coupling / n) * interaction
        history.append(theta.copy())
    history = np.array(history)
    fig, ax = plt.subplots()
    for i in range(n):
        ax.plot(history[:, i])
    st.pyplot(fig)
