import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.subheader("Kuramoto szimuláció")
    st.write("Kuramoto szimuláció modul fut.")

    N = 10  # oszcillátorok száma
    K = 1.0  # csatolási erősség
    T = 10   # szimuláció ideje
    dt = 0.05
    steps = int(T / dt)

    theta = np.random.uniform(0, 2*np.pi, N)
    omega = np.random.normal(1.0, 0.1, N)

    results = np.zeros((steps, N))
    for t in range(steps):
        results[t] = theta
        for i in range(N):
            interaction = np.sum(np.sin(theta - theta[i]))
            theta[i] += dt * (omega[i] + (K / N) * interaction)

    st.line_chart(results)
