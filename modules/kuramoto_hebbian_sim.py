import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.subheader("Kuramoto-Hebbian szimuláció")
    st.write("Kuramoto-Hebbian szimuláció modul fut.")

    N = 10
    T = 10
    dt = 0.05
    steps = int(T / dt)
    theta = np.random.uniform(0, 2*np.pi, N)
    omega = np.random.normal(1.0, 0.1, N)
    weights = np.ones((N, N)) / N

    results = np.zeros((steps, N))
    for t in range(steps):
        results[t] = theta
        for i in range(N):
            influence = np.sum(weights[i] * np.sin(theta - theta[i]))
            theta[i] += dt * (omega[i] + influence)

        # Hebbian súlyfrissítés (egyszerűsített):
        delta_w = np.outer(np.sin(theta), np.sin(theta))
        weights += 0.01 * delta_w
        weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)

    st.line_chart(results)
