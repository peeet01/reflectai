
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.subheader("📉 Lyapunov-spektrum kalkulátor")
    st.write("Nemlineáris oszcillátorháló stabilitási analízise – legnagyobb Lyapunov-exponens becslése.")

    N = st.slider("Oszcillátorok száma", 2, 50, 10)
    steps = st.slider("Lépések száma", 100, 2000, 1000)
    dt = st.slider("Időlépés", 0.001, 0.1, 0.01)
    K = st.slider("Kapcsolási erősség", 0.1, 2.0, 1.0)

    np.random.seed(42)
    theta = np.random.rand(N) * 2 * np.pi
    omega = np.random.normal(1.0, 0.1, N)
    delta = np.random.normal(0, 1e-8, N)  # kis perturbáció

    le_history = []

    for _ in range(steps):
        theta_dot = omega + (K / N) * np.sum(np.sin(theta[:, None] - theta), axis=1)
        theta_next = theta + dt * theta_dot

        perturbed_dot = omega + (K / N) * np.sum(np.sin((theta + delta)[:, None] - (theta + delta)), axis=1)
        theta_perturbed_next = theta + delta + dt * perturbed_dot

        delta_new = theta_perturbed_next - theta_next
        le = np.log(np.linalg.norm(delta_new) / np.linalg.norm(delta))
        le_history.append(le)

        theta = theta_next
        delta = delta_new / np.linalg.norm(delta_new) * 1e-8

    mean_le = np.mean(le_history[int(steps/2):])

    st.success(f"Becsült legnagyobb Lyapunov-exponens: {mean_le:.5f}")

    fig, ax = plt.subplots()
    ax.plot(le_history, alpha=0.7)
    ax.axhline(mean_le, color='red', linestyle='--', label="Átlag")
    ax.set_title("Instantán Lyapunov-exponensek")
    ax.set_xlabel("Lépés")
    ax.set_ylabel("LE")
    ax.legend()
    st.pyplot(fig)
