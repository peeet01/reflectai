import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def compute_lyapunov(f, x0, delta=1e-8, steps=1000):
    x = x0
    d = delta
    lyapunov_sum = 0.0

    for _ in range(steps):
        x1 = f(x)
        x2 = f(x + d)

        d = np.abs(x2 - x1)
        d = d if d != 0 else 1e-8  # prevent log(0)
        lyapunov_sum += np.log(np.abs(d / delta))
        x = x1

    return lyapunov_sum / steps

def logistic_map(r):
    return lambda x: r * x * (1 - x)

def run():
    st.title("Lyapunov spektrum")

    r_min = st.slider("r minimum érték", 2.5, 3.5, 2.5)
    r_max = st.slider("r maximum érték", 3.5, 4.0, 4.0)
    n_points = st.slider("Mintavételezési pontok száma", 100, 1000, 500)
    x0 = st.slider("Kezdeti érték (x₀)", 0.0, 1.0, 0.5)

    r_values = np.linspace(r_min, r_max, n_points)
    lyapunov_values = []

    for r in r_values:
        f = logistic_map(r)
        lyap = compute_lyapunov(f, x0)
        lyapunov_values.append(lyap)

    fig, ax = plt.subplots()
    ax.plot(r_values, lyapunov_values, lw=1)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel("r")
    ax.set_ylabel("Lyapunov-exponens")
    ax.set_title("Lyapunov spektrum a logisztikus térképre")
    st.pyplot(fig)
