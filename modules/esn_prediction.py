import streamlit as st
import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

def generate_lorenz_data(n_points=2000, dt=0.01, sigma=10.0, beta=8/3, rho=28.0):
    def lorenz_step(x, y, z):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return dx, dy, dz

    x, y, z = 1.0, 1.0, 1.0
    data = []
    for _ in range(n_points):
        dx, dy, dz = lorenz_step(x, y, z)
        x += dx * dt
        y += dy * dt
        z += dz * dt
        data.append([x, y, z])
    return np.array(data)

def run():
    st.title("ESN Lorenz-pálya predikció")

    n_points = st.slider("Adatpontok száma", 1000, 5000, 2000, step=100)
    washout = st.slider("Washout szakasz hossza", 50, 500, 100, step=10)
    res_size = st.slider("Reservoir méret", 50, 500, 200, step=10)
    spectral_radius = st.slider("Spektrálsugár", 0.1, 2.0, 0.9, step=0.1)

    data = generate_lorenz_data(n_points)
    input_data = data[:-1]
    target_data = data[1:]

    np.random.seed(42)
    Win = (np.random.rand(res_size, 3) - 0.5) * 1.0
    W = np.random.rand(res_size, res_size) - 0.5
    radius = np.max(np.abs(np.linalg.eigvals(W)))
    W *= spectral_radius / radius

    states = np.zeros((n_points - 1, res_size))
    x = np.zeros(res_size)

    for t in range(n_points - 1):
        u = input_data[t]
        x = np.tanh(np.dot(Win, u) + np.dot(W, x))
        states[t] = x

    model = Ridge(alpha=1e-6)
    model.fit(states[washout:], target_data[washout:])

    predictions = model.predict(states)

    fig, ax = plt.subplots()
    ax.plot(target_data[:, 0], label="Valódi", linewidth=1.5)
    ax.plot(predictions[:, 0], label="ESN predikció", linestyle="--")
    ax.set_title("X komponens predikciója")
    ax.legend()
    st.pyplot(fig)
