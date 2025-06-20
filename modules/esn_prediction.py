import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def generate_lorenz_data(T, dt=0.01):
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    x, y, z = 1.0, 1.0, 1.0
    data = []

    for _ in range(T):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
        data.append([x, y, z])

    return np.array(data)


def run():
    st.subheader("🔮 Echo State Network (ESN) predikció – Lorenz attractoron")

    # 🔧 Paraméterek
    T = st.slider("Időlépések száma", 500, 5000, 2000)
    reservoir_size = st.slider("Rezegő neuroncsoport mérete", 50, 1000, 200)
    spectral_radius = st.slider("Spektrális sugár (ρ)", 0.1, 1.5, 0.9)
    noise = st.slider("Zajszint", 0.0, 0.1, 0.001)
    ridge_alpha = st.slider("Ridge regularizációs súly", 0.0001, 1.0, 0.01)

    # 🌀 Lorenz adatok
    data = generate_lorenz_data(T)
    train_len = int(0.7 * T)
    test_len = T - train_len

    # 🎯 Cél: x komponens előrejelzése
    target = data[:, 0]

    # 🧠 ESN inicializálás
    input_size = 1
    reservoir = np.random.rand(reservoir_size, reservoir_size) - 0.5
    # Spektrális normalizálás
    eigvals = np.abs(np.linalg.eigvals(reservoir))
    reservoir *= spectral_radius / (np.max(eigvals) + 1e-10)

    input_weights = np.random.rand(reservoir_size, input_size) - 0.5
    states = np.zeros((T, reservoir_size))
    x = np.zeros(reservoir_size)

    # 🏃 Hajtás
    for t in range(1, T):
        u = target[t-1]
        x = np.tanh(np.dot(reservoir, x) + np.dot(input_weights, [u]))
        x += noise * np.random.randn(reservoir_size)
        states[t] = x

    # 🎓 Tanítás
    model = Ridge(alpha=ridge_alpha)
    model.fit(states[1:train_len], target[1:train_len])

    # 🔮 Predikció
    pred = model.predict(states[train_len:T])
    truth = target[train_len:T]
    rmse = np.sqrt(mean_squared_error(truth, pred))

    st.markdown(f"### 📉 RMSE (gyökös átlagos négyzetes hiba): `{rmse:.5f}`")

    # 📊 Vizualizáció
    fig, ax = plt.subplots()
    ax.plot(range(test_len), truth, label="Valódi", color='black')
    ax.plot(range(test_len), pred, label="Predikció", color='red', linestyle='--')
    ax.set_xlabel("Idő")
    ax.set_ylabel("x komponens")
    ax.set_title("🔁 ESN predikció vs. valós Lorenz adatok")
    ax.legend()
    st.pyplot(fig)
