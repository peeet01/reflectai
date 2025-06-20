import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def run():
    st.subheader("🔁 Echo State Network (ESN) predikció – Pro verzió")

    # Paraméterek
    T = st.slider("Időlépések száma", 100, 2000, 500)
    reservoir_size = st.slider("Reservoir méret", 50, 500, 100)
    spectral_radius = st.slider("Spektrális sugár (ρ)", 0.1, 1.5, 0.9)
    ridge_alpha = st.slider("Ridge regularizációs paraméter", 0.001, 1.0, 0.1)
    noise = st.slider("Zajszint", 0.0, 0.5, 0.01)

    # Lorenz-rendszer szimuláció (3D cél)
    def lorenz_step(state, dt=0.01, sigma=10.0, rho=28.0, beta=8/3):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return np.array([x + dx * dt, y + dy * dt, z + dz * dt])

    state = np.array([1.0, 1.0, 1.0])
    target = np.zeros((T, 3))
    for t in range(T):
        state = lorenz_step(state)
        target[t] = state

    # ESN inicializálás
    input_size = 3
    output_size = 3
    input_weights = np.random.randn(reservoir_size, input_size) * 0.1
    reservoir = np.random.randn(reservoir_size, reservoir_size)
    eigvals = np.linalg.eigvals(reservoir)
    reservoir *= spectral_radius / np.max(np.abs(eigvals))
    x = np.zeros(reservoir_size)
    states = np.zeros((T, reservoir_size))

    # Progress bar
    progress = st.progress(0)
    status = st.empty()

    for t in range(1, T
