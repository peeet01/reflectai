import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

def run():
    st.subheader("💧 Kuramoto–ESN predikció")
    st.write("Egyszerű Echo State Network (ESN) predikció Kuramoto adatokon.")

    N = 5
    steps = 300
    dt = 0.05
    K = 1.0

    theta = np.random.rand(N) * 2 * np.pi
    omega = np.random.normal(1.0, 0.1, N)
    history = [theta.copy()]

    for _ in range(steps):
        dtheta = omega + (K / N) * np.sum(np.sin(theta[:, None] - theta), axis=1)
        theta += dt * dtheta
        history.append(theta.copy())

    data = np.array(history)
    scaler = MinMaxScaler()
    signal = scaler.fit_transform(data)[:, 0]

    input_dim = 1
    reservoir_size = 100
    leaking_rate = 0.3

    np.random.seed(42)
    Win = np.random.rand(reservoir_size, input_dim) - 0.5
    W = np.random.rand(reservoir_size, reservoir_size) - 0.5
    rho_W = max(abs(np.linalg.eigvals(W)))
    W *= 0.95 / rho_W

    X = []
    state = np.zeros((reservoir_size,))

    for t in range(len(signal)-1):
        u = np.array([signal[t]])
        state = (1 - leaking_rate) * state + leaking_rate * np.tanh(np.dot(Win, u) + np.dot(W, state))
        X.append(state.copy())

    X = np.array(X)
    Y = signal[1:len(signal)]

    model = Ridge(alpha=1e-6)
    model.fit(X, Y)
    Y_pred = model.predict(X)

    fig, ax = plt.subplots()
    ax.plot(Y, label='Valódi')
    ax.plot(Y_pred, label='ESN predikció')
    ax.set_title("Kuramoto–ESN predikció")
    ax.legend()
    st.pyplot(fig)
