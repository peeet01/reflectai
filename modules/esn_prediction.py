import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

class EchoStateNetwork:
    def __init__(self, input_size, reservoir_size=100, spectral_radius=0.9, sparsity=0.1, random_state=42):
        np.random.seed(random_state)
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.Win = np.random.uniform(-1, 1, (reservoir_size, input_size))
        self.W = np.random.rand(reservoir_size, reservoir_size) - 0.5
        mask = np.random.rand(*self.W.shape) > sparsity
        self.W[mask] = 0
        eigvals = np.linalg.eigvals(self.W)
        self.W *= spectral_radius / np.max(np.abs(eigvals))
        self.Wout = None

    def _update(self, state, u):
        return np.tanh(np.dot(self.Win, u) + np.dot(self.W, state))

    def fit(self, inputs, outputs, washout=50, ridge_param=1e-6):
        states = np.zeros((len(inputs), self.reservoir_size))
        state = np.zeros(self.reservoir_size)
        for t, u in enumerate(inputs):
            state = self._update(state, u)
            states[t] = state

        extended_states = np.hstack([states, inputs])
        self.Wout = np.dot(np.linalg.pinv(extended_states[washout:]), outputs[washout:])

    def predict(self, inputs):
        states = np.zeros((len(inputs), self.reservoir_size))
        state = np.zeros(self.reservoir_size)
        for t, u in enumerate(inputs):
            state = self._update(state, u)
            states[t] = state
        extended_states = np.hstack([states, inputs])
        return np.dot(extended_states, self.Wout)

def run():
    st.subheader("🔁 Kuramoto–ESN predikció")
    st.write("Kuramoto fázisok predikciója Echo State Network segítségével.")

    N = 1
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

    signal = np.array(history).flatten()
    window = 10
    X, y = [], []
    for i in range(len(signal) - window):
        X.append(signal[i:i+window])
        y.append(signal[i+window])
    X, y = np.array(X), np.array(y)
    X = X.reshape(-1, window)
    y = y.reshape(-1, 1)

    esn = EchoStateNetwork(input_size=window, reservoir_size=200)
    esn.fit(X, y)
    y_pred = esn.predict(X)

    fig, ax = plt.subplots()
    ax.plot(y, label="Valódi")
    ax.plot(y_pred, label="ESN predikció")
    ax.set_title("Kuramoto–ESN predikció")
    ax.legend()
    st.pyplot(fig)

    st.success("Predikció sikeresen lefutott.")
