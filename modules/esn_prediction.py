
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from modules.data_upload import get_uploaded_data, show_data_overview

def generate_lorenz_data(n_points=1000, dt=0.01):
    def lorenz(x, y, z, s=10, r=28, b=8/3):
        dx = s * (y - x)
        dy = r * x - y - x * z
        dz = x * y - b * z
        return dx, dy, dz

    xs = np.empty(n_points)
    ys = np.empty(n_points)
    zs = np.empty(n_points)

    x, y, z = 0., 1., 1.05
    for i in range(n_points):
        dx, dy, dz = lorenz(x, y, z)
        x += dx * dt
        y += dy * dt
        z += dz * dt
        xs[i], ys[i], zs[i] = x, y, z

    return xs, ys, zs

class EchoStateNetwork:
    def __init__(self, n_inputs, n_reservoir, spectral_radius=1.25, sparsity=0.1, noise=0.001):
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.init_weights()

    def init_weights(self):
        self.Win = (np.random.rand(self.n_reservoir, self.n_inputs) - 0.5)
        self.W = np.random.rand(self.n_reservoir, self.n_reservoir) - 0.5
        self.W[np.random.rand(*self.W.shape) < self.sparsity] = 0
        radius = np.max(np.abs(np.linalg.eigvals(self.W)))
        self.W *= self.spectral_radius / radius

    def fit(self, inputs, outputs, washout=100, ridge_alpha=1e-6):
        n_samples = inputs.shape[0]
        self.states = np.zeros((n_samples, self.n_reservoir))
        x = np.zeros(self.n_reservoir)
        for t in range(n_samples):
            u = inputs[t]
            x = np.tanh(np.dot(self.Win, u) + np.dot(self.W, x)) + self.noise * (np.random.rand(self.n_reservoir) - 0.5)
            self.states[t] = x
        self.model = Ridge(alpha=ridge_alpha)
        self.model.fit(self.states[washout:], outputs[washout:])

    def predict(self, inputs, initial_state=None):
        n_samples = inputs.shape[0]
        x = np.zeros(self.n_reservoir) if initial_state is None else initial_state
        states = np.zeros((n_samples, self.n_reservoir))
        for t in range(n_samples):
            u = inputs[t]
            x = np.tanh(np.dot(self.Win, u) + np.dot(self.W, x))
            states[t] = x
        return self.model.predict(states)

def run():
    st.title("📈 Echo State Network (ESN) predikció")

    st.markdown("""
    Ez a modul bemutatja, hogyan lehet Echo State Network-öt alkalmazni Lorenz-rendszer előrejelzésére vagy saját feltöltött adatokon való tanulásra.
    A feltöltött adatnak legalább 3 oszlopos idősornak kell lennie (pl. x, y, z).
    """)

    steps = st.slider("Adatpontok száma", 500, 3000, 1000)
    train_fraction = st.slider("Tanítási arány", 0.1, 0.9, 0.5)
    reservoir_size = st.slider("Reservoir méret", 50, 500, 100)

    uploaded_df = get_uploaded_data()

    if uploaded_df is not None and uploaded_df.shape[1] >= 3:
        st.success("✅ Feltöltött adat sikeresen betöltve.")
        show_data_overview(uploaded_df)
        data = uploaded_df.iloc[:steps, :3].values
    else:
        st.warning("⚠️ Nem található megfelelő feltöltött adat – Lorenz szimuláció használata.")
        xs, ys, zs = generate_lorenz_data(steps)
        data = np.column_stack([xs, ys, zs])

    X = data[:-1]
    y = data[1:, 0]

    split = int(train_fraction * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    esn = EchoStateNetwork(n_inputs=3, n_reservoir=reservoir_size)
    esn.fit(X_train, y_train)
    prediction = esn.predict(X_test)

    fig, ax = plt.subplots()
    ax.plot(range(len(y_test)), y_test, label="Valós X")
    ax.plot(range(len(prediction)), prediction, label="Predikció", linestyle="--")
    ax.set_title("ESN előrejelzés Lorenz-rendszerre")
    ax.set_xlabel("Időlépések")
    ax.set_ylabel("X érték")
    ax.legend()
    st.pyplot(fig)
