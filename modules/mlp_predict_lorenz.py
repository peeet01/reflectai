import streamlit as st
import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def generate_lorenz_data(n_steps=1000, dt=0.01):
    def lorenz(t, state, sigma=10, rho=28, beta=8/3):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    t_span = (0, n_steps * dt)
    y0 = [1.0, 1.0, 1.0]
    t_eval = np.linspace(*t_span, n_steps)
    sol = solve_ivp(lorenz, t_span, y0, t_eval=t_eval)
    return sol.y.T  # shape: (n_steps, 3)

def create_dataset(data, window=10):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window].flatten())
        y.append(data[i+window][0])  # x komponens előrejelzése
    return np.array(X), np.array(y)

def run():
    st.header("🔮 Lorenz rendszer MLP-predikció")
    st.markdown("""
    Ez a modul egy **MLP (Multi-Layer Perceptron)** modellt tanít a Lorenz rendszer múltbeli állapotai alapján, hogy előre jelezze a jövőbeli **x-komponens** értékét.

    A cél a nemlineáris dinamikák tanulása és előrejelzése.
    """)

    steps = st.slider("Szimulációs lépések", 200, 3000, 1000, 100)
    window = st.slider("Ablakméret (window size)", 5, 30, 10)
    hidden_layer_size = st.slider("Rejtett réteg méret", 5, 100, 50)
    test_ratio = st.slider("Tesztelési arány", 0.1, 0.5, 0.2)

    data = generate_lorenz_data(n_steps=steps)
    X, y = create_dataset(data, window=window)

    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = MLPRegressor(hidden_layer_sizes=(hidden_layer_size,), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    st.subheader("📉 Előrejelzés vizualizáció")
    fig, ax = plt.subplots()
    ax.plot(y_test, label="Valódi", linewidth=2)
    ax.plot(predictions, label="MLP predikció", linestyle='dashed')
    ax.set_xlabel("Időlépések")
    ax.set_ylabel("x komponens")
    ax.legend()
    st.pyplot(fig)

    score = model.score(X_test, y_test)
    st.success(f"Modell pontosság (R²): {score:.3f}")
app = run
