import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

from modules.data_upload import get_uploaded_data, show_data_overview


def generate_lorenz_data(n_points=1000, dt=0.01):
    def lorenz(x, y, z, s=10, r=28, b=8/3):
        dx = s * (y - x)
        dy = r * x - y - x * z
        dz = x * y - b * z
        return dx, dy, dz

    xs, ys, zs = np.empty(n_points), np.empty(n_points), np.empty(n_points)
    x, y, z = 0., 1., 1.05
    for i in range(n_points):
        dx, dy, dz = lorenz(x, y, z)
        x += dx * dt
        y += dy * dt
        z += dz * dt
        xs[i], ys[i], zs[i] = x, y, z
    return xs, ys, zs


def run():
    st.title("🧠 MLP előrejelzés Lorenz-rendszerre vagy feltöltött adatra")

    st.markdown("""
    Ez a modul egy több rétegű perceptronnal (MLP) tanít előrejelzést 3D dinamikus rendszerből.  
    Használhatsz saját CSV-t is, amely 3 oszlopot tartalmaz (`x`, `y`, `z`).
    """)

    steps = st.slider("Adatpontok száma", 500, 3000, 1000)
    train_frac = st.slider("Tanítási arány", 0.1, 0.9, 0.7)

    df = get_uploaded_data(required_columns=["x", "y", "z"], allow_default=True, default="lorenz")

    if df is not None:
        st.success("✅ Adat betöltve.")
        show_data_overview(df)

        # Feldolgozás
        data = df[["x", "y", "z"]].values[:steps]
        X = data[:-1]
        y = data[1:, 0]  # következő időlépés x

        split = int(train_frac * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        prediction = model.predict(X_test)
        mse = mean_squared_error(y_test, prediction)

        fig, ax = plt.subplots()
        ax.plot(y_test, label="Valós x", linewidth=2)
        ax.plot(prediction, label="Predikció", linestyle="--")
        ax.set_title("MLP előrejelzés – Lorenz rendszer (x komponens)")
        ax.set_xlabel("Időlépések")
        ax.set_ylabel("x érték")
        ax.legend()
        st.pyplot(fig)

        st.markdown(f"### 📉 Átlagos négyzetes hiba (MSE): `{mse:.6f}`")
    else:
        st.warning("⚠️ Nem áll rendelkezésre megfelelő adat a predikcióhoz.")
