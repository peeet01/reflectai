import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

from modules.data_upload import get_uploaded_data, show_data_overview


def run():
    st.title("🧠 MLP előrejelzés Lorenz-rendszerre vagy feltöltött adatra")

    st.markdown("""
    Ez a modul egy több rétegű perceptronnal (MLP) tanít előrejelzést 3D dinamikus rendszerből.  
    Használhatsz saját CSV-t is, amely 3 oszlopot tartalmaz (`x`, `y`, `z`).
    """)

    # 🔽 Adatok betöltése
    df = get_uploaded_data(required_columns=["x", "y", "z"], allow_default=True, default="lorenz")

    if df is None:
        st.warning("⚠️ Nem áll rendelkezésre megfelelő adat a predikcióhoz.")
        return

    st.success("✅ Adat betöltve.")
    show_data_overview(df)

    max_len = len(df)
    if max_len < 2:
        st.error("❌ Az adathalmaz túl rövid előrejelzéshez.")
        return

    # ⚙️ Paraméterek
    steps = st.slider("Adatpontok száma", 100, min(3000, max_len), min(1000, max_len))
    train_frac = st.slider("Tanítási arány", 0.1, 0.9, 0.7)

    data = df[["x", "y", "z"]].values[:steps]

    X = data[:-1]
    y = data[1:, 0]  # a következő időpillanat x komponense

    split = int(train_frac * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 🤖 Modell tanítása
    model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    mse = mean_squared_error(y_test, prediction)

    # 📈 Eredmények megjelenítése
    fig, ax = plt.subplots()
    ax.plot(y_test, label="Valós x", linewidth=2)
    ax.plot(prediction, label="Predikció", linestyle="--")
    ax.set_title("MLP előrejelzés – Lorenz rendszer (x komponens)")
    ax.set_xlabel("Időlépések")
    ax.set_ylabel("x érték")
    ax.legend()
    st.pyplot(fig)

    st.markdown(f"### 📉 Átlagos négyzetes hiba (MSE): `{mse:.6f}`")
