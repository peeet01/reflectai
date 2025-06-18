import streamlit as st
from sklearn.neural_network import MLPClassifier
import numpy as np

def run():
    st.subheader("🧠 XOR predikciós feladat")
    st.write("Egyszerű MLP modell az XOR logikai művelet megtanulására.")

    # Bemenetek és célértékek
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 1, 1, 0])

    # Modell létrehozása és tanítása
    model = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000, random_state=1)
    model.fit(X, y)

    preds = model.predict(X)
    acc = model.score(X, y)

    # Eredmények megjelenítése
    st.write("Bemenetek (X):", X.tolist())
    st.write("Célértékek (y):", y.tolist())
    st.write("Predikciók:", preds.tolist())
    st.success(f"Pontosság: {acc * 100:.1f}%")
