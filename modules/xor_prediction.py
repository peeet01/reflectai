import streamlit as st
from sklearn.neural_network import MLPClassifier
import numpy as np

def run_xor():
    st.subheader("🧠 XOR predikció")
    st.write("Egyszerű MLP tanul XOR logikai kaput.")

    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 1, 1, 0])

    model = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000)
    model.fit(X, y)
    preds = model.predict(X)

    st.write("Bemenetek:", X.tolist())
    st.write("Predikciók:", preds.tolist())
