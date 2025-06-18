
import streamlit as st
import numpy as np
from sklearn.neural_network import MLPClassifier

def run():
    st.subheader("XOR predikció")
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])
    clf = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000)
    clf.fit(X, y)
    pred = clf.predict(X)
    acc = np.mean(pred == y)
    st.write("Predikció:", pred.tolist())
    st.success(f"Pontosság: {acc*100:.2f}%")
