import streamlit as st
import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

def run():
    st.subheader("Lorenz predikció")
    st.write("Lorenz predikció modul fut.")

    # Lorenz x komponens előállítása
    sigma, rho, beta = 10, 28, 8/3
    dt = 0.01
    steps = 1000
    x = np.empty(steps)
    y = np.empty(steps)
    z = np.empty(steps)

    x[0], y[0], z[0] = 0., 1., 1.05

    for i in range(1, steps):
        x[i] = x[i-1] + dt * sigma * (y[i-1] - x[i-1])
        y[i] = y[i-1] + dt * (x[i-1] * (rho - z[i-1]) - y[i-1])
        z[i] = z[i-1] + dt * (x[i-1] * y[i-1] - beta * z[i-1])

    # Idősor ablakos előkészítés
    data = x
    window = 10
    X = []
    y_target = []

    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y_target.append(data[i+window])

    X = np.array(X)
    y_target = np.array(y_target)

    # MLP regresszió
    model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000)
    model.fit(X, y_target)
    preds = model.predict(X)

    fig, ax = plt.subplots()
    ax.plot(y_target, label='Valódi')
    ax.plot(preds, label='Predikció')
    ax.legend()
    st.pyplot(fig)
