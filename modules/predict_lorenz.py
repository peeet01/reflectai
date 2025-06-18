import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def run():
    st.write("Lorenz predikció modul fut.")

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

    # Gépi tanulás predikció
    window = 5
    X, y_pred = [], []
    for i in range(len(x) - window):
        X.append(x[i:i+window])
        y_pred.append(x[i+window])
    X = np.array(X)
    y_pred = np.array(y_pred)

    model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000)
    model.fit(X, y_pred)
    pred = model.predict(X)

    st.line_chart(pred)
