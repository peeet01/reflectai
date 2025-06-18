import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

def run():
    st.subheader("🔁 Kuramoto–LSTM predikció")
    st.write("A Kuramoto modellből származó fázisadatok alapján LSTM predikció.")

    N = 5
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

    data = np.array(history)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Csak az első oszcillátort prediktáljuk
    signal = data_scaled[:, 0]
    window = 10
    X, y = [], []
    for i in range(len(signal) - window):
        X.append(signal[i:i+window])
        y.append(signal[i+window])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=(window, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, verbose=0)

    pred = model.predict(X, verbose=0).flatten()

    fig, ax = plt.subplots()
    ax.plot(y, label='Valódi')
    ax.plot(pred, label='Predikció')
    ax.set_title("Kuramoto–LSTM predikció (1. oszcillátor)")
    ax.legend()
    st.pyplot(fig)
