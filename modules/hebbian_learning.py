import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.subheader("🧬 Hebbian tanulás zajjal")
    st.write("Ez a modul egy egyszerű Hebbian tanulási folyamatot szimulál zajos környezetben.")

    # Szimuláció paraméterei
    np.random.seed(42)
    epochs = 100
    eta = 0.01  # tanulási ráta
    inputs = np.random.randn(epochs, 2)
    noise = np.random.normal(0, 0.1, size=inputs.shape)
    inputs_noisy = inputs + noise

    weights = np.zeros(2)
    weight_history = []

    for x in inputs_noisy:
        weights += eta * x * x  # Hebbian tanulás (önmegerősítés)
        weight_history.append(weights.copy())

    weight_history = np.array(weight_history)

    # Grafikon kirajzolása
    fig, ax = plt.subplots()
    ax.plot(weight_history[:, 0], label='w1')
    ax.plot(weight_history[:, 1], label='w2')
    ax.set_title("Hebbian súlyváltozás iterációnként")
    ax.set_xlabel("Iteráció")
    ax.set_ylabel("Súlyérték")
    ax.legend()
    st.pyplot(fig)
