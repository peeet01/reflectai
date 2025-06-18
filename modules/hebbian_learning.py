import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.subheader("🧬 Hebbian tanulás")
    st.write("Egyszerű Hebbian tanulási modell zajjal.")

    epochs = 100
    lr = 0.01
    noise_level = 0.2

    # Bemeneti és kimeneti minták (XOR logika alapján)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])

    W = np.random.randn(2)
    bias = np.random.randn()
    errors = []

    for epoch in range(epochs):
        total_error = 0
        for x, y in zip(X, Y):
            x_noisy = x + noise_level * np.random.randn(2)
            output = np.dot(W, x_noisy) + bias
            pred = 1 if output > 0 else 0
            error = y - pred
            W += lr * error * x
            bias += lr * error
            total_error += abs(error)
        errors.append(total_error)

    fig, ax = plt.subplots()
    ax.plot(errors)
    ax.set_title("Hibatörténet – Hebbian tanulás")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hibaösszeg")
    st.pyplot(fig)

    st.success("Hebbian tanulás lefutott.")
