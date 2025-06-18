import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def run():
    st.subheader("🧠 XOR predikció")
    st.write("Egyszerű MLP modell az XOR logikai kapu tanulására.")

    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])

    np.random.seed(42)
    hidden_weights = np.random.randn(2, 3)
    output_weights = np.random.randn(3, 1)

    learning_rate = 0.1
    epochs = 5000
    errors = []

    for _ in range(epochs):
        hidden_input = np.dot(X, hidden_weights)
        hidden_output = sigmoid(hidden_input)

        final_input = np.dot(hidden_output, output_weights)
        final_output = sigmoid(final_input)

        error = Y - final_output
        errors.append(np.mean(np.abs(error)))

        d_output = error * sigmoid_deriv(final_input)
        d_hidden = np.dot(d_output, output_weights.T) * sigmoid_deriv(hidden_input)

        output_weights += learning_rate * np.dot(hidden_output.T, d_output)
        hidden_weights += learning_rate * np.dot(X.T, d_hidden)

    fig, ax = plt.subplots()
    ax.plot(errors)
    ax.set_title("XOR predikció – Hiba alakulása")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Átlagos abszolút hiba")
    st.pyplot(fig)

    st.success(f"Végső predikciós hiba: {errors[-1]:.4f}")
