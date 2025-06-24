# modules/xor_prediction.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def train_xor(hidden_size, learning_rate, epochs):
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])

    input_size = 2
    output_size = 1

    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))

    losses = []

    for epoch in range(epochs):
        z1 = np.dot(X, W1) + b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        loss = np.mean((y - a2) ** 2)
        losses.append(loss)

        dz2 = (a2 - y) * sigmoid_deriv(z2)
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = np.dot(dz2, W2.T) * (1 - a1 ** 2)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0)

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    return losses, a2

def run():
    st.header("🔁 XOR predikció neurális hálóval")
    hidden_size = st.slider("Rejtett réteg mérete", 1, 10, 2)
    learning_rate = st.slider("Tanulási ráta", 0.001, 1.0, 0.1)
    epochs = st.number_input("Epochok száma", 100, 10000, 1000, step=100)

    losses, predictions = train_xor(hidden_size, learning_rate, epochs)

    st.subheader("📉 Tanulási veszteség")
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Veszteség")
    st.pyplot(fig)

    st.subheader("📊 Predikciók")
    st.write("Predikált értékek az XOR bemenetre:")
    st.dataframe(predictions.round(3))
