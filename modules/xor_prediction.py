import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

def run(hidden_size=2, learning_rate=0.1, epochs=1000, note=""):
    st.subheader("🔀 XOR predikciós háló továbbfejlesztve")
    
    # XOR adat
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    np.random.seed(42)
    W1 = np.random.randn(2, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, 1)
    b2 = np.zeros((1, 1))

    loss_history = []
    hidden_activations = []

    for epoch in range(epochs):
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        loss = np.mean((y - a2) ** 2)
        loss_history.append(loss)

        if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
            hidden_activations.append(a1.copy())

        # Backpropagation
        delta2 = (a2 - y) * sigmoid_deriv(z2)
        dW2 = np.dot(a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)

        delta1 = np.dot(delta2, W2.T) * sigmoid_deriv(z1)
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    # Előrejelzés és pontosság
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    predictions = (a2 > 0.5).astype(int)
    accuracy = np.mean(predictions == y)

    st.write("🎯 Pontosság:", f"{accuracy * 100:.2f}%")
    st.write("📈 Előrejelzések:", predictions.ravel().tolist())

    # Tanulási görbe
    fig1, ax1 = plt.subplots()
    ax1.plot(loss_history, label="Veszteség")
    ax1.set_title("📉 Tanulási görbe")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    st.pyplot(fig1)

    # Rejtett aktivációk vizualizációja
    fig2, ax2 = plt.subplots()
    for i, a in enumerate(hidden_activations):
        ax2.plot(range(a.shape[1]), a.mean(axis=0), label=f"Epoch {i * (epochs // 10)}")
    ax2.set_title("🧠 Rejtett réteg aktivációk")
    ax2.set_xlabel("Neuron index")
    ax2.set_ylabel("Átlagos aktiváció")
    ax2.legend()
    st.pyplot(fig2)

    # Megjegyzés megjelenítés
    if note:
        st.info(f"📎 Megjegyzés: {note}")
