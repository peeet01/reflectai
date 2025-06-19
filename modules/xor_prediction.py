import numpy as np
import streamlit as st

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def run(hidden_size=4, learning_rate=0.1, epochs=1000, note=""):
    st.subheader("XOR predikció neurális hálóval")
    st.markdown(f"**Jegyzet:** {note}")

    # Bemeneti és kimeneti adatok az XOR problémára
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])

    input_size = 2
    output_size = 1

    # Súlyok inicializálása
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))

    losses = []

    for epoch in range(epochs):
        # Előre terjesztés
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        output = sigmoid(z2)

        # Veszteség kiszámítása (MSE)
        loss = np.mean((y - output) ** 2)
        losses.append(loss)

        # Visszaterjesztés (gradient descent)
        error = y - output
        d_output = error * sigmoid_derivative(output)

        error_hidden = d_output.dot(W2.T)
        d_hidden = error_hidden * sigmoid_derivative(a1)

        # Súlyok frissítése
        W2 += a1.T.dot(d_output) * learning_rate
        b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        W1 += X.T.dot(d_hidden) * learning_rate
        b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    st.line_chart(losses)

    st.write("Végső kimenetek:")
    for i, (x_i, y_i) in enumerate(zip(X, output)):
        st.write(f"Bemenet: {x_i} → Predikció: {y_i[0]:.4f}")
