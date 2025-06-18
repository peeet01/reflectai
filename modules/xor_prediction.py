import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def run():
    st.subheader("🧠 XOR predikció – vizuális MLP tanulás")
    st.write("Egyszerű 2-3-1 MLP modell vizuális visszacsatolással az XOR logikai kapura.")

    # Tanító adatok
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])

    np.random.seed(42)
    hidden_weights = np.random.randn(2, 3)
    output_weights = np.random.randn(3, 1)

    learning_rate = 0.1
    epochs = 5000
    errors = []
    hidden_acts = []

    for _ in range(epochs):
        hidden_input = np.dot(X, hidden_weights)
        hidden_output = sigmoid(hidden_input)
        final_input = np.dot(hidden_output, output_weights)
        final_output = sigmoid(final_input)

        error = Y - final_output
        errors.append(np.mean(np.abs(error)))
        hidden_acts.append(hidden_output)

        d_output = error * sigmoid_deriv(final_input)
        d_hidden = np.dot(d_output, output_weights.T) * sigmoid_deriv(hidden_input)

        output_weights += learning_rate * np.dot(hidden_output.T, d_output)
        hidden_weights += learning_rate * np.dot(X.T, d_hidden)

    # ⬛ 1. Hibatörténet
    fig1, ax1 = plt.subplots()
    ax1.plot(errors)
    ax1.set_title("Tanulási hiba – XOR MLP")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Átlagos absz. hiba")
    st.pyplot(fig1)

    # ⬛ 2. Rejtett réteg vizualizáció (utolsó epoch)
    hidden_output = sigmoid(np.dot(X, hidden_weights))
    fig2, ax2 = plt.subplots()
    for i, label in enumerate(Y.flatten()):
        ax2.scatter(hidden_output[i, 0], hidden_output[i, 1], c='r' if label == 1 else 'b', label=f"Bemenet: {X[i]}")
    ax2.set_title("Rejtett réteg 2D aktivációi")
    ax2.set_xlabel("Neuron 1")
    ax2.set_ylabel("Neuron 2")
    st.pyplot(fig2)

    # ⬛ 3. Végső kimenet kiírás
    final_output = sigmoid(np.dot(hidden_output, output_weights))
    st.write("### Predikciók a bemenetekre:")
    for x, pred in zip(X, final_output):
        st.write(f"`{x}` → **{pred[0]:.4f}**")

    st.success(f"Végső átlagos hiba: {errors[-1]:.4f}")
