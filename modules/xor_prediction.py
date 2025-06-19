import streamlit as st
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def train_xor():
    # Bemenetek és elvárt kimenetek
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])

    # Súlyok inicializálása
    np.random.seed(1)
    weights0 = 2 * np.random.random((2, 4)) - 1
    weights1 = 2 * np.random.random((4, 1)) - 1

    # Tanítás
    for j in range(10000):
        layer0 = X
        layer1 = sigmoid(np.dot(layer0, weights0))
        layer2 = sigmoid(np.dot(layer1, weights1))

        layer2_error = y - layer2
        layer2_delta = layer2_error * sigmoid_deriv(layer2)
        layer1_error = layer2_delta.dot(weights1.T)
        layer1_delta = layer1_error * sigmoid_deriv(layer1)

        weights1 += layer1.T.dot(layer2_delta)
        weights0 += layer0.T.dot(layer1_delta)

    return weights0, weights1

def predict(x, weights0, weights1):
    layer1 = sigmoid(np.dot(x, weights0))
    output = sigmoid(np.dot(layer1, weights1))
    return output

def run():
    st.header("🧠 XOR Tanítás egyszerű hálóval")
    weights0, weights1 = train_xor()

    input1 = st.selectbox("Első bemenet", [0, 1], key="xor_input1")
    input2 = st.selectbox("Második bemenet", [0, 1], key="xor_input2")

    x = np.array([[input1, input2]])
    output = predict(x, weights0, weights1)

    st.write(f"🧩 Bemenet: ({input1}, {input2})")
    st.write(f"📈 Kimenet (0-1): {output[0][0]:.4f}")
    st.write(f"✅ Eredmény: {int(round(output[0][0]))}")
