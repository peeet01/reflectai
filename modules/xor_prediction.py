import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.subheader("🧠 XOR predikciós modell")
    st.write("Egyszerű MLP tanítás az XOR problémára.")

    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])
    
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    def sigmoid_deriv(x): return x * (1 - x)

    np.random.seed(1)
    w1 = 2*np.random.random((2,4)) - 1
    w2 = 2*np.random.random((4,1)) - 1
    losses = []

    for epoch in range(5000):
        l1 = sigmoid(np.dot(X, w1))
        l2 = sigmoid(np.dot(l1, w2))
        l2_error = y - l2
        if epoch % 500 == 0:
            losses.append(np.mean(np.abs(l2_error)))
        l2_delta = l2_error * sigmoid_deriv(l2)
        l1_error = l2_delta.dot(w2.T)
        l1_delta = l1_error * sigmoid_deriv(l1)
        w2 += l1.T.dot(l2_delta)
        w1 += X.T.dot(l1_delta)

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_title("XOR tanulási hiba (500 epizódonként)")
    st.pyplot(fig)
    st.success(f"Végső predikciók: {np.round(l2.T[0], 2)}")
