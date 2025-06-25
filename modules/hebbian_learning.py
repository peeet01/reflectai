# modules/hebbian_learning.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def hebbian_learning(inputs, targets, learning_rate):
    n_features = inputs.shape[1]
    weights = np.zeros((n_features,))
    for x, t in zip(inputs, targets):
        weights += learning_rate * x * t
    return weights

def run():
    st.header("🧠 Hebbian tanulás – szinaptikus súlytanulás")
    learning_rate = st.slider("Tanulási ráta (η)", 0.01, 1.0, 0.1)
    num_neurons = st.slider("Bemenetek száma", 2, 10, 3)

    # Bemeneti adatok és célértékek
    inputs = np.random.randint(0, 2, size=(10, num_neurons))
    targets = np.random.choice([-1, 1], size=10)

    st.subheader("🔢 Bemenetek és célértékek")
    st.write("Inputs:", inputs)
    st.write("Célértékek:", targets)

    weights = hebbian_learning(inputs, targets, learning_rate)

    st.subheader("📊 Tanult súlyok")
    fig, ax = plt.subplots()
    ax.bar(range(len(weights)), weights)
    ax.set_xlabel("Bemenet indexe")
    ax.set_ylabel("Súly érték")
    st.pyplot(fig)
app = run
