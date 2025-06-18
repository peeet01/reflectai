import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.subheader("🧬 Hebbian tanulás zajjal")
    st.write("Hebbian súlyfrissítés zajos bemenettel.")

    np.random.seed(42)
    eta = 0.01
    epochs = 100
    inputs = np.random.randn(epochs, 2)
    noise = np.random.normal(0, 0.1, inputs.shape)
    inputs += noise

    weights = np.zeros(2)
    history = []

    for x in inputs:
        weights += eta * x * x
        history.append(weights.copy())

    history = np.array(history)
    fig, ax = plt.subplots()
    ax.plot(history[:, 0], label='w1')
    ax.plot(history[:, 1], label='w2')
    ax.set_title("Hebbian súlyok változása")
    ax.legend()
    st.pyplot(fig)
