import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run(num_neurons, learning_rate, iterations, user_note):
    st.subheader("🧠 Hebbian tanulás vizualizáció")

    # Kezdeti bemenetek
    inputs = np.random.rand(iterations, num_neurons)
    weights = np.random.rand(num_neurons, num_neurons)

    # Hebbian tanulás
    for i in range(iterations):
        x = inputs[i].reshape(-1, 1)
        weights += learning_rate * x @ x.T

    # Megjelenítés
    fig, ax = plt.subplots()
    im = ax.imshow(weights, cmap='viridis')
    plt.colorbar(im, ax=ax)
    ax.set_title("Tanult súlymátrix")
    st.pyplot(fig)

    # Megjegyzés megjelenítése
    if user_note:
        st.info(f"📝 Megjegyzés: {user_note}")
