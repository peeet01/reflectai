import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.subheader("🧬 Hebbian tanulás szimuláció")
    st.write("Egyszerű Hebbian tanulás példája bemenet és kimenet alapján.")

    # Példabemenet
    X = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0]
    ])

    Y = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # Hebbian tanulás: W = Y.T @ X
    W = Y.T @ X

    st.write("Bemenet (X):")
    st.dataframe(X)

    st.write("Kimenet (Y):")
    st.dataframe(Y)

    st.write("Tanult súlymátrix (W = Yᵀ · X):")
    st.dataframe(W)

    # Vizualizáljuk a súlymátrixot
    fig, ax = plt.subplots()
    cax = ax.matshow(W, cmap='viridis')
    plt.title("Hebbian súlymátrix")
    plt.xlabel("Input neuronok")
    plt.ylabel("Output neuronok")
    fig.colorbar(cax)
    st.pyplot(fig)
