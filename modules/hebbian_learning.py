import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def hebbian_learning(learning_rate, num_neurons):
    # Szimulált bináris bemenetek (pl. 10 minta, num_neurons hosszú)
    inputs = np.random.randint(0, 2, (10, num_neurons))
    
    # Kezdeti súlyok nullák
    weights = np.zeros((num_neurons, num_neurons))
    
    # Hebbian tanulás alkalmazása
    for x in inputs:
        x = x.reshape(-1, 1)  # Oszlopvektor
        weights += learning_rate * np.dot(x, x.T)  # Hebb-szabály: Δw = η * x * x^T

    return weights

def plot_weights(weights):
    fig, ax = plt.subplots()
    cax = ax.matshow(weights, cmap='viridis')
    plt.title("Hebbian Súlymátrix")
    plt.colorbar(cax)
    st.pyplot(fig)

def run(learning_rate, num_neurons):
    weights = hebbian_learning(learning_rate, num_neurons)
    plot_weights(weights)
