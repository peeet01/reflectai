import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def run(learning_rate=0.1, num_neurons=10):
    st.subheader("🧠 Hebbian tanulás vizualizáció")

    # Véletlenszerű bemenetek generálása
    num_inputs = num_neurons
    inputs = np.random.rand(num_inputs, num_neurons)

    # Kezdeti súlyok
    weights = np.zeros((num_neurons, num_neurons))
    snapshots = []

    epochs = 10
    for epoch in range(epochs):
        for x in inputs:
            x = x.reshape(-1, 1)
            weights += learning_rate * np.dot(x, x.T)

        # Snapshot minden 2. epoch után
        if epoch % 2 == 0 or epoch == epochs - 1:
            snapshots.append(weights.copy())

    # 🔥 Súlymátrix (heatmap) kirajzolása
    st.markdown("### 🔁 Súlymátrix változása")
    for i, w in enumerate(snapshots):
        fig, ax = plt.subplots()
        cax = ax.matshow(w, cmap="viridis")
        plt.title(f"Epoch {i*2}")
        plt.colorbar(cax)
        st.pyplot(fig)

    # 📊 Végső súlymátrix
    st.markdown("### 🧮 Végső súlymátrix")
    fig, ax = plt.subplots()
    cax = ax.matshow(weights, cmap='plasma')
    plt.title("Végső Hebbian súlymátrix")
    plt.colorbar(cax)
    st.pyplot(fig)

    # Aktiváció megjelenítése
    st.markdown("### ⚡ Neuronaktivációk példája")
    activation = np.dot(weights, inputs.T).T
    fig, ax = plt.subplots()
    ax.plot(activation)
    plt.title("Neuronaktivációk")
    plt.xlabel("Minta index")
    plt.ylabel("Aktiváció szint")
    st.pyplot(fig)
