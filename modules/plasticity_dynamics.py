
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    st.subheader("🔄 Plaszticitási dinamika szimuláció")
    st.write("Hebbian elvű súlyváltozási dinamika bemutatása egy tanulási ciklusban.")

    st.markdown("**Paraméterek**:")
    epochs = st.slider("Epoch-ok száma", min_value=10, max_value=200, value=100, step=10)
    lr = st.slider("Tanulási ráta", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
    noise_level = st.slider("Zaj mértéke", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    # Szintetikus bemenetek és célok
    X = np.random.randint(0, 2, (5, 10))  # 5 bemenet, 10 mintán
    Y = np.random.randint(0, 2, (3, 10))  # 3 kimenet, 10 mintán

    weight_history = []

    W = np.zeros((3, 5))  # 3 kimenet × 5 bemenet

    for epoch in range(epochs):
        noisy_X = X + noise_level * np.random.randn(*X.shape)
        W += lr * Y @ noisy_X.T
        weight_history.append(W.copy())

    # Vizualizáció
    final_weights = weight_history[-1]

    st.markdown("### 🔍 Végső súlymátrix")
    fig, ax = plt.subplots()
    sns.heatmap(final_weights, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Súlyváltozási trend egy adott kapcsolat esetén
    w_trend = [W[0, 0] for W in weight_history]
    fig2, ax2 = plt.subplots()
    ax2.plot(w_trend)
    ax2.set_title("Súlyváltozás trend (Neuron 0 – Bemenet 0)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Súlyérték")
    st.pyplot(fig2)

    st.success("Plaszticitás szimuláció sikeresen lefutott.")
