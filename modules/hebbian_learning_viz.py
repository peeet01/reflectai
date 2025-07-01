import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def hebbian_learning(x, y):
    return y @ x.T

def run():
    st.title("🧠 Hebbian Tanulás Vizualizáció")
    st.markdown("""
    A **Hebbian tanulás** egy egyszerű szabály:  
    _"A neuronok, amelyek együtt tüzelnek, össze is kapcsolódnak."_  
    Ez a vizualizáció bemutatja a tanult súlymátrixot és annak hatását egy új bemenetre.
    """)

    # Interaktív beállítások
    input_neurons = st.slider("🔢 Bemeneti neuronok száma", 2, 10, 3)
    output_neurons = st.slider("🔢 Kimeneti neuronok száma", 2, 10, 2)
    patterns = st.slider("📊 Minták száma", 3, 20, 5)

    # Bemeneti és kimeneti minták (véletlenszerű bináris)
    x = np.random.randint(0, 2, (input_neurons, patterns))
    y = np.random.randint(0, 2, (output_neurons, patterns))

    # Súlytanulás Hebbian szabály szerint
    w = hebbian_learning(x, y)

    # Vizualizáció – súlymátrix
    st.subheader("📘 Súlymátrix $W = Y \\cdot X^T$")
    fig1, ax1 = plt.subplots()
    sns.heatmap(w, annot=True, cmap='coolwarm', ax=ax1, cbar=True)
    ax1.set_xlabel("Bemeneti neuronok")
    ax1.set_ylabel("Kimeneti neuronok")
    st.pyplot(fig1)

    # Vizualizáció – súlyzott kimenet (aktiváció)
    st.subheader("🔁 Súlyozott aktiváció $Y_{pred} = W \\cdot X$")
    y_pred = w @ x
    fig2, ax2 = plt.subplots()
    sns.heatmap(y_pred, annot=True, cmap='YlGnBu', ax=ax2, cbar=True)
    ax2.set_xlabel("Minták")
    ax2.set_ylabel("Kimeneti neuronok (jósolt)")
    st.pyplot(fig2)

    # Nyers mátrixok megjelenítése
    with st.expander("🧾 Részletes mátrixok"):
        st.write("Bemeneti mátrix (X):")
        st.dataframe(x)
        st.write("Célmátrix (Y):")
        st.dataframe(y)
        st.write("Tanult súlymátrix (W):")
        st.dataframe(w)
        st.write("Kiszámított jósolt kimenet (Y_pred):")
        st.dataframe(y_pred)

# Kötelező ReflectAI-kompatibilitás
app = run
