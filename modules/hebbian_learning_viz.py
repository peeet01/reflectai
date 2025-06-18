import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def run():
    st.subheader("Hebbian tanulási vzualizáció")
    st.write("Hebbian tanulás vizualizáció modul fut.")

    # Bemenet és kimenet mátrix
    x = np.random.randint(0, 2, (3, 5))
    y = np.random.randint(0, 2, (2, 5))

    # Hebbian súlymátrix tanulás
    w = y @ x.T

    st.write("Bemeneti mátrix (x):")
    st.write(x)
    st.write("Kimeneti mátrix (y):")
    st.write(y)

    fig, ax = plt.subplots()
    sns.heatmap(w, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
