
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.subheader("Hebbian tanulás zajjal")
    X = np.random.randint(0, 2, (100, 3))
    Y = np.roll(X, 1, axis=0)
    W = np.zeros((3, 3))
    for x, y in zip(X, Y):
        W += np.outer(y, x)
    fig, ax = plt.subplots()
    im = ax.imshow(W, cmap="viridis")
    st.pyplot(fig)
    st.write("Súlymátrix", W)
