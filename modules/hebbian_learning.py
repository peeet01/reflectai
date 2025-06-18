import streamlit as st
import numpy as np

def run():
    st.write("Hebbian tanulás modul fut.")

    x = np.array([[1], [0], [1]])  # bemenet
    y = np.array([[1], [0], [0]])  # kimenet
    w = np.dot(y, x.T)            # Hebbian szabály: w = y * x^T

    st.write("Bemenet (x):", x)
    st.write("Kimenet (y):", y)
    st.write("Súlymátrix (w):", w)
