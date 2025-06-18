import streamlit as st
import numpy as np

def run_hebbian():
    st.subheader("🧬 Hebbian tanulás")
    st.write("Klasszikus Hebbian súlytanulás bemeneti és kimeneti mintákon.")

    x = np.array([[1, 0, 1]])
    y = np.array([[1, 0, 0]])
    w = y.T @ x

    st.write("Bemenet (x):", x)
    st.write("Kimenet (y):", y)
    st.write("Súlymátrix (w):", w)
