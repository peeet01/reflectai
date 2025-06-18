import streamlit as st
import numpy as np

def run():
    st.subheader("Hebbian tanulás modul")
    st.write("Hebbian tanulás modul fut.")

    x = np.array([[1, 0, 1]])
    y = np.array([[1, 0, 0]])

    w = y.T @ x

    st.write("Bemenet (x):", x)
    st.write("Kimenet (y):", y)
    st.write("Hebbian súlymátrix (w):")
    st.write(w)
