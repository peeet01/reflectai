
import streamlit as st
import numpy as np

def run():
    st.subheader("Hebbian tanulás zajjal")
    x = np.array([[1, 0, 1]])
    y = np.array([[0, 1, 0]])
    w = y.T @ x
    st.write("Súlymátrix:")
    st.write(w)
