import streamlit as st
import numpy as np

def run():
    st.title("Persistent Homology")
    st.write("Ez a modul bemutatja a perzisztens homológia alapjait szintetikus adatokon.")
    
    points = np.random.rand(100, 2)
    st.scatter_chart(points)
    st.info("Ez csak egy vizualizációs példa a modulra.")