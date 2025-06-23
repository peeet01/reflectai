import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.title("📊 Perzisztens homológia")
    st.write("Ez a modul bemutatja a perzisztens homológia alapjait szintetikus adatokon.")

    # Szintetikus pontfelhő generálása
    points = np.random.rand(100, 2)

    # Egyszerű vizualizáció
    st.subheader("Pontfelhő")
    st.scatter_chart(points)

    st.info("Ez csak egy vizualizációs példa a perzisztens homológia bevezetéséhez. "
            "A TDA részletes analízise külső könyvtárakat (pl. GUDHI, Ripser) igényelne.")
