
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

def fractal_dimension(Z, threshold=0.9):
    assert(len(Z.shape) == 2)

    def boxcount(Z, k):
        S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                   np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where(S > 0)[0])

    Z = (Z < threshold)
    p = min(Z.shape)
    n = 2**np.floor(np.log2(p))
    sizes = 2**np.arange(int(np.log2(n)), 1, -1)
    counts = [boxcount(Z, int(size)) for size in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0], sizes, counts

def run():
    st.title("Fraktáldimenzió analízis (Box-counting)")
    st.write("Ez a modul binarizált mátrixon számolja ki a fraktáldimenziót box-counting módszerrel.")

    size = st.slider("Mátrixméret", 64, 512, 128, step=64)
    density = st.slider("Zaj sűrűsége", 0.01, 0.3, 0.1)

    np.random.seed(42)
    Z = np.random.rand(size, size)
    Z = (Z < density).astype(int)

    dim, sizes, counts = fractal_dimension(Z)

    st.success(f"Számított fraktáldimenzió: {dim:.4f}")
