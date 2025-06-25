import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def run():
    st.markdown("## Szinkronfraktál dimenzióanalízis")

    st.markdown(
        "Ez a modul a hálózat szinkronizációs mintázatainak fraktál dimenzióját méri a doboz-számlálási módszerrel "
        "egy binarizált fázistér alapján."
    )

    size = st.slider("Mátrix méret (NxN)", 50, 300, 100, step=10)
    threshold = st.slider("Küszöb a binarizáláshoz", 0.1, 1.0, 0.5, step=0.1)

    # Véletlenszerű "szinkron" mátrix generálása
    np.random.seed(0)
    matrix = np.random.rand(size, size)
    binary_matrix = (matrix > threshold).astype(int)

    # Fraktál dimenzió számítás (box-counting)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where(S > 0)[0])

    Z = binary_matrix
    sizes = 2**np.arange(1, int(np.log2(size)))
    counts = [boxcount(Z, s) for s in sizes]

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    fd = -coeffs[0]

    # Ábra
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(binary_matrix, cmap='binary')
    ax[0].set_title("Binarizált mátrix")

    ax[1].plot(np.log(sizes), np.log(counts), 'o-', label=f"FD ≈ {fd:.2f}")
    ax[1].set_title("Fraktál dimenzió log–log skálán")
    ax[1].set_xlabel("log(doboz méret)")
    ax[1].set_ylabel("log(doboz száma)")
    ax[1].legend()

    st.pyplot(fig)
    st.success(f"🔢 Becsült fraktál dimenzió: **{fd:.3f}**")
app = run
