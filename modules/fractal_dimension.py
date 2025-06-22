import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from modules.data_upload import get_uploaded_data, show_data_overview


def fractal_dimension(Z, threshold=0.9):
    assert len(Z.shape) == 2

    def boxcount(Z, k):
        S = zoom(Z, (1.0 / k, 1.0 / k), order=0)
        return np.sum(S > threshold)

    sizes = 2 ** np.arange(1, 7)
    counts = [boxcount(Z, size) for size in sizes]

    coeffs = np.polyfit(np.log(1.0 / sizes), np.log(counts), 1)
    return -coeffs[0], sizes, counts


def run():
    st.title("🧮 Fraktáldimenzió becslés (dobozolásos módszerrel)")

    st.markdown("""
    Ez a modul 2D bináris képen vagy mátrixon becsli a fraktáldimenziót.  
    Tölthetsz fel saját CSV-t is, ahol a cellák értékei 0–1 közötti számok.
    """)

    df = get_uploaded_data(allow_default=True, default="fractal", required_columns=None)

    if df is None:
        st.warning("⚠️ Nem áll rendelkezésre megfelelő mátrix.")
        return

    show_data_overview(df)

    try:
        data = df.values.astype(float)
        if np.isnan(data).any():
            st.error("❌ Az adathalmaz NaN értékeket tartalmaz.")
            return
        if data.ndim != 2:
            st.error("❌ A fraktáldimenzió csak 2D mátrixokra alkalmazható.")
            return
    except Exception as e:
        st.error(f"Hiba az adat értelmezésekor: {e}")
        return

    threshold = st.slider("Küszöb érték (binárosításhoz)", 0.0, 1.0, 0.9, 0.01)

    dim, sizes, counts = fractal_dimension(data, threshold=threshold)

    st.markdown(f"### 📏 Becsült fraktáldimenzió: `{dim:.4f}`")

    fig, ax = plt.subplots()
    ax.plot(np.log(1.0 / sizes), np.log(counts), 'o-', label='Box count')
    ax.set_xlabel("log(1/box size)")
    ax.set_ylabel("log(count)")
    ax.set_title("Fraktáldimenzió becslés")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.markdown("### 🖼️ Binárosított bemenet (vizualizáció)")
    st.image((data > threshold).astype(float), use_column_width=True)
