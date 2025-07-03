"""
Fractal Dimension Analysis Module
---------------------------------
Ez a modul a fraktál dimenzió becslését valósítja meg bináris képadatok alapján,
box-counting (dobozszámlálásos) módszerrel.

A fraktál dimenzió mértéke megmutatja, hogy egy adott geometriai objektum (pl. agyi aktivitásmintázat)
milyen komplexitással tölti ki a teret. Különösen hasznos agyi képalkotásban, EEG/MEG jelek elemzésében
és neurális struktúrák vizsgálatában.

📚 Tudományos háttér:
- Falconer, K. (2003). *Fractal geometry: Mathematical foundations and applications.*
- Esteban, F. J., et al. (2009). *Fractal dimension and white matter changes in Alzheimer’s disease*. NeuroImage.

Author: ReflectAI
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.transform import resize
import io

def boxcount(Z, k):
    """Kiszámítja, hány doboz szükséges az objektum lefedésére adott k méret mellett."""
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k), axis=1)
    return len(np.where(S > 0)[0])

def fractal_dimension(Z, threshold=0.9):
    """Fraktál dimenzió becslése dobozszámlálásos módszerrel."""
    assert len(Z.shape) == 2, "Képnek kétdimenziósnak kell lennie"
    Z = Z < threshold  # binarizálás
    p = min(Z.shape)
    n = 2**np.floor(np.log2(p))  # legnagyobb 2 hatvány, ami belefér
    sizes = 2**np.arange(int(np.log2(n)), 1, -1)
    counts = [boxcount(Z, size) for size in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def app():
    st.title("🧮 Fractal Dimension Analyzer")

    st.markdown("""
    Ez a modul a képi fraktál dimenzió becslésére szolgál.  
    A módszer a **box-counting** eljáráson alapul.

    > *A fraktál dimenzió egy nem egész számú dimenzió, amely azt írja le,  
    hogy egy objektum mennyire tölti ki a teret különböző skálákon.*
    """)

    uploaded_file = st.file_uploader("📤 Tölts fel képet (pl. neuronrajz, agyi minta)...", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")
        img_arr = np.array(image)
        thresh = threshold_otsu(img_arr)
        binary = img_arr > thresh
        fd = fractal_dimension(binary)

        st.image(image, caption="📷 Eredeti kép", use_column_width=True)
        st.subheader(f"🧠 Becsült fraktál dimenzió: `{fd:.4f}`")

        fig, ax = plt.subplots()
        ax.imshow(binary, cmap='gray')
        ax.set_title("🧩 Binarizált kép (küszöb: Otsu)")
        ax.axis('off')
        st.pyplot(fig)

    with st.expander("📘 Tudományos háttér"):
        st.markdown("""
        A **fraktál dimenzió** (D) egy mérőszám, amely megmutatja, hogy egy objektum  
        hogyan változik a részletgazdagsága különböző nagyítási szinteken.

        A **box-counting dimenzió** formulája:
        $$
        D = \\lim_{\\varepsilon \\to 0} \\frac{\\log N(\\varepsilon)}{\\log(1/\\varepsilon)}
        $$

        Ahol:
        - $N(\\varepsilon)$ a szükséges dobozok száma, amelyek lefedik az objektumot,
        - $\\varepsilon$ a doboz mérete.

        **Alkalmazásai:**
        - agyi EEG mintázatok komplexitásának elemzése,
        - morfológiai vizsgálatok (pl. neuronformák),
        - Alzheimer- és Parkinson-kór strukturális biomarkerei.
        """)

# Kötelező ReflectAI-kompatibilitás
app = app
