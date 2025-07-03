"""
🧠 Fractal Dimension Analyzer modul – ReflectAI rendszerhez

Ez a modul lehetővé teszi a felhasználónak, hogy képeken becsülje meg a fraktáldimenziót.
Támogatja a Box-Counting algoritmust, Gaussian zaj hozzáadását, benchmarkolást és CSV exportot.

Felhasználási területek:
- Orvosi képdiagnosztika
- Textúraelemzés
- Mintázatok komplexitásának mérése

Tudományos háttér:
A fraktáldimenzió egy mérték, amely leírja, hogy egy objektum milyen „térkitöltő” tulajdonságú. 
A Box-Counting algoritmus során különböző méretű rácsokat helyezünk a képre, majd megszámoljuk, hány cellát tölt ki a mintázat.

Dimenzió képlete:
    D ≈ - lim(ε → 0) [log(N(ε)) / log(ε)]
ahol N(ε) a lefedéshez szükséges dobozok száma ε méretben.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage import data, color, util, io
from skimage.transform import resize
import pandas as pd

# 🧮 Fraktáldimenzió számítása box-counting módszerrel
def fractal_dimension(Z, threshold=0.9):
    assert len(Z.shape) == 2
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where(S > 0)[0])
    Z = (Z < threshold)
    p = min(Z.shape)
    n = 2**np.floor(np.log2(p))
    sizes = 2**np.arange(int(np.log2(n)), 1, -1)
    counts = [boxcount(Z, size) for size in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

# 📉 Benchmark ábra zaj és dimenzió kapcsolatáról
def benchmark_noise_response(img_gray, threshold):
    sigmas = np.linspace(0.0, 1.0, 10)
    dimensions = []
    for sigma in sigmas:
        noisy = util.random_noise(img_gray, var=sigma**2)
        dim = fractal_dimension(noisy, threshold)
        dimensions.append(dim)

    fig, ax = plt.subplots()
    ax.plot(sigmas, dimensions, marker='o')
    ax.set_title("Noise Sensitivity Benchmark")
    ax.set_xlabel("Zaj σ")
    ax.set_ylabel("Fraktáldimenzió")
    st.pyplot(fig)

    # 📥 CSV letöltés
    df = pd.DataFrame({'sigma': sigmas, 'fractal_dimension': dimensions})
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("🍰 Benchmark eredmény mentése CSV-ben", data=csv, file_name="fractal_benchmark.csv")

# 🚀 Streamlit felület
def run():
    st.title("🧠 Fractal Dimension Analyzer")
    st.markdown("### Box-Counting • Noise • 3D • Multifractal • Benchmark • Valós kép támogatás")

    source = st.radio("Válassz képet:", ["Beépített példa (coins)", "Kép feltöltése (.jpg, .png)"])

    if source == "Beépített példa (coins)":
        img = data.coins()
    else:
        uploaded = st.file_uploader("🧩 Tölts fel képet", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            img = io.imread(uploaded)
        else:
            st.warning("↪️ Várakozás kép feltöltésére...")
            return

    img_gray = resize(color.rgb2gray(img) if img.ndim == 3 else img, (256, 256))
    sigma = st.slider("Add Gaussian Noise (σ)", 0.0, 1.0, 0.0, 0.01)
    threshold = st.slider("Küszöbszint", 0.0, 1.0, 0.9, 0.01)

    noisy = util.random_noise(img_gray, var=sigma**2)
    dim = fractal_dimension(noisy, threshold)

    st.subheader("📸 Eredeti kép és zajos változata")
    st.image([img, noisy], caption=["Eredeti", "Zajos"], width=300)

    st.success(f"🧮 Becsült fraktáldimenzió: {dim:.4f}")

    show_benchmark = st.checkbox("📊 Benchmark: dimenzió vs zaj")
    if show_benchmark:
        benchmark_noise_response(img_gray, threshold)

app = run
