import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, filters
from skimage.transform import resize
import plotly.graph_objects as go

def fractal_dimension(Z, threshold=0.9):
    assert(len(Z.shape) == 2)
    def boxcount(Z, k):
        S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                   np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where(S > 0)[0])

    Z = Z < threshold
    p = min(Z.shape)
    n = 2**np.floor(np.log2(p))
    n = int(n)
    sizes = 2**np.arange(int(np.log2(n)), 1, -1)
    counts = [boxcount(Z, size) for size in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0], sizes, counts

def plot_3d_voxel(binary_image):
    fig = go.Figure(data=go.Volume(
        x=np.repeat(np.arange(binary_image.shape[0]), binary_image.shape[1]),
        y=np.tile(np.arange(binary_image.shape[1]), binary_image.shape[0]),
        z=np.zeros(binary_image.size),
        value=binary_image.flatten().astype(float),
        opacity=0.1,
        surface_count=20,
        colorscale="Viridis"
    ))
    fig.update_layout(
        scene=dict(zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, b=0, t=0),
        title="3D Voxel Binarizált Fraktál"
    )
    st.plotly_chart(fig)

def run():
    st.title("🧩 Fraktáldimenzió Számítás")
    st.markdown("Ez a modul binarizált képből számít fraktáldimenziót box-counting módszerrel.")

    image = data.coins()
    image = resize(image, (256, 256))
    gray_image = color.rgb2gray(image) if image.ndim == 3 else image
    binary = gray_image > filters.threshold_otsu(gray_image)

    st.image(binary.astype(float), caption="Binarizált kép", use_column_width=True)

    D, sizes, counts = fractal_dimension(binary)
    st.markdown(f"**Becsült fraktáldimenzió:** `{D:.4f}`")

    fig, ax = plt.subplots()
    ax.plot(np.log(sizes), np.log(counts), 'o', label="Adatok")
    ax.plot(np.log(sizes), np.poly1d(np.polyfit(np.log(sizes), np.log(counts), 1))(np.log(sizes)), '-', label="Illesztés")
    ax.set_xlabel("log(Box size)")
    ax.set_ylabel("log(Count)")
    ax.legend()
    ax.set_title("Box-counting módszer")
    st.pyplot(fig)

    if st.checkbox("📦 3D binarizált voxel vizualizáció"):
        plot_3d_voxel(binary)

# Kötelező ReflectAI-hoz
app = run
