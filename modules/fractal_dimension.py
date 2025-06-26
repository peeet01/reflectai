import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from skimage import data, color
from skimage.transform import resize
from scipy.stats import entropy

def boxcount(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                           np.arange(0, Z.shape[1], k), axis=1)
    return len(np.where(S > 0)[0])

def fractal_dimension(Z, threshold=0.9, visualize=False):
    Z = Z < threshold
    assert len(Z.shape) == 2

    p = min(Z.shape)
    n = 2**np.floor(np.log2(p)).astype(int)
    Z = Z[:n, :n]

    sizes = 2**np.arange(int(np.log2(n)), 1, -1)
    counts = [boxcount(Z, size) for size in sizes]

    coeffs = np.polyfit(np.log(1.0 / sizes), np.log(counts), 1)
    fd = -coeffs[0]

    if visualize:
        fig, ax = plt.subplots()
        ax.plot(np.log(1.0 / sizes), np.log(counts), 'o', mfc='none')
        ax.plot(np.log(1.0 / sizes), np.polyval(coeffs, np.log(1.0 / sizes)), 'r')
        ax.set_title(f"Fractal Dimension = {fd:.4f}")
        ax.set_xlabel("log(1 / ε)")
        ax.set_ylabel("log(N(ε))")
        st.pyplot(fig)

    return fd

def shannon_entropy(Z):
    hist, _ = np.histogram(Z.flatten(), bins=256, range=(0, 1), density=True)
    hist = hist[hist > 0]
    return entropy(hist)

def visualize_3d(Z, threshold=0.9):
    Z_bin = Z < threshold
    x, y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    fig = go.Figure(data=[go.Surface(z=Z.astype(float), x=x, y=y, colorscale='Plasma')])
    fig.update_layout(title="3D Thresholded Surface", scene=dict(zaxis_title="Intensity"))
    st.plotly_chart(fig, use_container_width=True)

def run():
    st.title("📐 Fractal Dimension Analyzer")
    st.markdown("A fraktáldimenzióval a komplex struktúrák térkitöltő képességét mérjük.")

    img = data.coins()
    img_gray = resize(color.rgb2gray(img) if img.ndim == 3 else img, (256, 256))

    threshold = st.slider("Binarizálási küszöb", 0.0, 1.0, 0.9)
    show_2d = st.checkbox("📈 Log–Log Plot megjelenítése", value=True)
    show_3d = st.checkbox("🌀 3D felületvizualizáció")
    show_entropy = st.checkbox("🧠 Shannon-entropia kiszámítása")

    fd = fractal_dimension(img_gray, threshold=threshold, visualize=show_2d)
    st.success(f"💡 Fraktáldimenzió: {fd:.4f}")

    if show_entropy:
        ent = shannon_entropy(img_gray)
        st.info(f"📊 Shannon-entropia: {ent:.4f}")

    if show_3d:
        visualize_3d(img_gray, threshold)

# Kötelező ReflectAI integrációhoz
app = run
