import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from skimage import data, color
from skimage.transform import resize

def boxcount(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                           np.arange(0, Z.shape[1], k), axis=1)
    return len(np.where(S > 0)[0])

def fractal_dimension(Z, threshold=0.9, visualize=False):
    # Binarizálás
    Z = Z < threshold

    # Ellenőrzés
    assert len(Z.shape) == 2

    # Legnagyobb 2-hatvány méret
    p = min(Z.shape)
    n = 2**np.floor(np.log2(p))
    n = int(n)
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
        st.pyplot(fig)

    return fd

def visualize_3d(Z, threshold=0.9):
    Z_bin = Z < threshold
    x, y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    fig = go.Figure(data=[go.Surface(z=Z.astype(float), x=x, y=y, colorscale='Inferno')])
    fig.update_layout(title="3D Representation of Input", autosize=True)
    st.plotly_chart(fig)

def run():
    st.title("Fractal Dimension Analyzer")
    st.markdown("### Box-Counting Method")

    img = data.coins()
    img_gray = resize(color.rgb2gray(img) if img.ndim == 3 else img, (256, 256))

    threshold = st.slider("Threshold", 0.0, 1.0, 0.9)
    show_3d = st.checkbox("Show 3D Visualization")
    show_2d = st.checkbox("Show 2D Log-Log Plot")

    fd = fractal_dimension(img_gray, threshold=threshold, visualize=show_2d)
    st.success(f"Estimated Fractal Dimension: {fd:.4f}")

    if show_3d:
        visualize_3d(img_gray, threshold=threshold)
app = run
