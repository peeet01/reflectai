import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from skimage import data, color, util
from skimage.transform import resize
import pandas as pd

def boxcount(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k), axis=1)
    return len(np.where(S > 0)[0])

def fractal_dimension(Z, threshold=0.9, visualize=False):
    Z = Z < threshold
    assert len(Z.shape) == 2
    p = min(Z.shape)
    n = 2 ** int(np.floor(np.log2(p)))
    Z = Z[:n, :n]
    sizes = 2 ** np.arange(int(np.log2(n)), 1, -1)
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
    x, y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    fig = go.Figure(data=[go.Surface(z=Z.astype(float), x=x, y=y, colorscale='Inferno')])
    fig.update_layout(title="3D Representation", autosize=True)
    st.plotly_chart(fig)

def compute_multifractal_spectrum(Z, qs=np.linspace(-5, 5, 21)):
    Z = Z / np.sum(Z)
    epsilons = 2 ** np.arange(1, int(np.log2(Z.shape[0])) - 1)
    taus = []

    for q in qs:
        chi_q = []
        for e in epsilons:
            blocks = Z.reshape(Z.shape[0] // e, e, Z.shape[1] // e, e).sum(axis=(1, 3))
            P = blocks[blocks > 0].flatten()
            if q == 1:
                chi_q.append(np.sum(P * np.log(P)))
            else:
                chi_q.append(np.sum(P ** q))
        if q == 1:
            taus.append(-np.polyfit(np.log(epsilons), chi_q, 1)[0])
        else:
            taus.append(np.polyfit(np.log(epsilons), np.log(chi_q), 1)[0])

    alphas = np.gradient(taus, qs)
    f_alphas = qs * alphas - np.array(taus)

    fig, ax = plt.subplots()
    ax.plot(alphas, f_alphas, 'o-')
    ax.set_xlabel("Alpha")
    ax.set_ylabel("f(Alpha)")
    ax.set_title("Multifractal Spectrum")
    st.pyplot(fig)

def benchmark_noise_response(img_gray, threshold):
    sigmas = np.linspace(0.0, 1.0, 15)
    dimensions = []

    for sigma in sigmas:
        noisy = util.random_noise(img_gray, mode='gaussian', var=sigma**2)
        fd = fractal_dimension(noisy, threshold=threshold)
        dimensions.append(fd)

    fig, ax = plt.subplots()
    ax.plot(sigmas, dimensions, 'o-')
    ax.set_xlabel("Gaussian Noise σ")
    ax.set_ylabel("Fractal Dimension")
    ax.set_title("Noise Sensitivity Benchmark")
    st.pyplot(fig)

    # Letölthető eredmény
    df = pd.DataFrame({'sigma': sigmas, 'fractal_dimension': dimensions})
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Benchmark eredmény mentése CSV-ben", data=csv, file_name="fractal_benchmark.csv")

def run():
    st.title("🧠 Fractal Dimension Analyzer")
    st.markdown("### Box-Counting • Noise • 3D • Multifractal • Benchmark • CSV export")

    img = data.coins()
    img_gray = resize(color.rgb2gray(img) if img.ndim == 3 else img, (256, 256))

    sigma = st.slider("Add Gaussian Noise (σ)", 0.0, 1.0, 0.0, 0.01)
    threshold = st.slider("Threshold", 0.0, 1.0, 0.9)
    show_3d = st.checkbox("Show 3D Visualization")
    show_2d = st.checkbox("Show 2D Log-Log Plot")
    show_multifractal = st.checkbox("Show Multifractal Spectrum")
    show_benchmark = st.checkbox("Benchmark Noise Sensitivity")

    if sigma > 0.0:
        img_gray = util.random_noise(img_gray, mode='gaussian', var=sigma**2)

    fd = fractal_dimension(img_gray, threshold=threshold, visualize=show_2d)
    st.success(f"Estimated Fractal Dimension: {fd:.4f}")

    if show_3d:
        visualize_3d(img_gray, threshold=threshold)

    if show_multifractal:
        compute_multifractal_spectrum(img_gray)

    if show_benchmark:
        benchmark_noise_response(img_gray, threshold=threshold)

# Kötelező ReflectAI integrációhoz
app = run
