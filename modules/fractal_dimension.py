import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from skimage import data, color, util
from skimage.transform import resize
from io import BytesIO
from PIL import Image

def boxcount(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                         np.arange(0, Z.shape[1], k), axis=1)
    return len(np.where(S > 0)[0])

def fractal_dimension(Z, threshold=0.9, visualize=False):
    Z = Z < threshold
    assert len(Z.shape) == 2
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
        ax.set_title(f"Fraktáldimenzió = {fd:.4f}")
        ax.set_xlabel("log(1/box size)")
        ax.set_ylabel("log(count)")
        st.pyplot(fig)

    return fd

def visualize_3d(Z, threshold=0.9):
    x, y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    fig = go.Figure(data=[go.Surface(z=Z.astype(float), x=x, y=y, colorscale='Inferno')])
    fig.update_layout(title="3D fraktál vizualizáció", autosize=True)
    st.plotly_chart(fig)

def compute_multifractal_spectrum(Z, qs=np.linspace(-5, 5, 21)):
    Z = Z / np.sum(Z)
    epsilons = 2**np.arange(1, int(np.log2(Z.shape[0])) - 1)
    taus = []
    for q in qs:
        chi_q = []
        for e in epsilons:
            blocks = Z.reshape(Z.shape[0]//e, e, Z.shape[1]//e, e).sum(axis=(1, 3))
            P = blocks[blocks > 0].flatten()
            if q == 1:
                chi_q.append(np.sum(P * np.log(P)))
            else:
                chi_q.append(np.sum(P**q))
        if q == 1:
            taus.append(-np.polyfit(np.log(epsilons), chi_q, 1)[0])
        else:
            taus.append(np.polyfit(np.log(epsilons), np.log(chi_q), 1)[0])
    alphas = np.gradient(taus, qs)
    f_alphas = qs * alphas - np.array(taus)

    fig, ax = plt.subplots()
    ax.plot(alphas, f_alphas, 'o-')
    ax.set_xlabel("α")
    ax.set_ylabel("f(α)")
    ax.set_title("Multifraktál spektrum")
    st.pyplot(fig)

def add_gaussian_noise(image, sigma=0.05):
    noisy = util.random_noise(image, mode='gaussian', var=sigma**2)
    return noisy

def download_image(image):
    img_rescaled = (255 * image).astype(np.uint8)
    img_pil = Image.fromarray(img_rescaled)
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    st.download_button("📥 Eredmény letöltése", data=buf.getvalue(), file_name="fraktal_input.png", mime="image/png")

def run():
    st.title("🧩 Fraktáldimenzió elemzés")
    st.markdown("Fraktáldimenzió számítás a box-counting módszerrel, 3D és multifraktál analízissel.")

    img = data.coins()
    img_gray = resize(color.rgb2gray(img) if img.ndim == 3 else img, (256, 256))

    threshold = st.slider("Binarizációs küszöb", 0.0, 1.0, 0.9)
    show_3d = st.checkbox("🌀 3D vizualizáció")
    show_2d = st.checkbox("📈 Log-log diagram")
    show_multifractal = st.checkbox("🧠 Multifraktál spektrum")
    apply_noise = st.checkbox("🌫️ Gaussian zaj hozzáadása")

    if apply_noise:
        sigma = st.slider("Zaj szórása", 0.0, 0.2, 0.05)
        img_gray = add_gaussian_noise(img_gray, sigma)

    fd = fractal_dimension(img_gray, threshold=threshold, visualize=show_2d)
    st.success(f"🔍 Becsült fraktáldimenzió: {fd:.4f}")

    if show_3d:
        visualize_3d(img_gray, threshold=threshold)

    if show_multifractal:
        compute_multifractal_spectrum(img_gray)

    download_image(img_gray)

# Kötelező ReflectAI integrációhoz
app = run
