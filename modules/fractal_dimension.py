import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from skimage import data, color
from skimage.transform import resize
from skimage.util import random_noise

# === Box-counting ===
def boxcount(Z, k):
    S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
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

# === 3D Vizualizáció ===
def visualize_3d(Z, threshold=0.9):
    x, y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    fig = go.Figure(data=[go.Surface(z=Z.astype(float), x=x, y=y, colorscale='Inferno')])
    fig.update_layout(title="3D Representation of Input", autosize=True)
    st.plotly_chart(fig)

# === Multifraktál spektrum ===
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

# === Benchmark fraktál képeken ===
def generate_sierpinski(size=256):
    img = np.ones((size, size))
    for i in range(size):
        for j in range(size):
            x, y = i, j
            while x > 0 or y > 0:
                if x % 2 == 1 and y % 2 == 1:
                    img[i, j] = 0
                    break
                x //= 2
                y //= 2
    return img

def generate_cantor(size=256):
    img = np.ones((size, size))
    for row in range(size):
        line = np.ones(size)
        def cantor(l, r):
            if r - l < 3:
                return
            third = (r - l) // 3
            line[l + third: r - third] = 0
            cantor(l, l + third)
            cantor(r - third, r)
        cantor(0, size)
        img[row] = line
    return img

def generate_koch_snowflake(size=256):
    x = np.linspace(0, 4 * np.pi, size)
    y = np.sin(x) * 0.5 + 0.5
    img = np.tile(y, (size, 1))
    return img

def run_benchmark():
    st.subheader("📏 Benchmark ismert fraktálokon")
    fraktalok = {
        "Sierpinski háromszög": (generate_sierpinski(), np.log(3)/np.log(2)),
        "Cantor-sáv": (generate_cantor(), np.log(2)/np.log(3)),
        "Koch-hópehely (szimulált)": (generate_koch_snowflake(), np.log(4)/np.log(3))
    }

    for nev, (img, expected_fd) in fraktalok.items():
        fd = fractal_dimension(img, threshold=0.5)
        st.markdown(f"**{nev}**")
        st.image(img, caption=f"Fractal Dimension ≈ {fd:.4f} | Expected ≈ {expected_fd:.4f}", use_container_width=True)
        st.markdown(f"- Hiba: {abs(fd - expected_fd):.4f}")

# === Fő futtató ===
def run():
    st.title("🧠 Fractal Dimension Analyzer")
    st.markdown("### Box-Counting Method with Extensions")

    img = data.coins()
    img_gray = resize(color.rgb2gray(img) if img.ndim == 3 else img, (256, 256))

    # Interakciók
    st.sidebar.header("Beállítások")
    threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.9)
    sigma = st.sidebar.slider("Zaj szórása", 0.0, 0.2, 0.0)
    apply_noise = st.sidebar.checkbox("Apply Gaussian Noise")
    show_3d = st.sidebar.checkbox("Show 3D Visualization")
    show_2d = st.sidebar.checkbox("Show 2D Log-Log Plot")
    show_multifractal = st.sidebar.checkbox("Show Multifractal Spectrum")
    show_benchmark = st.sidebar.checkbox("Run Benchmark")

    if apply_noise and sigma > 0:
        img_gray = random_noise(img_gray, mode='gaussian', var=sigma**2)
        st.image(img_gray, caption=f"Zajos kép (σ={sigma:.2f})", use_container_width=True)

    fd = fractal_dimension(img_gray, threshold=threshold, visualize=show_2d)
    st.success(f"Fractal Dimension ≈ {fd:.4f}")

    if show_3d:
        visualize_3d(img_gray, threshold=threshold)
    if show_multifractal:
        compute_multifractal_spectrum(img_gray)
    if show_benchmark:
        run_benchmark()

app = run
