import numpy as np import matplotlib.pyplot as plt import plotly.graph_objects as go import streamlit as st from skimage import data, color, util from skimage.transform import resize

def boxcount(Z, k): S = np.add.reduceat( np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0), np.arange(0, Z.shape[1], k), axis=1) return len(np.where(S > 0)[0])

def fractal_dimension(Z, threshold=0.9, visualize=False): Z = Z < threshold assert len(Z.shape) == 2 p = min(Z.shape) n = 2np.floor(np.log2(p)) n = int(n) Z = Z[:n, :n] sizes = 2np.arange(int(np.log2(n)), 1, -1) counts = [boxcount(Z, size) for size in sizes] coeffs = np.polyfit(np.log(1.0 / sizes), np.log(counts), 1) fd = -coeffs[0]

if visualize:
    fig, ax = plt.subplots()
    ax.plot(np.log(1.0 / sizes), np.log(counts), 'o', mfc='none')
    ax.plot(np.log(1.0 / sizes), np.polyval(coeffs, np.log(1.0 / sizes)), 'r')
    ax.set_title(f"Fractal Dimension = {fd:.4f}")
    st.pyplot(fig)

return fd

def visualize_3d(Z, threshold=0.9): x, y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0])) fig = go.Figure(data=[go.Surface(z=Z.astype(float), x=x, y=y, colorscale='Inferno')]) fig.update_layout(title="3D Representation of Input", autosize=True) st.plotly_chart(fig)

def compute_multifractal_spectrum(Z, qs=np.linspace(-5, 5, 21)): Z = Z / np.sum(Z) epsilons = 2**np.arange(1, int(np.log2(Z.shape[0])) - 1) taus = []

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
        taus.append(-np.polyfit(np

