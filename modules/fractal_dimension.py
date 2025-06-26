import streamlit as st import numpy as np import matplotlib.pyplot as plt import plotly.graph_objects as go from skimage import data, color from skimage.transform import resize

def boxcount(Z, k): S = np.add.reduceat( np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0), np.arange(0, Z.shape[1], k), axis=1) return len(np.where(S > 0)[0])

def fractal_dimension(Z, threshold=0.9, visualize=False): Z = Z < threshold assert len(Z.shape) == 2 p = min(Z.shape) n = 2**np.floor(np.log2(p)) n = int(n) Z = Z[:n, :n]

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

def visualize_3d(Z, threshold=0.9, colorscale='Viridis', surface_opacity=0.95): Z_bin = Z < threshold x, y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0])) fig = go.Figure(data=[ go.Surface(z=Z.astype(float), x=x, y=y, colorscale=colorscale, opacity=surface_opacity, showscale=True) ]) fig.update_layout( title="3D Surface Representation of Input", autosize=True, margin=dict(l=0, r=0, t=30, b=0), scene=dict( xaxis_title='X', yaxis_title='Y', zaxis_title='Intensity', camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)) ) ) st.plotly_chart(fig, use_container_width=True)

def run(): st.title("🧮 Fractal Dimension Analyzer") st.markdown(""" Ez a modul a Box-Counting módszert használja egy kép fraktáldimenziójának becslésére. Betölthetsz saját képet, vagy használhatod az alapértelmezett mintát. """)

# --- Bemeneti kép kiválasztás ---
uploaded = st.file_uploader("📤 Tölts fel egy képet (szürkeárnyalatos javasolt):", type=['jpg', '

