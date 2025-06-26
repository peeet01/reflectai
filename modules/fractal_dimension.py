import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu
import nibabel as nib
import tempfile
import os

def boxcount(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
    return len(np.where(S > 0)[0])

def compute_2d_fractal_dimension(Z):
    Z = Z > threshold_otsu(Z)
    p = min(Z.shape)
    n = 2**np.floor(np.log2(p)).astype(int)
    sizes = 2**np.arange(int(np.log2(n)), 1, -1)
    counts = [boxcount(Z, size) for size in sizes]
    coeffs = np.polyfit(np.log(1/sizes), np.log(counts), 1)
    return -coeffs[0], sizes, counts

def compute_3d_fractal_dimension(volume):
    """Egyszerű 3D box-counting fraktáldimenzió"""
    from scipy.ndimage import zoom

    vol = (volume > volume.mean()).astype(int)
    original_shape = vol.shape
    N = min(original_shape)
    M = 2**int(np.floor(np.log2(N)))
    vol = zoom(vol, (M/original_shape[0], M/original_shape[1], M/original_shape[2]), order=0)

    sizes = 2**np.arange(int(np.log2(M)), 1, -1)
    counts = []

    for size in sizes:
        new_shape = (vol.shape[0]//size, vol.shape[1]//size, vol.shape[2]//size)
        reduced = vol.reshape(new_shape[0], size, new_shape[1], size, new_shape[2], size)
        reduced = reduced.max(axis=(1,3,5))
        counts.append(np.sum(reduced > 0))

    coeffs = np.polyfit(np.log(1/sizes), np.log(counts), 1)
    return -coeffs[0], sizes, counts

def extract_roi(image, center, size):
    x, y = center
    half = size // 2
    x1, x2 = max(0, x - half), min(image.shape[1], x + half)
    y1, y2 = max(0, y - half), min(image.shape[0], y + half)
    return image[y1:y2, x1:x2]

def run():
    st.title("🧠 Fraktáldimenzió Analízis – 2D & 3D")

    mode = st.radio("Mód kiválasztása:", ["2D ROI kép", "3D MRI térfogat"])
    
    if mode == "2D ROI kép":
        uploaded = st.file_uploader("Tölts fel agyképet (jpg/png)", type=["jpg", "png"])
        if uploaded:
            img = io.imread(uploaded, as_gray=True)
            img = img_as_ubyte(img)
            st.image(img, caption="Eredeti kép", use_column_width=True)

            x = st.slider("Középpont X", 0, img.shape[1]-1, img.shape[1]//2)
            y = st.slider("Középpont Y", 0, img.shape[0]-1, img.shape[0]//2)
            size = st.slider("ROI méret", 32, 256, 128, 16)

            roi = extract_roi(img, (x, y), size)
            if roi.size == 0:
                st.error("A kivágás üres")
                return

            D, sizes, counts = compute_2d_fractal_dimension(roi)
            st.success(f"🧮 Fraktáldimenzió (2D): **{D:.4f}**")
            st.image(roi, caption="ROI kép", clamp=True)

            fig, ax = plt.subplots()
            ax.plot(np.log(1/sizes), np.log(counts), 'o-')
            ax.set_title("Box-counting skála – 2D")
            st.pyplot(fig)

    elif mode == "3D MRI térfogat":
        uploaded = st.file_uploader("Tölts fel NIfTI fájlt (.nii/.nii.gz)", type=["nii", "nii.gz"])
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
                tmp.write(uploaded.read())
                img = nib.load(tmp.name)
                data = img.get_fdata()
                os.unlink(tmp.name)

            mid_slice = int(data.shape[2] / 2)
            st.image(data[:, :, mid_slice], caption=f"Középső szelet (z={mid_slice})", clamp=True)

            D3, sizes, counts = compute_3d_fractal_dimension(data)
            st.success(f"🧮 Fraktáldimenzió (3D térfogat): **{D3:.4f}**")

            fig, ax = plt.subplots()
            ax.plot(np.log(1/sizes), np.log(counts), 'o-')
            ax.set_title("Box-counting skála – 3D")
            st.pyplot(fig)

# ReflectAI integráció
app = run
