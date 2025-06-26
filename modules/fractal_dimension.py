import streamlit as st import numpy as np import matplotlib.pyplot as plt from PIL import Image from skimage.color import rgb2gray from skimage import data import io import math

def box_count(img, box_size): h, w = img.shape count = 0 for y in range(0, h, box_size): for x in range(0, w, box_size): if np.any(img[y:y + box_size, x:x + box_size]): count += 1 return count

def fractal_dimension(img, box_sizes): counts = [] for size in box_sizes: counts.append(box_count(img, size)) coeffs = np.polyfit(np.log(box_sizes), np.log(counts), 1) return -coeffs[0], counts

def run(): st.title("🌌 Fraktáldimenzió Vizsgálat") st.markdown("Interaktív fraktál analízis box-counting módszerrel.")

option = st.radio("Forrás kiválasztása:", ["Minta fraktál", "Saját kép feltöltése"])

if option == "Minta fraktál":
    image = rgb2gray(data.coins())  # Sierpinski helyett coin minta
    image = image < 0.5
else:
    uploaded = st.file_uploader("Tölts fel fekete-fehér képet", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("L").resize((256, 256))
        image = np.array(image) < 128
    else:
        st.stop()

st.image(image.astype(float), caption="Elemzett bináris kép", width=300)

st.markdown("---")
st.markdown("## 📐 Fraktáldimenzió számítása")
sizes = np.array([2, 4, 8, 16, 32, 64])
D, counts = fractal_dimension(image, sizes)

fig, ax = plt.subplots()
ax.plot(np.log(sizes), np.log(counts), 'o-', label=f'D ≈ {D:.2f}')
ax.set_xlabel("log(Box size)")
ax.set_ylabel("log(Count)")
ax.set_title("Fraktáldimenzió (box-counting)")
ax.legend()
st.pyplot(fig)

st.markdown("---")
st.markdown("### 🧠 Matematikai háttér")
st.latex(r"N(s) \sim s^{-D} \Rightarrow D = -\frac{\log N(s)}{\log s}")
st.markdown("Ahol $s$ a dobozméret, $N(s)$ a lefedő dobozok száma.")

ReflectAI modulkompatibilitás

app = run

