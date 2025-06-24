import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def generate_landscape(size, n_memories, noise_level):
    landscape = np.zeros((size, size))
    for _ in range(n_memories):
        cx, cy = np.random.randint(0, size, size=2)
        intensity = np.random.uniform(0.5, 1.0)
        sigma = np.random.uniform(2, size // 6)
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        blob = intensity * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        landscape += blob

    noise = noise_level * np.random.randn(size, size)
    return landscape + noise

def plot_landscape(landscape):
    plt.figure(figsize=(6, 6))
    plt.imshow(landscape, cmap='viridis', origin='lower')
    plt.colorbar(label='Memória intenzitás')
    plt.title("🧠 Memória tájkép")
    st.pyplot(plt.gcf())
    plt.clf()

def run():
    st.header("🌄 Memória tájkép (Pro)")
    st.write("Ez a modul szintetikus memóriaintenzitás-térképeket generál és jelenít meg.")

    size = st.slider("Rács mérete", 20, 200, 100)
    n_memories = st.slider("Memória pontok száma", 1, 20, 5)
    noise_level = st.slider("Zajszint", 0.0, 2.0, 0.2)

    if st.button("Térkép generálása"):
        landscape = generate_landscape(size, n_memories, noise_level)
        plot_landscape(landscape)
