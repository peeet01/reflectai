import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- EREDETI MEMÓRIA TÁJKÉP GENERÁLÁS ---

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

# --- HOPFIELD HÁLÓ KIEGÉSZÍTÉS ---

def hopfield_train(patterns):
    size = patterns.shape[1]
    W = np.zeros((size, size))
    for p in patterns:
        W += np.outer(p, p)
    np.fill_diagonal(W, 0)
    return W / patterns.shape[0]

def hopfield_recall(pattern, W, steps=5):
    s = pattern.copy()
    for _ in range(steps):
        s = np.sign(W @ s)
    return s

def display_pattern(pattern, title):
    dim = int(np.sqrt(len(pattern)))
    image = pattern.reshape((dim, dim))
    plt.imshow(image, cmap="binary")
    plt.title(title)
    plt.axis("off")
    st.pyplot(plt.gcf())
    plt.clf()

# --- STREAMLIT APP ---

def run():
    st.header("🌄 Memória tájkép (Pro)")
    st.write("Ez a modul szintetikus memóriaintenzitás-térképeket generál **és tartalmaz egy Hopfield-háló kiegészítést is.**")

    # Eredeti tájkép vezérlők
    size = st.slider("🗺️ Rács mérete", 20, 200, 100)
    n_memories = st.slider("🧠 Memória pontok száma", 1, 20, 5)
    noise_level = st.slider("🌫️ Zajszint", 0.0, 2.0, 0.2)

    if st.button("🎲 Térkép generálása"):
        landscape = generate_landscape(size, n_memories, noise_level)
        plot_landscape(landscape)

    # --- HOPFIELD BLOKK ---

    st.markdown("---")
    st.subheader("🔁 Hopfield háló – asszociatív memória")

    with st.expander("📘 Aktiváld a neurális memória visszahívást"):
        st.markdown("Taníts be egyszerű bináris mintákat (pl. 10×10 képek), és nézd meg, hogyan emlékszik vissza a háló egy zajos bemenet alapján.")

        use_hopfield = st.checkbox("🧠 Hopfield aktiválása")
        if use_hopfield:
            dim = st.slider("📐 Minta dimenzió (NxN)", 5, 20, 10)
            pattern_size = dim * dim

            # Előre definiált minták
            base_patterns = np.array([
                np.random.choice([-1, 1], size=pattern_size),
                np.tile([1, -1], pattern_size // 2)
            ])

            st.markdown("**Eredeti minták:**")
            for i, pat in enumerate(base_patterns):
                display_pattern(pat, f"Minta {i + 1}")

            W = hopfield_train(base_patterns)

            # Véletlenszerű zajos minta kiválasztása
            idx = np.random.randint(len(base_patterns))
            original = base_patterns[idx]
            noise_level = st.slider("🌀 Zajszint (%)", 0, 100, 30)
            noise = np.random.rand(pattern_size) < (noise_level / 100)
            noisy_input = original.copy()
            noisy_input[noise] *= -1

            st.markdown("**Zajos bemenet:**")
            display_pattern(noisy_input, "Zajos bemenet")

            recalled = hopfield_recall(noisy_input, W)

            st.markdown("**Visszahívott minta:**")
            display_pattern(recalled, "Hopfield kimenet")

# Kötelező belépési pont
app = run
