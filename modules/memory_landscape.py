import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import io

# --- MEMÓRIA TÁJKÉP GENERÁLÁS ---
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
    fig, ax = plt.subplots()
    im = ax.imshow(landscape, cmap='viridis', origin='lower')
    plt.colorbar(im, ax=ax, label='Memória intenzitás')
    ax.set_title("🧠 Memória tájkép (2D)")
    st.pyplot(fig)

def plot_landscape_3d(landscape):
    x, y = np.meshgrid(np.arange(landscape.shape[1]), np.arange(landscape.shape[0]))
    fig = go.Figure(data=[go.Surface(z=landscape, x=x, y=y, colorscale='Viridis')])
    fig.update_layout(
        title="🧠 Memória tájkép (3D)",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Intenzitás"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- HOPFIELD HÁLÓ ---
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
    fig, ax = plt.subplots()
    ax.imshow(pattern.reshape((dim, dim)), cmap="binary")
    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig)

# --- APP ---
def run():
    st.title("🌄 Memória Tájkép és Hopfield-háló")

    st.markdown("""
Ez a modul két fontos neurális elvet mutat be vizuálisan:
- **Memória tájkép**: szintetikus memóriaintenzitás-térkép, mely az emlékezeti nyomok eloszlását szemlélteti.
- **Hopfield-háló**: asszociatív memória modell, amely képes minták visszahívására zajos bemenetek alapján.
""")

    # --- Tájkép paraméterek ---
    size = st.slider("🗺️ Rács mérete", 20, 200, 100)
    n_memories = st.slider("🧠 Memória pontok száma", 1, 20, 5)
    noise_level = st.slider("🌫️ Zajszint", 0.0, 2.0, 0.2)

    if st.button("🎲 Térkép generálása"):
        landscape = generate_landscape(size, n_memories, noise_level)
        st.subheader("🖼️ 2D Vizualizáció")
        plot_landscape(landscape)
        st.subheader("🌐 3D Vizualizáció")
        plot_landscape_3d(landscape)

        df = pd.DataFrame(landscape)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Térkép letöltése CSV-ben", data=csv, file_name="memory_landscape.csv")

    # --- Hopfield ---
    st.markdown("---")
    st.subheader("🔁 Hopfield-háló – Asszociatív memória")

    if st.checkbox("🧠 Aktiváld a Hopfield hálót"):
        dim = st.slider("📐 Minta dimenzió (NxN)", 5, 20, 10)
        pattern_size = dim * dim

        pattern_type = st.selectbox("📂 Minta típusa", ["Random 1", "Sakktábla", "Függőleges csíkok"])
        if pattern_type == "Random 1":
            base_patterns = np.array([
                np.random.choice([-1, 1], size=pattern_size),
                np.random.choice([-1, 1], size=pattern_size)
            ])
        elif pattern_type == "Sakktábla":
            base_patterns = np.array([
                np.tile([1, -1], pattern_size // 2),
                np.tile([-1, 1], pattern_size // 2)
            ])
        else:
            pattern = np.ones((dim, dim))
            pattern[:, ::2] = -1
            base_patterns = np.array([pattern.flatten(), -pattern.flatten()])

        st.markdown("**🧩 Eredeti minták:**")
        for i, pat in enumerate(base_patterns):
            display_pattern(pat, f"Minta {i + 1}")

        W = hopfield_train(base_patterns)

        idx = np.random.randint(len(base_patterns))
        original = base_patterns[idx]
        noise_level = st.slider("🌀 Zajszint (%)", 0, 100, 30)
        noise = np.random.rand(pattern_size) < (noise_level / 100)
        noisy_input = original.copy()
        noisy_input[noise] *= -1

        st.markdown("**📥 Zajos bemenet:**")
        display_pattern(noisy_input, "Zajos bemenet")

        recalled = hopfield_recall(noisy_input, W)
        st.markdown("**📤 Visszahívott minta:**")
        display_pattern(recalled, "Hopfield kimenet")

        with io.BytesIO() as buffer:
            np.savez(buffer, patterns=base_patterns, weights=W)
            st.download_button("💾 Minták és súlymátrix letöltése", data=buffer.getvalue(), file_name="hopfield_data.npz")

    # --- Tudományos háttér ---
    st.markdown("### 📚 Tudományos háttér")
    st.markdown(r"""
A **memória tájkép** egy absztrakt vizuális modell, amely reprezentálja, hogy a memória-nyomok hogyan oszlanak el egy adott térben.  
A csúcsok az intenzív emlékeket, a völgyek az elhalványultakat jelképezik.

---

A **Hopfield-háló** egy bináris, teljesen összekapcsolt neurális hálózat, amely képes a tanult mintákat visszahívni még zajos bemenetek alapján is.

#### Súlymátrix tanulás:

$$
W_{ij} = \sum_{\mu=1}^{P} \xi_i^\mu \cdot \xi_j^\mu, \quad W_{ii} = 0
$$

#### Állapotfrissítés:

$$
s_i^{t+1} = \text{sign} \left( \sum_{j} W_{ij} \cdot s_j^t \right)
$$

Ez a folyamat egy energiafüggvény minimumának keresésén alapul, ami biztosítja a háló stabil konvergenciáját a legközelebbi tanult mintára.
""")

# ReflectAI kompatibilitás
app = run
