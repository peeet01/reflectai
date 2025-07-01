import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Shannon-entrópia
def shannon_entropy(signal, bins):
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist[hist > 0]
    return entropy(hist, base=2)

# Rényi-entrópia
def renyi_entropy(signal, alpha, bins):
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist[hist > 0]
    if alpha == 1.0:
        return entropy(hist, base=2)
    return 1 / (1 - alpha) * np.log2(np.sum(hist ** alpha))

# Jelszimuláció
def generate_signal(kind, length, noise):
    t = np.linspace(0, 4 * np.pi, length)
    if kind == "Szinusz":
        sig = np.sin(t)
    elif kind == "Káosz (logisztikus)":
        sig = np.zeros(length)
        sig[0] = 0.5
        r = 3.9
        for i in range(1, length):
            sig[i] = r * sig[i - 1] * (1 - sig[i - 1])
    elif kind == "Fehér zaj":
        sig = np.random.randn(length)
    else:
        sig = np.zeros(length)
    return sig + np.random.normal(0, noise, size=length)

def run():
    st.title("🧠 Neurális Entrópia Idősorokon")
    st.markdown("""
    Vizsgáld meg, hogyan változik az entrópia különböző típusú időjelek esetén.
    Hasznos lehet neurális aktivitások, ESN-kimenetek, vagy szimulált EEG elemzéséhez.
    """)

    kind = st.selectbox("Jel típusa", ["Szinusz", "Káosz (logisztikus)", "Fehér zaj"])
    noise = st.slider("Zajszint (σ)", 0.0, 1.0, 0.1, step=0.01)
    signal_len = st.slider("Jel hossza", 200, 5000, 1000, step=100)
    window = st.slider("Ablakméret", 50, 500, 200, step=10)
    stride = st.slider("Lépésköz", 10, 200, 50, step=10)
    bins = st.slider("Hisztogram bin szám", 5, 100, 30)
    entropy_type = st.selectbox("Entrópia típusa", ["Shannon", "Renyi"])
    alpha = 1.0
    if entropy_type == "Renyi":
        alpha = st.slider("Renyi α paraméter", 0.1, 5.0, 2.0, step=0.1)

    sig = generate_signal(kind, signal_len, noise)
    entropies = []
    times = []

    for start in range(0, len(sig) - window, stride):
        segment = sig[start:start + window]
        if entropy_type == "Shannon":
            h = shannon_entropy(segment, bins)
        else:
            h = renyi_entropy(segment, alpha, bins)
        entropies.append(h)
        times.append(start)

    # 2D matplotlib plot
    st.subheader("📉 Entrópia időben")
    fig, ax = plt.subplots()
    ax.plot(times, entropies, marker='o')
    ax.set_xlabel("Idő (mintavételi index)")
    ax.set_ylabel("Entrópia (bit)")
    ax.set_title("Entrópia görbe")
    ax.grid(True)
    st.pyplot(fig)

    # 3D Plotly surface plot
    st.subheader("🌐 3D Entrópiafelület (idő, jeltípus, entrópia)")
    x = np.array(times)
    y = np.array([0])  # később bővíthető több jeltípusra
    z = np.expand_dims(entropies, axis=0)  # 1 sorú 2D mátrix

    fig3d = go.Figure(data=[go.Surface(z=z, x=[x], y=[y], colorscale='Viridis')])
    fig3d.update_layout(
        scene=dict(
            xaxis_title="Idő",
            yaxis_title="Típusindex",
            zaxis_title="Entrópia (bit)"
        ),
        height=500
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # Export
    st.subheader("📥 Export")
    df = pd.DataFrame({"index": times, "entropy": entropies})
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Letöltés CSV-ben", data=csv, file_name="entropy_time_series.csv")

    # Tudományos magyarázat
    st.markdown("""
    ### 📚 Matematikai háttér

    Az **entrópia** mértéke annak, mennyire véletlenszerű, rendezetlen vagy információban gazdag egy jel.

    - **Shannon-entrópia** az információelmélet alapfogalma. Ha a valószínűségi eloszlás p_i, akkor:

\[
        H = -\sum_i p_i \log_2 p_i
\]

      Ez kifejezi az átlagos információmennyiséget.

    - **Rényi-entrópia** általánosítás, érzékenyebb lehet extrém eseményekre vagy domináns mintákra:

\[
        H_\alpha = \frac{1}{1 - \alpha} \log_2 \sum_i p_i^\alpha
\]

      Az \alpha paraméter szabályozza a súlyozást: kis \alpha-val a ritka események dominálnak, nagy \alpha-val a gyakoriak.

    #### 🔬 Alkalmazás neurológiai rendszerekre

    - A **neurális jelek entrópiája** korrelálhat az éberségi állapottal (pl. alvás vs. ébrenlét)
    - Az entrópiacsökkenés a rendszer **szinkronizációjára** utal (pl. rohamaktivitás)
    - A Rényi-entrópia érzékenyebb lehet **lokális mintázatokra**, például tüskesűrűség, eseményritmus

    Ez az eszköz tehát nemcsak vizualizációra, hanem **kutatási célokra is alkalmas**, például:
    - ESN rejtett rétegének entrópiájának monitorozása
    - különféle jeltípusok megkülönböztetése
    - entrópiaalapú klaszterezés vagy anomália-érzékelés

    """)

# ReflectAI kompatibilis belépési pont
app = run
