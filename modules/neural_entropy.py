import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy

# Shannon-entrópia számítása
def shannon_entropy(signal, bins):
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist[hist > 0]
    return entropy(hist, base=2)

# Renyi-entrópia
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

    st.subheader("📉 Entrópia időben")
    fig, ax = plt.subplots()
    ax.plot(times, entropies, marker='o', label="Entrópia")
    ax.set_xlabel("Idő (mintavételi index)")
    ax.set_ylabel("Entrópia (bit)")
    ax.set_title("Entrópia görbe")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("📥 Export")
    df = pd.DataFrame({"index": times, "entropy": entropies})
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Letöltés CSV-ben", data=csv, file_name="entropy_time_series.csv")

    st.markdown("""
    ### 📚 Matematikai háttér
    Az entrópia a rendezetlenség vagy információmennyiség mértéke.

    - **Shannon-entrópia**: az információelmélet klasszikus mértéke.
    - **Renyi-entrópia**: általánosítás, érzékenyebb lehet ritka vagy gyakori mintákra (függ az \(\alpha\)-tól).

    Használható pl. neurális dinamika rendezettségének becslésére, szinkronitás elemzésre vagy ESN aktivitás vizsgálatára.
    """)

app = run

