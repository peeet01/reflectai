import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Shannon-entrópia számítása
def shannon_entropy(signal, bins):
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist[hist > 0]
    return entropy(hist, base=2)

# Renyi-entrópia számítása
def renyi_entropy(signal, alpha, bins):
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist[hist > 0]
    if alpha == 1.0:
        return entropy(hist, base=2)
    return 1 / (1 - alpha) * np.log2(np.sum(hist ** alpha))

# Szintetikus jel generálása
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

def app():
    st.title("🧠 Neurális Entrópia Idősorokon")

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

    # 2D plot
    st.subheader("📉 Entrópia időben")
    fig, ax = plt.subplots()
    ax.plot(times, entropies, marker='o')
    ax.set_xlabel("Idő (mintavételi index)")
    ax.set_ylabel("Entrópia (bit)")
    ax.set_title("Entrópia görbe")
    ax.grid(True)
    st.pyplot(fig)

    # 3D plot
    st.subheader("🌐 3D entrópiafelület")
    x = np.array(times)
    y = np.zeros_like(x)
    z = np.array(entropies)

    fig3d = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='lines+markers',
        marker=dict(size=4, color=z, colorscale='Viridis'),
        line=dict(color='blue', width=2)
    )])
    fig3d.update_layout(scene=dict(
        xaxis_title="Idő",
        yaxis_title="Típusindex",
        zaxis_title="Entrópia (bit)"
    ))
    st.plotly_chart(fig3d, use_container_width=True)

    # Export CSV
    st.subheader("📥 Export")
    df = pd.DataFrame({"index": times, "entropy": entropies})
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Letöltés CSV-ben", data=csv, file_name="entropy_time_series.csv")

    # Matematikai háttér (egységes latex)
    st.markdown(r"""
    ### 📚 Matematikai háttér

    Az **entrópia** az információelmélet alapfogalma, amely a rendezetlenség, bizonytalanság vagy információmennyiség mértéke egy rendszerben. Matematikailag az entrópiát valószínűségi eloszlásokhoz rendelhetjük.

    #### Shannon-entrópia

    A Shannon-entrópia az átlagos információtartalmat méri:

    \[
    H = -\sum_i p_i \log_2 p_i
    \]

    ahol p_i az adott állapot előfordulásának valószínűsége. Ez az entrópia érték 0 (teljes rendezettség) és \log_2 N (egyenletes eloszlás) között változik.

    #### Rényi-entrópia

    A Rényi-entrópia általánosítja a Shannon-entrópiát egy valós paraméter, \alpha segítségével:

    \[
    H_\alpha = \frac{1}{1 - \alpha} \log_2 \left( \sum_i p_i^\alpha \right), \quad \alpha > 0, \ \alpha \ne 1
    \]

    - Ha \alpha \to 1, akkor H_\alpha \to H (Shannon-entrópia).
    - Kis \alpha esetén érzékenyebb a ritka eseményekre, nagy \alpha esetén a domináns mintázatokra.

    ---

    ### 🧠 Alkalmazás az idegtudományban

    Az entrópia mérése fontos eszköz az **agyi aktivitás komplexitásának** vagy **rendezettségének** vizsgálatában.

    - A **neuronális jelek entrópiája** összefügg az információfeldolgozás hatékonyságával.
    - **Alacsony entrópia** → nagyfokú szinkronitás, ismétlődő mintázatok (pl. epilepsziás rohamokban).
    - **Magas entrópia** → változatos, adaptív dinamika (pl. tanulás, figyelem során).
    - A Rényi-entrópia paramétereivel finomhangolható a ritka vagy domináns minták érzékenysége.

    Ez az entrópiaalapú elemzés különösen hasznos **Echo State Network (ESN)** kimenetek, EEG szignálok vagy bármilyen nemlineáris időjel komplexitásának kvantifikálására.
    """)
    # Fontos: csak akkor fut le, ha lokálisan teszteled (a deployhoz NE írd be)
    # if __name__ == "__main__":
    #     app()
