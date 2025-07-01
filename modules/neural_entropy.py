import streamlit as st import numpy as np import matplotlib.pyplot as plt import pandas as pd from scipy.stats import entropy import plotly.graph_objects as go

Shannon-entrópia számítása

def shannon_entropy(signal, bins): hist, _ = np.histogram(signal, bins=bins, density=True) hist = hist[hist > 0] return entropy(hist, base=2)

Renyi-entrópia

def renyi_entropy(signal, alpha, bins): hist, _ = np.histogram(signal, bins=bins, density=True) hist = hist[hist > 0] if alpha == 1.0: return entropy(hist, base=2) return 1 / (1 - alpha) * np.log2(np.sum(hist ** alpha))

Jelszimuláció

def generate_signal(kind, length, noise): t = np.linspace(0, 4 * np.pi, length) if kind == "Szinusz": sig = np.sin(t) elif kind == "Káosz (logisztikus)": sig = np.zeros(length) sig[0] = 0.5 r = 3.9 for i in range(1, length): sig[i] = r * sig[i - 1] * (1 - sig[i - 1]) elif kind == "Fehér zaj": sig = np.random.randn(length) else: sig = np.zeros(length) return sig + np.random.normal(0, noise, size=length)

Multi-skálás entrópia számítása

def compute_multiscale_entropy(signal, bins, entropy_type, alpha, scales): mse = [] for scale in scales: if scale <= 1: coarse = signal else: n = len(signal) // scale coarse = np.mean(signal[:n * scale].reshape((n, scale)), axis=1)

if entropy_type == "Shannon":
        h = shannon_entropy(coarse, bins)
    else:
        h = renyi_entropy(coarse, alpha, bins)
    mse.append(h)
return mse

def run(): st.title("🧐 Neurális Entrópia Idősorokon") st.markdown(""" Vizsgáld meg, hogyan változik az entrópia különböző típusú időjelek esetén. Hasznos lehet neurális aktivitások, ESN-kimenetek, vagy szimulált EEG elemzéséhez. """)

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

st.subheader("🧈 3D entrópia vizualizáció")
fig3d = go.Figure(data=[go.Scatter3d(
    x=times,
    y=[noise]*len(times),
    z=entropies,
    mode='lines+markers',
    marker=dict(size=4, color=entropies, colorscale='Viridis'),
    line=dict(width=2)
)])
fig3d.update_layout(
    scene=dict(
        xaxis_title='Idő',
        yaxis_title='Zajszint',
        zaxis_title='Entrópia'
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    title="Entrópia térbeli reprezentációja"
)
st.plotly_chart(fig3d, use_container_width=True)

# Multi-skálás entrópia megjelenítés
st.subheader("🌍 Multi-skálás entrópia")
scale_range = st.slider("Skálák (ablak aggregáció)", 1, 50, (1, 10))
scales = list(range(scale_range[0], scale_range[1] + 1))
mse_values = compute_multiscale_entropy(sig, bins, entropy_type, alpha, scales)

fig_mse, ax_mse = plt.subplots()
ax_mse.plot(scales, mse_values, marker='s')
ax_mse.set_xlabel("Skála (aggregált ablakméret)")
ax_mse.set_ylabel("Entrópia (bit)")
ax_mse.set_title("Multi-skálás entrópia")
ax_mse.grid(True)
st.pyplot(fig_mse)

# Export CSV
st.subheader("📅 Export")
df = pd.DataFrame({
    "index": times,
    "entropy": entropies,
    "signal_type": kind,
    "noise": noise,
    "alpha": alpha if entropy_type == "Renyi" else np.nan,
    "entropy_type": entropy_type
})
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Letöltés CSV-ben", data=csv, file_name="entropy_time_series.csv")

st.markdown("""
### 📚 Matematikai háttér
Az entrópia a rendezetlenség vagy információmennyiség mértéke.

- **Shannon-entrópia**: az információelélet klasszikus mértéke.
- **Renyi-entrópia**: általánosítás, érzékenyebb lehet ritka vagy gyakori mintákra (függ az \alpha-tól).
- **Multi-skálás entrópia**: az információ komplexitásának skálafüggő mérése, hasznos neurális dinamika időskáláinak feltérképezésére.
""")

app = run

