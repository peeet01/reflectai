import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def generate_soc_signal(n, p):
    signal = np.zeros(n)
    for i in range(1, n):
        signal[i] = signal[i - 1] + np.random.choice([-1, 1], p=[1 - p, p])
    return signal

def detect_avalanches(signal, threshold):
    above = signal > threshold
    starts = np.where((~above[:-1]) & (above[1:]))[0] + 1
    ends = np.where((above[:-1]) & (~above[1:]))[0] + 1
    if starts.size == 0 or ends.size == 0:
        return np.array([])
    if ends[0] < starts[0]:
        ends = ends[1:]
    if starts.size > ends.size:
        starts = starts[:-1]
    durations = ends - starts
    return durations

def run():
    st.title("🌋 Criticality Explorer – Önszerveződő kritikusság")

    st.markdown("""
Az **önszerveződő kritikusság (SOC)** olyan rendszerek jellemzője, amelyek belső szabályaik révén  
természetes módon alakulnak át kritikus állapotba – külső beavatkozás nélkül.

Ez az app egy egyszerű **egydimenziós sztochasztikus séta** szimulációját mutatja be,  
amely képes lavinaszerű eseményeket generálni és feltárni azok eloszlását.
""")

    # 👉 Paraméterek
    st.subheader("🔧 Szimulációs paraméterek")
    n = st.slider("📏 Jel hossza", 500, 10000, 3000, step=100)
    p = st.slider("🎲 Elmozdulás valószínűsége (p)", 0.01, 1.0, 0.5, step=0.01)
    threshold = st.slider("⚠️ Küszöbszint (avalanches)", 0.1, 10.0, 3.0, step=0.1)

    # 👉 Szimuláció
    signal = generate_soc_signal(n, p)
    avalanches = detect_avalanches(signal, threshold)

    # 📊 3D vizualizáció
    st.subheader("🌐 Jel alakulása – színes 3D Plotly nézet")
    fig3d = go.Figure(data=[go.Scatter3d(
        x=np.arange(len(signal)),
        y=signal,
        z=np.zeros_like(signal),
        mode='lines',
        line=dict(color=signal, colorscale='Turbo', width=4)
    )])
    fig3d.update_layout(
        scene=dict(
            xaxis_title="Idő",
            yaxis_title="Jel",
            zaxis_title="",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=500,
        title="Önszerveződő jelalak térben"
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # 📈 Lavina histogram
    st.subheader("📈 Lavinák időtartamának eloszlása")
    if avalanches.size > 0:
        hist_df = pd.DataFrame(avalanches, columns=["Duration"])
        st.bar_chart(hist_df["Duration"].value_counts().sort_index())
    else:
        st.warning("❗ Nem észlelhető lavina a megadott küszöbszinten.")

    # 💾 CSV export
    st.subheader("⬇️ Exportálás")
    df_export = pd.DataFrame({
        "index": np.arange(len(signal)),
        "signal": signal
    })
    csv = df_export.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Jel letöltése CSV-ben", data=csv, file_name="critical_signal.csv")

    # 📘 Tudományos háttér
    st.markdown("### 📚 Tudományos háttér")
    st.markdown("""
Az **önszerveződő kritikusság** egy olyan koncepció, amely szerint bizonyos rendszerek  
külső vezérlés nélkül is képesek **kritikus állapotba** fejlődni, ahol kis zavarok is nagy következményekkel járhatnak.

#### 📌 Alapmodell:
A szimuláció egy sztochasztikus séta (random walk), ahol minden lépés valószínűségi alapon történik.  
A **lavinák** azok a szakaszok, ahol a jel átlépi a megadott küszöböt.

#### 🧠 Kritikus viselkedés jellemzői:
- **Skálafüggetlen eloszlás**: a lavinák hossza gyakran hatványfüggvény szerint oszlik el.
- **Emergens struktúra**: az egyszerű szabályok bonyolult mintázatokhoz vezetnek.
- **Stabilitás és instabilitás határán mozog**: mint pl. az agy vagy földrengések.

#### 📐 Egyszerűsített képlet:
$$
x_{t+1} = x_t + \epsilon_t \quad \text{ahol } \epsilon_t \in \{-1, 1\}
$$

A lavinák hossza az alábbi módon számolható:
$$
D = t_{\text{end}} - t_{\text{start}}
$$

#### 🔬 Alkalmazás:
- **EEG elemzés**
- **Földrengés-szimuláció**
- **Gazdasági rendszerek stabilitása**
- **Adatok skálafüggetlen szerkezete**
""")

# ReflectAI kompatibilitás
app = run
