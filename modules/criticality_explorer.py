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
    st.markdown("Vizsgáld meg, hogyan jelenik meg az **önszerveződő kritikusság (SOC)** egyszerű szimulációkban.")

    # Paraméterek
    n = st.slider("Jel hossza", 500, 10000, 3000, step=100)
    p = st.slider("Elmozdulás valószínűsége (p)", 0.01, 1.0, 0.5, step=0.01)
    threshold = st.slider("Küszöbszint (avalanches)", 0.1, 10.0, 3.0, step=0.1)

    # Szimuláció
    signal = generate_soc_signal(n, p)
    avalanches = detect_avalanches(signal, threshold)

    # Plotly 3D vizualizáció
    st.subheader("📊 Jel alakulása 3D-ben")
    fig3d = go.Figure(data=[go.Scatter3d(
        x=np.arange(len(signal)),
        y=signal,
        z=np.zeros_like(signal),
        mode='lines',
        line=dict(width=3)
    )])
    fig3d.update_layout(scene=dict(
        xaxis_title="Idő",
        yaxis_title="Jel",
        zaxis_title="",
    ), margin=dict(l=0, r=0, b=0, t=30), height=500)
    st.plotly_chart(fig3d)

    # Histogram of avalanche durations
    st.subheader("📈 Avalanche időtartamok eloszlása")
    if avalanches.size > 0:
        hist_df = pd.DataFrame(avalanches, columns=["Duration"])
        st.bar_chart(hist_df["Duration"].value_counts().sort_index())
    else:
        st.warning("Nem észlelhető lavina a megadott küszöbszinten.")

    # CSV export
    st.subheader("📥 Export")
    df_export = pd.DataFrame({
        "index": np.arange(len(signal)),
        "signal": signal
    })
    csv = df_export.to_csv(index=False).encode("utf-8")
    st.download_button("Jel letöltése CSV-ben", data=csv, file_name="critical_signal.csv")

    # Tudományos háttér
    st.markdown("### 📚 Tudományos háttér")
    st.markdown("""
**Önszerveződő kritikusság (SOC)** olyan dinamikus rendszerek jellemzője, amelyek belső szabályaik alapján természetes módon állítódnak be a kritikus pontra, külső finomhangolás nélkül.

---

#### 🧠 Jelentősége:

- **Kritikus állapotban** a rendszer **skálafüggetlen** viselkedést mutat (pl. lavina-méret eloszlás hatványfüggvény szerint).
- Megfigyelhető idegrendszerben (EEG, spike sorozatok), földrengésekben, pénzügyi rendszerekben.

---

#### ⚙️ Egyszerű modell:

A szimulációban egy **egydimenziós zajos séta** jel reprodukálja a kritikus dinamika egy lehetséges formáját. A lavinák a küszöbszintet átlépő aktivitásokból származnak.

---

#### 🧪 Felhasználás az appban:

- Szimulálható, hogyan alakulnak ki **kritikus események** egyszerű szabályrendszerekből.
- Vizsgálható a **threshold** és a **p** paraméter hatása az események sűrűségére és skálájára.
- Elősegíti a **top-down** rendszerértelmezést adatvezérelt vizsgálatok előtt.
    """)

# Entry point
app = run
