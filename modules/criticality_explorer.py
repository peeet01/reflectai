import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 🔁 Kritikus jel generálása
def generate_soc_signal(n, p):
    signal = np.zeros(n)
    for i in range(1, n):
        signal[i] = signal[i - 1] + np.random.choice([-1, 1], p=[1 - p, p])
    return signal

# 🔍 Lavinák detektálása
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

# 🎬 Fő alkalmazás
def run():
    st.title("🌋 Criticality Explorer – Önszerveződő kritikusság")

    st.markdown("""
Az **önszerveződő kritikusság (SOC)** egy dinamikus rendszer tulajdonsága, ahol a rendszer **külső vezérlés nélkül** beáll egy **kritikus állapotba**.  
Ebben az állapotban kis hatások **nagy, skálafüggetlen eseményeket** (pl. lavinákat) válthatnak ki – ez a **kritikus dinamika**.

Ez az app azt vizualizálja, hogyan alakulnak ki ilyen lavinaszerű események egy szimulált jel alapján.
""")

    # ⚙️ Beállítások
    n = st.slider("🔢 Jel hossza", 500, 10000, 3000, step=100)
    p = st.slider("🎲 Elmozdulás valószínűsége (p)", 0.01, 1.0, 0.5, step=0.01)
    threshold = st.slider("📏 Lavina küszöbszint", 0.1, 10.0, 3.0, step=0.1)

    signal = generate_soc_signal(n, p)
    avalanches = detect_avalanches(signal, threshold)

    # 🌄 Interaktív lavinatáj (3D Surface)
    st.subheader("🌐 Lavinatáj – Jel domborzatként")
    z = np.tile(signal, (100, 1))  # "kiterítjük" a jelet 2D felületté
    x = np.arange(z.shape[1])
    y = np.arange(z.shape[0])
    X, Y = np.meshgrid(x, y)

    fig3d = go.Figure(data=[go.Surface(z=z, x=X, y=Y, colorscale=[
        [0.0, 'rgb(0, 0, 80)'],
        [0.5, 'rgb(0, 100, 200)'],
        [0.9, 'rgb(180, 180, 255)'],
        [1.0, 'white']
    ])])

    fig3d.update_layout(
        scene=dict(
            xaxis_title="Idő",
            yaxis_title="Lavinatér",
            zaxis_title="Jel",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9))
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # 📊 Eloszlás
    st.subheader("📈 Avalanche időtartamok eloszlása")
    if avalanches.size > 0:
        hist_df = pd.DataFrame(avalanches, columns=["Duration"])
        st.bar_chart(hist_df["Duration"].value_counts().sort_index())
    else:
        st.warning("❗ Nem észlelhető lavina a megadott küszöbszinten.")

    # 💾 Export
    st.subheader("📥 CSV export")
    df_export = pd.DataFrame({
        "index": np.arange(len(signal)),
        "signal": signal
    })
    csv = df_export.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Jel letöltése CSV-ben", data=csv, file_name="critical_signal.csv")

    # 📚 Tudományos háttér
    st.markdown("### 📚 Tudományos háttér")
    st.markdown(r"""
A **kritikusság** egy olyan pont a rendszer viselkedésében, ahol kis változások is nagy átrendeződésekhez vezethetnek.  
Ha egy rendszer **önszerveződően** ér el ilyen pontot – külső beavatkozás nélkül – azt **önszerveződő kritikusságnak (SOC)** nevezzük.

---

#### 🧠 Miért érdekes?

- Kritikus rendszerek **hatványfüggvény szerinti** eloszlást mutatnak (pl. lavinák hossza, földrengések mérete).
- A rendszer viselkedése **nemlineáris** és **skálafüggetlen** – a kis és nagy események ugyanazon szabályt követik.

---

#### 📐 Szimulációs szabály:

$$
s_{t+1} = s_t + \epsilon_t, \quad \epsilon_t \in \{-1, 1\}
$$

Ahol \( \epsilon_t \) véletlenszerű lépés a jelsorozatban.  
Lavina ott keletkezik, ahol a jel átlépi a küszöbszintet.

---

#### 🧪 Mit látsz itt?

- A jel **időbeli változását**, mint **térképet** 3D-ben: „kritikus tájat” kapsz
- A **lavinák gyakoriságát és méretét** – ezek jellemzik a rendszer kritikus állapotát
- Beállíthatod, mennyire legyen „ingerlékeny” a rendszer a `p` és `küszöbszint` alapján

""")

# ReflectAI-kompatibilitás
app = run
