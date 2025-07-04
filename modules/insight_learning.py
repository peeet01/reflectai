import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

# --- Beállítások ---
st.set_page_config(layout="wide")
st.title("🧠 Insight Learning – Belátás alapú tanulás modellezése")

st.markdown("""
Ez a modul a **belátásos tanulás** folyamatát modellezi. Az insight learning során az ügynök
egyszer csak *rájön*, hogyan oldjon meg egy problémát, miután elég információt halmozott fel.
""")

# --- Paraméterek ---
st.sidebar.header("🔧 Paraméterek")
grid_size = st.sidebar.slider("Rács mérete", 5, 20, 10)
episodes = st.sidebar.slider("Epizódok száma", 10, 100, 50)
max_steps = st.sidebar.slider("Maximális lépések epizódonként", 10, 100, 30)
insight_threshold = st.sidebar.slider("Belátási küszöbérték (θ)", 0.1, 5.0, 2.5, 0.1)

# --- Szimuláció ---
def simulate_insight_learning(grid_size, episodes, max_steps, threshold):
    activation_total = np.zeros((grid_size, grid_size))
    insight_happened = False
    insight_map = np.zeros_like(activation_total)

    for ep in range(episodes):
        pos = [grid_size - 1, 0]
        for _ in range(max_steps):
            activation_total[pos[0], pos[1]] += 0.1
            if pos[1] < grid_size - 1:
                pos[1] += 1
            else:
                pos[0] = max(0, pos[0] - 1)

    activation_smoothed = gaussian_filter(activation_total, sigma=1.0)
    if np.max(activation_smoothed) >= threshold:
        insight_happened = True
        insight_map = activation_smoothed >= threshold

    return activation_smoothed, insight_happened, insight_map

activation_map, insight_flag, insight_mask = simulate_insight_learning(
    grid_size, episodes, max_steps, insight_threshold
)

# --- 2D Megjelenítés ---
st.subheader("📈 Aktivációs térkép – 2D")
fig2d, ax2d = plt.subplots()
im = ax2d.imshow(activation_map, cmap="plasma")
plt.colorbar(im, ax=ax2d)
st.pyplot(fig2d)

# --- 3D Plotly Megjelenítés ---
st.subheader("🌋 Aktivációs térkép – 3D Plotly")
x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
z = activation_map

fig3d = go.Figure(data=[
    go.Surface(z=z, x=x, y=y, colorscale="Inferno", showscale=False),
    go.Scatter3d(
        x=x[insight_mask], y=y[insight_mask], z=z[insight_mask] + 0.2,
        mode='markers',
        marker=dict(size=6, color='cyan'),
        name="Belátási pontok"
    )
])
fig3d.update_layout(title="3D Aktivációs térkép", scene=dict(
    xaxis_title="X",
    yaxis_title="Y",
    zaxis_title="Aktiváció"
))
st.plotly_chart(fig3d, use_container_width=True)

# --- Eredmények ---
st.subheader("🧠 Belátás eredménye")
if insight_flag:
    st.success("✅ Belátás megtörtént! Az aktiváció átlépte a küszöbértéket.")
else:
    st.warning("❌ Még nem történt meg a belátás. Növeld az epizódok számát vagy csökkentsd a küszöböt.")

# --- CSV export ---
st.subheader("📥 CSV export")
df_export = pd.DataFrame(activation_map)
csv = df_export.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Aktivációs mátrix letöltése", data=csv, file_name="insight_activation_map.csv")

# --- Tudományos háttér ---
st.markdown("### 📘 Tudományos háttér")
st.markdown(r"""
A **belátásos tanulás** egy olyan tanulási forma, ahol a megoldás *nem fokozatosan* jön létre,
hanem egy hirtelen felismerés révén:

#### Matematikai modell:

Az aktivációs értékek egy rácsban gyűlnek össze epizódonként:

$$
A(x, y, t) = A(x, y, t-1) + \delta
$$

Ahol:
- \( A(x, y, t) \) az adott hely aktivációja \( t \)-edik időpillanatban
- \( \delta \) az aktivációs hozzájárulás

A belátás akkor történik, ha:

$$
\max_{x, y} A(x, y) \geq \theta
$$

Ahol:
- \( \theta \) a belátási küszöb

#### Használhatóság:
- Problémamegoldás modellezése
- Viselkedés és memória tanulmányozása
- Kreatív AI rendszerek szimulációja

#### Következtetés:
A modell lehetővé teszi annak vizsgálatát, hogy milyen feltételek mellett történik „aha” élmény, és miként terjed az aktiváció a memóriarendszerben.
""")

# Kötelező Streamlit hívás
app = run = lambda: None
