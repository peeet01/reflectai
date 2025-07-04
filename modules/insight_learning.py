import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go

# Beállítás
st.set_page_config(layout="wide")
st.title("🧠 Insight Learning – Belátás alapú tanulás vizualizáció")

# 📘 Bevezetés
st.markdown("""
A **belátásos tanulás** (insight learning) során a megoldás **nem fokozatos próbálkozásokkal**, hanem **egy hirtelen felismeréssel** jelenik meg.

A modellünk egy **aktivációs térképet** szimulál, ahol a neuronok tapasztalati úton aktiválódnak.  
Amikor az aktiváció egy **kritikus szintet (θ)** elér egy célterületen, akkor történik meg a belátás, amit az "Aha!" pillanatként vizualizálunk.
""")

# 🎛️ Paraméterek
st.sidebar.header("🎚️ Paraméterek")

grid_size = st.sidebar.slider("Rács méret (N×N)", 5, 50, 20)
episodes = st.sidebar.slider("Epizódok száma", 10, 500, 100, step=10)
theta = st.sidebar.slider("Belátási küszöb θ", 10, 100, 40)
sigma = st.sidebar.slider("Simítás (Gauss-szűrés)", 0.0, 5.0, 1.0)
seed = st.sidebar.number_input("Véletlenszám seed (opcionális)", value=42)

np.random.seed(int(seed))

# 🔁 Aktivációs szimuláció
def simulate_activation(grid, episodes, threshold):
    activation_map = np.zeros((grid, grid))
    goal = (grid // 2, grid // 2)
    insight_episode = None

    for ep in range(episodes):
        pos = [np.random.randint(grid), np.random.randint(grid)]
        for _ in range(grid):  # korlátozott lépésszám
            dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
            pos[0] = np.clip(pos[0] + dx, 0, grid - 1)
            pos[1] = np.clip(pos[1] + dy, 0, grid - 1)
            activation_map[pos[0], pos[1]] += 1

        if activation_map[goal] >= threshold and insight_episode is None:
            insight_episode = ep

    return activation_map, goal, insight_episode

# 🔢 Számítás
activation_raw, goal_pos, insight_ep = simulate_activation(grid_size, episodes, theta)
activation = gaussian_filter(activation_raw, sigma=sigma)

# 🖼️ 2D Ábra
st.subheader("🧭 Aktivációs térkép (2D)")
fig2d, ax2d = plt.subplots()
img = ax2d.imshow(activation, cmap="plasma", interpolation="nearest")
ax2d.set_title("Aktiváció eloszlás")
plt.colorbar(img, ax=ax2d)
st.pyplot(fig2d)

# 🌋 3D Vizualizáció
st.subheader("🌐 Aktivációs felszín (3D)")
x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
fig3d = go.Figure(data=[go.Surface(z=activation, x=x, y=y, colorscale="Inferno")])
fig3d.update_layout(
    scene=dict(
        xaxis_title='Neuron X',
        yaxis_title='Neuron Y',
        zaxis_title='Aktiváció',
        zaxis=dict(nticks=6, range=[0, np.max(activation) + 1])
    ),
    margin=dict(l=10, r=10, t=50, b=10),
    height=600
)
st.plotly_chart(fig3d, use_container_width=True)

# 🎯 Eredmény
st.subheader("📌 Belátás eredménye")
if insight_ep is not None:
    st.success(f"✅ A belátás megtörtént a(z) {insight_ep}. epizódban.")
else:
    st.warning("❌ Nem történt belátás a megadott epizódok alatt.")

# 📁 CSV Export
st.subheader("💾 CSV exportálás")
df = pd.DataFrame(activation)
csv_data = df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Aktivációs térkép letöltése", data=csv_data, file_name="activation_map.csv")

# 📚 Tudományos háttér
st.markdown("### 📘 Tudományos háttér")

st.latex(r"""
\text{Aktiváció: } A_{i,j}^{(t+1)} = A_{i,j}^{(t)} + \Delta A
""")
st.latex(r"""
\text{Belátás feltétele: } A_{goal} \geq \theta
""")

st.markdown("""
A neuronhálózat aktivációja minden epizódban növekszik egy véletlenszerű séta (random walk) során.

- **\( A_{i,j}^{(t)} \)**: az aktiváció a \( t \)-edik időlépésben az adott (i,j) pozíción
- **\( \Delta A \)**: aktivációs növekedés lépésenként (itt 1-gyel növeljük)
- **\( \theta \)**: a belátási küszöb – ha ezt a célpozíció aktivációja eléri, megtörténik az „aha!” pillanat

---

#### 🎓 Következtetések

- A **belátás** akkor valósul meg, amikor az aktiváció elég koncentráltan gyűlik össze egy régióban.
- A **σ** paraméterrel szabályozható a „mentális simítás”, amely befolyásolja a felismerés esélyét.
- A szimuláció **nem determinisztikus**, így ugyanazokkal a paraméterekkel is más-más eredmény adódhat.

Ez a modell egy leegyszerűsített, de illusztratív nézete a belátás alapú tanulási folyamatnak.
""")

# 🔁 Integrációs interfész
def run():
    pass

app = run
