import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go

# Alapbeállítás
st.set_page_config(layout="wide")
st.title("🧠 Insight Learning – Belátás alapú tanulás szimuláció")

# 📘 Bevezetés
st.markdown("""
### 🔍 Bevezetés

A **belátásos tanulás** (insight learning) során a tanuló nem fokozatosan, hanem hirtelen, egy **„aha!”** pillanatban jut el a megoldásig.

Ebben a szimulációban egy **aktivációs térképet** hozunk létre, és azt vizsgáljuk, hogy a **kritikus szint** átlépésével megtörténik-e a felismerés.
""")

# 🎛️ Paraméterek
st.sidebar.header("🎚️ Szimulációs paraméterek")
grid_size = st.sidebar.slider("Rács méret", 5, 30, 15)
episodes = st.sidebar.slider("Epizódok száma", 10, 500, 100, step=10)
theta = st.sidebar.slider("Belátási küszöb (θ)", 1, 100, 20)
sigma = st.sidebar.slider("Gauss szűrés simasága", 0.0, 3.0, 1.0)

# 🔄 Aktivációs modell szimuláció
def simulate_activation(grid, episodes, threshold):
    activation_map = np.zeros((grid, grid))
    goal = (grid // 2, grid // 2)
    insight_ep = None

    for ep in range(episodes):
        pos = [np.random.randint(grid), np.random.randint(grid)]
        for _ in range(grid):
            dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
            pos[0] = np.clip(pos[0] + dx, 0, grid - 1)
            pos[1] = np.clip(pos[1] + dy, 0, grid - 1)
            activation_map[pos[0], pos[1]] += 1

        if activation_map[goal] >= threshold and insight_ep is None:
            insight_ep = ep

    return activation_map, goal, insight_ep

# 🔢 Számítás
activation, goal_pos, insight_ep = simulate_activation(grid_size, episodes, theta)
smoothed = gaussian_filter(activation, sigma=sigma)

# 📊 2D Ábra
st.subheader("🧭 Aktivációs térkép (2D)")
fig2d, ax2d = plt.subplots()
img = ax2d.imshow(smoothed, cmap="plasma", interpolation="nearest")
ax2d.set_title("2D Aktivációs eloszlás")
plt.colorbar(img, ax=ax2d)
st.pyplot(fig2d)

# 🌐 3D Ábra
st.subheader("🌋 Aktivációs domborzat (3D)")
x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
fig3d = go.Figure(data=[go.Surface(z=smoothed, x=x, y=y, colorscale="Inferno")])
fig3d.update_layout(
    scene=dict(
        xaxis_title='Neuron X',
        yaxis_title='Neuron Y',
        zaxis_title='Aktiváció',
        zaxis=dict(nticks=6, range=[0, np.max(smoothed)+1])
    ),
    margin=dict(l=10, r=10, t=50, b=10),
    height=600
)
st.plotly_chart(fig3d, use_container_width=True)

# 🧠 Eredmény
st.subheader("📌 Belátási eredmény")
if insight_ep is not None:
    st.success(f"✅ A belátás megtörtént a(z) {insight_ep}. epizódban.")
else:
    st.warning("🚫 Nem történt belátás a megadott paraméterek mellett.")

# 💾 CSV Export
st.subheader("💾 Aktiváció exportálása CSV-be")
df = pd.DataFrame(smoothed)
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Letöltés CSV formátumban", csv, file_name="activation_map.csv")

# 🧪 Tudományos háttér
st.markdown("### 📚 Tudományos háttér")
st.latex(r"A_{i,j}^{(t+1)} = A_{i,j}^{(t)} + \Delta A")
st.latex(r"\text{Belátás akkor történik, ha } A_{goal} \geq \theta")

st.markdown("""
A szimuláció célja az **aktivációs eloszlás** modellezése, amely a tanulás során tapasztalati úton épül fel.

#### 📐 Képletek magyarázata:

- \( A_{i,j}^{(t)} \): aktiváció az (i,j) neuronban t időpillanatban
- \( \Delta A \): aktiváció növekedése egy esemény során
- \( \theta \): a belátási küszöb (kritikus érték)

#### 🎯 Használhatóság:

- Kreatív problémamegoldás modellezése
- Nemlineáris tanulási rendszerek szimulációja
- Neuronális aktivációs mintázatok értelmezése

""")
