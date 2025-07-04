import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go

# Beállítások
st.set_page_config(layout="wide")
st.title("🧠 Insight Learning – Belátás alapú tanulás szimuláció")

# 📝 Bevezetés
st.markdown("""
### 🔍 Bevezetés

A **belátásos tanulás** során a megoldás *nem fokozatosan*, hanem hirtelen ugrásszerűen jelenik meg – ezt nevezzük **„Aha” pillanatnak**.

Ez a modul egy absztrakt aktivációs térkép szimulációval modellezi azt a folyamatot, ahogy a **kritikus szint** elérése után megtörténik a felismerés.

""")

# 🎛️ Csúszkák – paraméterek
st.sidebar.header("🎚️ Szimulációs paraméterek")

grid_size = st.sidebar.slider("Rács méret", 5, 20, 10)
episodes = st.sidebar.slider("Epizódok száma", 10, 500, 100, step=10)
threshold = st.sidebar.slider("Belátás küszöbérték (θ)", 1, 100, 15)

# 💡 Szimuláció
def simulate(grid, episodes, threshold):
    activation = np.zeros((grid, grid))
    goal = (grid // 2, grid // 2)
    insight_at = None

    for ep in range(episodes):
        pos = [np.random.randint(grid), np.random.randint(grid)]
        for _ in range(grid * 2):
            dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
            pos[0] = np.clip(pos[0] + dx, 0, grid - 1)
            pos[1] = np.clip(pos[1] + dy, 0, grid - 1)
            activation[pos[0], pos[1]] += 1

            if tuple(pos) == goal and activation[goal] >= threshold and insight_at is None:
                insight_at = ep
                break

    return activation, goal, insight_at

activation_map, goal_pos, insight_ep = simulate(grid_size, episodes, threshold)

# 📊 2D Ábra
st.subheader("🖼️ 2D Aktivációs térkép")
fig, ax = plt.subplots()
ax.imshow(activation_map, cmap="plasma", interpolation="nearest")
ax.set_title("2D Aktivációs eloszlás")
st.pyplot(fig)

# 🌐 3D Ábra – Plotly
st.subheader("🌋 3D Aktivációs térkép")
x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
z = gaussian_filter(activation_map, sigma=1)
fig3d = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Inferno')])
fig3d.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Aktiváció'),
    margin=dict(l=10, r=10, t=30, b=10)
)
st.plotly_chart(fig3d, use_container_width=True)

# ✅ Eredmény kiértékelés
st.subheader("📌 Eredmény")
if insight_ep is not None:
    st.success(f"✅ A belátás megtörtént a(z) {insight_ep}. epizódban!")
else:
    st.warning("🚫 A szimuláció során nem történt belátás.")

# 💾 CSV Export
st.subheader("💾 CSV exportálás")
df = pd.DataFrame(activation_map)
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Aktivációs térkép letöltése", csv, file_name="activation_map.csv", mime="text/csv")

# 📚 Tudományos háttér
st.markdown("### 📚 Tudományos háttér")
st.latex(r"A_{i,j}^{(t+1)} = A_{i,j}^{(t)} + \Delta A")
st.latex(r"\text{Ha } A_{\text{goal}} \geq \theta \Rightarrow \text{Belátás (Insight)}")

st.markdown("""
A **belátásos tanulás** (\(Insight Learning\)) során az egyén **nem próba-szerencse alapján**, hanem hirtelen
„összeáll a kép”, amint elegendő tapasztalati aktiváció gyűlt össze a megoldási térben.

Ez egy **nemlineáris átmenet**, amely az alábbi jellemzőkkel bír:

- **A(x, y)**: aktiváció egy adott pozíción
- **θ (theta)**: kritikus aktivációs szint
- A célhely (\(goal\)) aktivációjának elérése kiváltja a felismerést.

#### 📈 Használhatóság:
- Problémamegoldás modellezése
- Nonlineáris tanulási modellek szemléltetése
- A „kritikus tömeg” elérésének dinamikus ábrázolása

#### 🧪 Következtetés:
A szimuláció alapján a **belátás egy adott epizódban, nem fokozatosan** jelenik meg, hanem egyetlen ugrással,
amit a kritikus aktivációs küszöb átlépése vált ki.

""")

# Kötelező app meghívás
app = run
