import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

# Streamlit oldal beállítása
st.set_page_config(layout="wide")

# Bevezetés
st.title("🧠 Insight Learning – Belátás alapú tanulás szimuláció")

st.markdown("""
A **belátásos tanulás** egy olyan tanulási forma, amikor a megoldás nem fokozatos próbálkozásokkal, 
hanem egy **hirtelen felismeréssel** (az ún. *aha!* élménnyel) jelenik meg.

Ez a szimuláció egy egyszerűsített **aktivációs térképen** modellezi ezt a folyamatot. 
A neuronok aktivációja tapasztalati alapon növekszik. 
Ha a **célpozíció aktivációja** átlép egy küszöböt, akkor bekövetkezik a **belátás**.
""")

# Oldalsáv csúszkák
st.sidebar.header("🧪 Paraméterek")
grid_size = st.sidebar.slider("Rács mérete", 5, 50, 20)
episodes = st.sidebar.slider("Epizódok száma", 1, 200, 50)
activation_increment = st.sidebar.slider("Aktiváció növekedés (ΔA)", 0.1, 5.0, 1.0)
aha_threshold = st.sidebar.slider("Belátási küszöb (θ)", 1.0, 50.0, 15.0)
sigma = st.sidebar.slider("Mentális simítás (σ)", 0.0, 5.0, 1.0)

# Aktivációs térkép generálása
def generate_activation_map(grid_size, episodes, increment, sigma):
    activation_map = np.zeros((grid_size, grid_size))
    for _ in range(episodes):
        x, y = np.random.randint(0, grid_size, 2)
        activation_map[x, y] += increment
    if sigma > 0:
        activation_map = gaussian_filter(activation_map, sigma=sigma)
    return activation_map

activation_map = generate_activation_map(grid_size, episodes, activation_increment, sigma)

# 2D Aktivációs térkép
st.header("🗺️ Aktivációs térkép (2D)")
fig2d, ax = plt.subplots()
cax = ax.imshow(activation_map, cmap="plasma")
fig2d.colorbar(cax, ax=ax)
ax.set_title("Aktiváció eloszlás")
st.pyplot(fig2d)

# 3D Aktivációs felszín
st.header("🌐 Aktivációs felszín (3D)")
x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
fig3d = go.Figure(data=[go.Surface(z=activation_map, x=x, y=y, colorscale="Inferno")])
fig3d.update_layout(
    title="3D aktivációs felszín",
    scene=dict(
        xaxis_title="Neuron X",
        yaxis_title="Neuron Y",
        zaxis_title="Aktiváció"
    ),
    margin=dict(l=0, r=0, t=50, b=0)
)
st.plotly_chart(fig3d, use_container_width=True)

# Belátás vizsgálata
center = grid_size // 2
center_activation = activation_map[center, center]
insight_occurred = center_activation >= aha_threshold

st.header("📌 Belátás eredménye")
if insight_occurred:
    st.success(f"✅ Belátás megtörtént! ({center_activation:.2f} ≥ {aha_threshold})")
else:
    st.warning(f"❌ Nem történt belátás ({center_activation:.2f} < {aha_threshold})")

# CSV export
st.header("💾 CSV exportálás")
csv_data = "\n".join([",".join(map(str, row)) for row in activation_map])
st.download_button("⬇️ Aktivációs térkép letöltése", csv_data.encode("utf-8"), file_name="activation_map.csv")

# Tudományos háttér
st.header("📘 Tudományos háttér")

st.latex(r"""
\textbf{Aktiváció:} \quad A_{i,j}^{(t+1)} = A_{i,j}^{(t)} + \Delta A
""")

st.latex(r"""
\textbf{Belátás feltétele:} \quad A_{\text{goal}} \geq \theta
""")

st.markdown("""
Ez a modell egy szimplifikált szimulációja annak, hogyan alakulhat ki a belátás:

- Az aktivációs térkép egy neuronhálót reprezentál.
- Minden epizódban egy véletlen pozíció aktiválódik.
- A mentális simítás (σ) segít a felismerési "minták" megjelenésében.
- Ha a célpozíció elég sokszor aktiválódik (vagy környezete simítással), bekövetkezik a **belátás**.

---

### 🎓 Következtetések:

- A belátás akkor következik be, ha az aktiváció **koncentráltan gyűlik** össze egy adott régióban.
- A **σ** érték befolyásolja az általánosítást: nagyobb érték – nagyobb elterülés.
- A szimuláció **nem determinisztikus**: minden futás más eredményt adhat ugyanazzal a beállítással is.

Ez a szimuláció egy jó alap az **aha-jelenség** elméleti és gyakorlati vizsgálatához.

---
""")

# Kötelező illesztés
def run():
    pass

app = run
