import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from datetime import datetime

# ✨ Modul bemutatás
st.title("🧠 Insight Learning – Belátás alapú tanulás szimuláció")

st.markdown("""
A belátás alapú tanulás olyan kognitív mechanizmus, ahol a tanulás nem fokozatos, hanem hirtelen, egyfajta "áttörés" élménnyel jár. Ebben a szimulációban egy ügynök tanulási folyamatát modellezzük egy vizuális aktivációs térképen keresztül.
""")

# 🔧 Paraméterek
st.sidebar.header("🔧 Paraméterek")
grid_size = st.sidebar.slider("🔹 Rács méret", 5, 15, 7)
episodes = st.sidebar.slider("🔄 Epizódok száma", 10, 200, 50, step=10)
max_steps = st.sidebar.slider("🛃️ Lépések epizódonként", 5, 50, 20)
insight_threshold = st.sidebar.slider("💡 Belátási szint (aktiváció)", 1, 10, 5)

# 🔹 Tanulási folyamat szimulálása
def simulate_insight_learning():
    activations = np.zeros((grid_size, grid_size))
    for _ in range(episodes):
        pos = [np.random.randint(grid_size), np.random.randint(grid_size)]
        for _ in range(max_steps):
            activations[pos[0], pos[1]] += 1
            if np.random.rand() < 0.5:
                pos[0] = min(grid_size - 1, pos[0] + 1)
            else:
                pos[1] = min(grid_size - 1, pos[1] + 1)
    return activations

activ_map = simulate_insight_learning()
smoothed = gaussian_filter(activ_map, sigma=1.2)

# 🔢 Belátás megtörtént-e
insight_happened = np.any(smoothed > insight_threshold)

# 📉 2D vizualizáció
st.subheader("📉 Aktivációs térkép (2D)")
fig2d, ax = plt.subplots(figsize=(5, 5))
cax = ax.imshow(smoothed, cmap="plasma")
ax.set_title("Neuronaktivációs térkép")
fig2d.colorbar(cax, ax=ax)
st.pyplot(fig2d)

# 🌍 3D vizualizáció
st.subheader("🌍 Aktivációs felszín (3D)")
x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
fig3d = go.Figure()

# Domborzat
fig3d.add_trace(go.Surface(z=smoothed, x=x, y=y, colorscale='Inferno', opacity=0.95))

# Aha-szint
fig3d.add_trace(go.Surface(z=np.full_like(smoothed, insight_threshold), x=x, y=y,
                            colorscale=[[0, 'white'], [1, 'white']], showscale=False, opacity=0.25,
                            name='Belátási küszöb'))

fig3d.update_layout(
    scene=dict(
        xaxis_title="Neuron X",
        yaxis_title="Neuron Y",
        zaxis_title="Aktivitás szint"
    ),
    margin=dict(l=0, r=0, t=30, b=0),
    height=500
)
st.plotly_chart(fig3d, use_container_width=True)

# 📊 Eredmény
st.subheader("📊 Eredmény")
if insight_happened:
    st.success("🎉 Belátás megtörtént! Az aktiváció elérte a küszöbszintet.")
else:
    st.warning("⚠️ Még nem történt meg belátás.")

# 📂 CSV export
st.subheader("📂 Aktivációk letöltése")
df = pd.DataFrame(smoothed, columns=[f"Y{i}" for i in range(grid_size)])
df.index = [f"X{i}" for i in range(grid_size)]
csv = df.to_csv().encode('utf-8')
st.download_button("⬇️ CSV letöltése", csv, file_name="insight_activation_map.csv")

# 📚 Tudományos háttér (LaTeX)
st.markdown("### 📚 Tudományos háttér")
st.markdown(r'''
A **belátás alapú tanulás** (insight learning) egy kognitív modell, amely szerint a problémamegoldás nem puszta próba-szerencse alapon történik,

hanem strukturált mentális átlátás és hirtelen megértés által.

A szimuláció során egy neurális rács aktivációja modellezi az ügynök gondolkodási fölyamatát.

#### ⚖️ A belátási küszöbszint:

$$
I_{\text{threshold}} = \theta
$$

ahol \(\theta\) a felhasználó által megadott kritikus szint.

#### 🔢 Aktivációs függvény:

$$
A_{ij}^{(t+1)} = A_{ij}^{(t)} + \delta_{ij}
$$

ahol \( \delta_{ij} \in \{0,1\} \) egy random választott irányból származó impulzus.

#### 🔍 Használhatóság:
- Áttörés-szerű tanulási folyamatok szimulációja
- Agykutatás, mesterséges intelligencia tanulmányozása
- Oktatási stratégiák modellje

#### 🔹 Konklúzó:
Ha az aktiváció átlépi a \( I_{\text{threshold}} \) szintet, az a **belátás pillanatát** reprezentálja.
''')
