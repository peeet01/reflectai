import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

st.set_page_config(layout="wide")

def run():
    # Cím és leírás
    st.title("🧠 Insight Learning – Belátás alapú tanulás szimuláció")

    st.markdown("""
A **belátásos tanulás** során a megoldás nem fokozatos próbálkozásokkal,
hanem egy **hirtelen felismeréssel** (aha-élmény) jelenik meg.

Ez a szimuláció egy **aktivációs térképen** modellezi a tapasztalati tanulást,
ahol az aktiváció egy adott küszöb felett **belátást** vált ki.
""")

    # Paraméterek
    st.sidebar.header("🔧 Paraméterek")
    grid_size = st.sidebar.slider("📏 Rács mérete", 5, 50, 20)
    episodes = st.sidebar.slider("🔁 Epizódok száma", 1, 200, 50)
    activation_increment = st.sidebar.slider("⚡ Aktiváció növekedés (ΔA)", 0.1, 5.0, 1.0)
    aha_threshold = st.sidebar.slider("🎯 Belátási küszöb (θ)", 1.0, 20.0, 10.0)
    sigma = st.sidebar.slider("🧠 Mentális simítás (σ)", 0.0, 3.0, 1.0)

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
    center = grid_size // 2
    center_activation = activation_map[center, center]
    insight_occurred = center_activation >= aha_threshold

    # Aktivációs térkép – 2D
    st.header("🗺️ Aktivációs térkép (2D)")
    fig2d, ax2d = plt.subplots()
    cax = ax2d.imshow(activation_map, cmap="plasma")
    fig2d.colorbar(cax, ax=ax2d, label="Aktiváció")
    ax2d.set_title("Aktiváció eloszlás")
    st.pyplot(fig2d)

    # Aktivációs felszín – 3D
    st.header("🌐 Aktivációs felszín (3D)")
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    fig3d = go.Figure(data=[go.Surface(z=activation_map, x=x, y=y, colorscale="Inferno")])
    fig3d.update_layout(
        scene=dict(
            xaxis_title="Neuron X",
            yaxis_title="Neuron Y",
            zaxis_title="Aktiváció"
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # Eredmény
    st.header("📌 Belátás eredménye")
    if insight_occurred:
        st.success(f"✅ Belátás megtörtént! A középpont aktivációja: {center_activation:.2f} ≥ {aha_threshold}")
    else:
        st.warning(f"❌ Nem történt belátás. A középpont aktivációja: {center_activation:.2f} < {aha_threshold}")

    # CSV export
    st.header("💾 CSV exportálás")
    csv = "\n".join([",".join(map(str, row)) for row in activation_map])
    st.download_button("⬇️ Aktivációs térkép letöltése", csv.encode("utf-8"), file_name="activation_map.csv")

    # Tudományos háttér
    st.header("📘 Tudományos háttér")

    st.latex(r"""
    \text{Aktiváció:} \quad A_{i,j}^{(t+1)} = A_{i,j}^{(t)} + \Delta A
    """)
    st.latex(r"""
    \text{Belátás feltétele:} \quad A_{\text{goal}} \geq \theta
    """)

    st.markdown("""
A neuronhálózat aktivációja minden epizódban növekszik egy véletlenszerű séta során.

- $A_{i,j}^{(t)}$: aktiváció a $(i,j)$ pozíción a $t$-edik időlépésben  
- $\Delta A$: aktivációs növekedés lépésenként  
- $\theta$: belátási küszöb – ha ezt a célpozíció aktivációja eléri, megtörténik az „aha!” pillanat

---

### 🎓 Következtetések

- A belátás akkor valósul meg, amikor az aktiváció koncentráltan gyűlik össze egy régióban.
- A `σ` érték szabályozza a **mentális simítás** mértékét.
- A szimuláció **nem determinisztikus**, tehát ugyanazon paraméterekkel is más eredmény adódhat.

Ez a modell egy leegyszerűsített, de jól illusztrált nézete a belátásos tanulási folyamatnak.
""")

# Rendszerillesztéshez:
app = run
