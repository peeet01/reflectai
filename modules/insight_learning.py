import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

st.set_page_config(layout="wide")

def run():
    st.title("🧠 Insight Learning – Belátás alapú tanulás szimuláció")

    st.markdown("""
    A **belátásos tanulás** során a megoldás nem fokozatos próbálkozásokkal,
    hanem egy **hirtelen felismeréssel** (aha-élmény) jelenik meg.

    Ez a szimuláció egy **aktivációs térképen** modellezi a tapasztalati tanulást,
    ahol az aktiváció egy adott küszöb felett **belátást** vált ki.
    """)

    # Csúszkák
    st.sidebar.header("🔧 Paraméterek")
    grid_size = st.sidebar.slider("Rács mérete", 5, 50, 20)
    episodes = st.sidebar.slider("Epizódok száma", 1, 200, 50)
    activation_increment = st.sidebar.slider("Aktiváció növekedés (ΔA)", 0.1, 5.0, 1.0)
    aha_threshold = st.sidebar.slider("Belátási küszöb (θ)", 1.0, 20.0, 10.0)
    sigma = st.sidebar.slider("Mentális simítás (σ)", 0.0, 3.0, 1.0)

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

    # 2D térkép
    st.header("🗺️ Aktivációs térkép (2D)")
    fig2d, ax2d = plt.subplots()
    cax = ax2d.imshow(activation_map, cmap="plasma")
    fig2d.colorbar(cax, ax=ax2d, label="Aktiváció")
    ax2d.set_title("Aktiváció eloszlás")
    st.pyplot(fig2d)

    # 3D felszín
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

    # Belátás
    st.header("📌 Belátás eredménye")
    if insight_occurred:
        st.success(f"✅ Belátás megtörtént! A középpont aktivációja: {center_activation:.2f} ≥ {aha_threshold}")
    else:
        st.warning(f"❌ Nem történt belátás. A középpont aktivációja: {center_activation:.2f} < {aha_threshold}")

    # CSV export
    st.header("💾 CSV exportálás")
    csv = "\n".join([",".join(map(str, row)) for row in activation_map])
    st.download_button("⬇️ Aktivációs térkép letöltése", csv.encode("utf-8"), file_name="activation_map.csv")

    st.markdown("### 📘 Tudományos háttér")

    st.markdown("**Lorenz-egyenletek**")
    st.latex(r"""
    \begin{aligned}
    \frac{dx}{dt} &= \sigma\,(y - x),\\
    \frac{dy}{dt} &= x\,(\rho - z) - y,\\
    \frac{dz}{dt} &= x\,y - \beta\,z.
    \end{aligned}
    """)

    st.markdown("A rendszer determinisztikus, de **kaotikus**, ezért a hosszú távú előrejelzés erősen érzékeny a kezdeti feltételekre.")

    st.markdown("---")
    st.markdown("**MLP-alapú előrejelzés (csúszóablak)**")
    st.latex(r"""
    \hat{x}_{t+1} = f\!\big(x_t,\, x_{t-1},\, \dots,\, x_{t-w+1}\big)
    """)
    st.markdown(r"ahol \(w\) az ablakméret; a bemenetek a múltbeli minták, a kimenet a következő \(x\) érték.")

    st.markdown("**Teljesítménymutatók**")
    st.markdown("- Determinációs együttható \(R^2\)")
    st.latex(r"""
    R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
    """)
    st.markdown("- Átlagos négyzetes hiba (MSE)")
    st.latex(r"""
    \mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}\big(y_i - \hat{y}_i\big)^2
    """)

app = run
