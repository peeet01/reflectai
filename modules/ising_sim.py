import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# 🧲 Ising-modell inicializálása
def init_lattice(N):
    return np.random.choice([-1, 1], size=(N, N))

# 🔁 Egy MC-lépés
def ising_step(lattice, beta, J=1.0):
    N = lattice.shape[0]
    for _ in range(N * N):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        spin = lattice[i, j]
        neighbors = lattice[(i+1)%N, j] + lattice[i,(j+1)%N] + lattice[(i-1)%N, j] + lattice[i, (j-1)%N]
        delta_E = 2 * J * spin * neighbors
        if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
            lattice[i, j] *= -1
    return lattice

# 🧮 Energia és mágnesezettség
def calculate_observables(lattice, J=1.0):
    N = lattice.shape[0]
    E, M = 0, np.sum(lattice)
    for i in range(N):
        for j in range(N):
            spin = lattice[i, j]
            neighbors = lattice[(i+1)%N, j] + lattice[i, (j+1)%N]
            E -= J * spin * neighbors
    return E / (N*N), M / (N*N)

# 🌡️ Szimuláció
def simulate_ising(N, beta, steps, J=1.0):
    lattice = init_lattice(N)
    energies = []
    magnetizations = []
    snapshots = []

    for step in range(steps):
        lattice = ising_step(lattice, beta, J)
        E, M = calculate_observables(lattice, J)
        energies.append(E)
        magnetizations.append(M)
        if step % (steps // 5) == 0:
            snapshots.append(np.copy(lattice))
    return lattice, energies, magnetizations, snapshots

# 📈 Vizualizációk
def plot_observables(energies, magnetizations):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(energies, label='Energia')
    ax[0].set_ylabel("E")
    ax[0].set_title("Energia alakulása")
    ax[0].grid(True)

    ax[1].plot(magnetizations, label='Mágnesezettség', color='orange')
    ax[1].set_ylabel("M")
    ax[1].set_title("Mágnesezettség alakulása")
    ax[1].set_xlabel("Lépések")
    ax[1].grid(True)

    st.pyplot(fig)

def plot_3d_lattice(lattice):
    x, y = np.meshgrid(np.arange(lattice.shape[0]), np.arange(lattice.shape[1]))
    fig = go.Figure(data=[go.Surface(z=lattice, x=x, y=y, colorscale='RdBu', showscale=False)])
    fig.update_layout(
        title="3D Spin konfiguráció",
        scene=dict(zaxis_title='Spin', xaxis_title='X', yaxis_title='Y'),
        margin=dict(l=10, r=10, b=10, t=40)
    )
    st.plotly_chart(fig, use_container_width=True)

# 🚀 Streamlit app
def run():
    st.set_page_config(layout="wide")
    st.title("🧲 Ising-modell szimuláció – 2D rácson")

    st.markdown("""
Ez a modul a klasszikus **2D Ising-modell** viselkedését szimulálja és vizualizálja különböző hőmérsékleteken.  
Feltárható a **mágnesezettségi átmenet**, a **kritikus viselkedés** és a **spintextúrák** kialakulása.
""")

    st.sidebar.header("🧪 Paraméterek")
    N = st.sidebar.slider("Rács mérete (NxN)", 10, 100, 30)
    beta = st.sidebar.slider("Inverz hőmérséklet (β)", 0.1, 1.0, 0.4, 0.01)
    steps = st.sidebar.slider("MC-lépések száma", 100, 5000, 1000, step=100)
    J = st.sidebar.slider("Csatolási állandó (J)", 0.1, 5.0, 1.0, 0.1)

    # 🔍 Kritikus viselkedés detektálás
    beta_c = 0.44  # Kritikus pont J=1 esetén
    if abs(beta - beta_c) / beta_c < 0.05:
        st.warning(f"⚠️ A β={beta:.2f} érték közel van a kritikus ponthoz (βc≈{beta_c}). Kritikus viselkedés várható!")

    if st.button("▶️ Szimuláció indítása"):
        st.info("⏳ Szimuláció fut...")
        lattice, E, M, snapshots = simulate_ising(N, beta, steps, J)
        st.success("✅ Kész!")

        st.subheader("📉 Energetikai és mágneses lefutás")
        plot_observables(E, M)

        st.subheader("🌐 Záró spinállapot – 3D")
        plot_3d_lattice(lattice)

        st.subheader("🧊 Snapshot-ok 2D-ben")
        cols = st.columns(len(snapshots))
        for i, snap in enumerate(snapshots):
            fig, ax = plt.subplots()
            ax.imshow(snap, cmap='coolwarm')
            ax.set_title(f"Lépés {int((i+1)*(steps/5))}")
            ax.axis('off')
            cols[i].pyplot(fig)

        st.subheader("📥 CSV export")
        df = pd.DataFrame({"step": list(range(steps)), "energy": E, "magnetization": M})
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Letöltés CSV-ben", data=csv, file_name="ising_observables.csv")

        st.markdown("---")
        st.subheader("📘 Tudományos háttér")

        st.markdown(r"""
A **2D Ising-modell** egy egyszerű, de mély fizikai jelentéssel bíró rácsmodell, amely bináris spinekből áll ($s_{i,j} = \pm 1$).  
A rendszer Hamilton-függvénye:

$$
H = -J \sum_{\langle i,j \rangle} s_i s_j
$$

- $J$: csatolási állandó (pozitív – ferromágneses)
- $s_i$: az $i$-edik spin értéke
- $\langle i,j \rangle$: szomszédos spinpárok

A szimuláció **Metropolis-algoritmust** használ a hőmérséklet szerinti eloszlás közelítésére.  
A **mágnesezettség**:

$$
M = \frac{1}{N^2} \sum_{i,j} s_{i,j}
$$

A **belső energia**:

$$
E = -\frac{J}{N^2} \sum_{\langle i,j \rangle} s_i s_j
$$

### 🔥 Fázisátmenet

- Alacsony hőmérsékleten a spinek **rendezetten** állnak be (magas $|M|$)
- Magas hőmérsékleten **kaotikus** állapot alakul ki (kis $|M|$)
- Kritikus pont: $T_c \approx 2.27$ (vagy $\beta_c \approx 0.44$ ha $J=1$)

Ez az egyszerű modell képes leírni **fázisátmeneteket, kritikus viselkedést és rendezettségi dinamika** kialakulását.

### 📌 Konklúzió

- Az Ising-modell jó alapja a **sztochasztikus rendszerek**, **neurális dinamikák** vagy akár **szociális hálózatok** szimulációjának.
- A **hőmérséklet szabályozásával** jól megfigyelhető a spontán rend kialakulása.
        """)

# ReflectAI-kompatibilitás
app = run
