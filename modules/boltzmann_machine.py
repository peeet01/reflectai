import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def run():
    st.title("🌡️ Boltzmann-gép – Energián alapuló tanulás")

    st.markdown("""
    A **Boltzmann-gép** egy generatív, energián alapuló modell, amely képes **mintázatokat tárolni és rekonstruálni**.
    A tanulás alapja az energia minimalizálása és a valószínűségi aktiváció.

    Az alábbi szimuláció egy kis **Bináris Boltzmann-gépet** mutat be.
    """)

    # 🔧 Paraméterek
    n_visible = st.slider("Látható egységek száma", 2, 10, 6)
    n_hidden = st.slider("Rejtett egységek száma", 2, 10, 4)
    temperature = st.slider("Hőmérséklet (T)", 0.1, 5.0, 1.0, 0.1)
    epochs = st.slider("Epochok száma", 10, 500, 100, 10)

    np.random.seed(42)
    W = np.random.normal(0, 0.1, size=(n_visible + n_hidden, n_visible + n_hidden))
    np.fill_diagonal(W, 0)
    state = np.random.randint(0, 2, size=n_visible + n_hidden)

    # 📉 Energia számítás
    def energy(s, W):
        return -0.5 * np.dot(s, np.dot(W, s.T))

    energies = []
    snapshots = []

    for _ in range(epochs):
        for i in range(len(state)):
            net_input = np.dot(W[i], state)
            p = sigmoid(net_input / temperature)
            state[i] = np.random.rand() < p
        energies.append(energy(state, W))
        snapshots.append(state.copy())

    snapshots = np.array(snapshots)

    # 📈 Energia alakulása
    st.subheader("📉 Energiagörbe")
    fig1, ax1 = plt.subplots()
    ax1.plot(energies, color='orange')
    ax1.set_xlabel("Iteráció")
    ax1.set_ylabel("Energia")
    ax1.set_title("Rendszer energiájának alakulása")
    st.pyplot(fig1)

    # 🌐 3D Állapottér vizualizáció (redundáns térkép)
    st.subheader("🌐 3D Állapotminták vizualizációja")
    if n_visible + n_hidden >= 3:
        fig3d = go.Figure(data=[go.Scatter3d(
            x=snapshots[:, 0],
            y=snapshots[:, 1],
            z=snapshots[:, 2],
            mode='markers+lines',
            marker=dict(size=3, color=np.arange(len(snapshots)), colorscale='Viridis'),
            line=dict(width=2)
        )])
        fig3d.update_layout(
            scene=dict(xaxis_title='Bit 0', yaxis_title='Bit 1', zaxis_title='Bit 2'),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        st.plotly_chart(fig3d, use_container_width=True)
    else:
        st.info("3D megjelenítéshez legalább 3 egység szükséges.")

    # 💾 CSV export
    st.subheader("💾 Állapotminták exportálása")
    df = pd.DataFrame(snapshots, columns=[f"unit_{i}" for i in range(n_visible + n_hidden)])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Letöltés CSV-ben", data=csv, file_name="boltzmann_states.csv")

    # 📘 Tudományos háttér
    st.markdown("### 📘 Tudományos háttér")
    st.latex(r"""
    E(s) = -\frac{1}{2} s^T W s
    """)
    st.markdown("""
    - \( s \): bináris állapotvektor
    - \( W \): súlymátrix (szimmetrikus, önmagát nem kapcsolja)
    - Az alacsonyabb energiaállapotok valószínűbbek

    A tanulás célja, hogy a rendszer az **alacsony energiájú állapotokat részesítse előnyben**,  
    melyek reprezentálják az eltanult mintákat.

    **Felhasználás:**
    - Mintafelismerés
    - Dimenziócsökkentés (mély Boltzmann-hálók)
    - Generatív modellezés
    """)

app = run
