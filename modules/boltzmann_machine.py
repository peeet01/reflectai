import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# 🔢 Energiafüggvény Boltzmann-géphez
def energy(state, W, b):
    return -0.5 * np.dot(state, np.dot(W, state)) - np.dot(b, state)

# 🔁 Gibbs sampling
def gibbs_sampling(W, b, n_samples=100, n_iter=1000, T=1.0):
    n_units = W.shape[0]
    samples = []
    state = np.random.choice([0, 1], size=n_units)

    for _ in range(n_samples):
        for _ in range(n_iter):
            for i in range(n_units):
                net_input = np.dot(W[i, :], state) + b[i]
                p = 1 / (1 + np.exp(-net_input / T))
                state[i] = np.random.rand() < p
        samples.append(state.copy())

    return np.array(samples)

# 📉 Energia kiszámítása több állapotra
def compute_energies(samples, W, b):
    return np.array([energy(s, W, b) for s in samples])

# 🎯 Vizualizációhoz: bináris állapot konvertálása decimális címkére
def state_to_int(states):
    return np.dot(states, 1 << np.arange(states.shape[1])[::-1])

# 🚀 Streamlit alkalmazás
def run():
    st.title("🧠 Boltzmann-gép – Sztochasztikus neuronháló tanulás")

    st.markdown("""
A **Boltzmann-gép** egy **generatív, sztochasztikus** neurális hálózat, amely bináris állapotvektorok eloszlását tanulja meg.  
Fontos eszköz a **mélytanulás előfutáraként**, alkalmazzák **jellemzőtanulásra**, **generatív modellezésre**, illetve **neurális energiabázisú tanulásban**.

""")

    # 🔧 Paraméterek
    st.sidebar.header("🔧 Paraméterek")
    n_units = st.sidebar.slider("Neuronok száma", 2, 12, 6)
    n_samples = st.sidebar.slider("Minták száma", 10, 500, 100)
    n_iter = st.sidebar.slider("Gibbs iterációk mintánként", 10, 500, 100)
    temperature = st.sidebar.slider("Hőmérséklet (T)", 0.1, 5.0, 1.0, step=0.1)
    seed = st.sidebar.number_input("Véletlen seed", 0, 10000, 42)

    np.random.seed(seed)
    W = np.random.normal(0, 1, (n_units, n_units))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    b = np.random.normal(0, 0.5, n_units)

    st.subheader("📊 Súlymátrix és bias vektor")
    st.write("**W (kapcsolati súlyok):**")
    st.dataframe(pd.DataFrame(W))
    st.write("**b (bias):**")
    st.dataframe(pd.DataFrame(b.reshape(-1, 1), columns=["b"]))

    # 🔁 Mintagenerálás
    st.subheader("🔁 Minták generálása (Gibbs sampling)")
    samples = gibbs_sampling(W, b, n_samples=n_samples, n_iter=n_iter, T=temperature)
    energies = compute_energies(samples, W, b)
    labels = state_to_int(samples)

    # 📈 Energiaeloszlás
    st.subheader("📉 Energiaeloszlás hisztogram")
    fig1, ax1 = plt.subplots()
    ax1.hist(energies, bins=20, color='skyblue', edgecolor='black')
    ax1.set_xlabel("Energia")
    ax1.set_ylabel("Előfordulás")
    st.pyplot(fig1)

    # 🌐 3D Plotly – állapotkódok, energia, előfordulás
    st.subheader("🌐 3D energiaállapot-térkép")
    unique_labels, counts = np.unique(labels, return_counts=True)
    energies_by_label = [energy(np.array(list(np.binary_repr(i, width=n_units)), dtype=int), W, b) for i in unique_labels]

    fig3d = go.Figure(data=[go.Scatter3d(
        x=unique_labels,
        y=counts,
        z=energies_by_label,
        mode='markers',
        marker=dict(
            size=6,
            color=energies_by_label,
            colorscale='Inferno',
            colorbar=dict(title="Energia"),
            opacity=0.9
        )
    )])
    fig3d.update_layout(
        scene=dict(
            xaxis_title='Állapot (bin. → dec)',
            yaxis_title='Előfordulás',
            zaxis_title='Energia'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig3d)

    # 📁 Export
    st.subheader("📁 Adatok exportálása")
    df_export = pd.DataFrame(samples, columns=[f"v{i}" for i in range(n_units)])
    df_export["energia"] = energies
    df_export["állapot (dec)"] = labels
    csv = df_export.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ CSV letöltése", data=csv, file_name="boltzmann_samples.csv")

    # 📚 Tudományos háttér
    st.markdown("### 📚 Tudományos háttér")

    st.latex(r"""
    E(\mathbf{v}) = -\frac{1}{2} \sum_{i,j} w_{ij} v_i v_j - \sum_i b_i v_i
    """)

    st.markdown("""
A Boltzmann-gép egy bináris állapotokat kezelő, teljesen kapcsolt neuronháló, ahol minden egység:
- állapota: $v_i \in \{0, 1\}$
- frissülése: sztochasztikusan, a **Gibbs sampling** algoritmus alapján
- hőmérséklet (T) paraméter befolyásolja a "zajosságot" és a minták eloszlását

A **mintavétel** során egyensúlyi állapotokat generálunk, és azok **energia** szerinti gyakorisága megmutatja a rendszer preferenciáit.

**Alkalmazási területek:**
- Jellemzők megtanulása (unsupervised learning)
- Generatív modellek (Restricted Boltzmann Machine → Deep Belief Network)
- Statisztikai fizika és idegrendszeri szimulációk

---

""")

# Kötelező lezárás
app = run
