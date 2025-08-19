import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# ----------------------------- Alap függvények -----------------------------

# 🔢 Energiafüggvény Boltzmann-géphez
def energy(state, W, b):
    return -0.5 * np.dot(state, np.dot(W, state)) - np.dot(b, state)

# 🔁 Gibbs sampling: több minta (mindegyik előtt n_iter keverés)
def gibbs_sampling(W, b, n_samples=100, n_iter=1000, T=1.0, seed=None, init_state=None):
    if seed is not None:
        np.random.seed(seed)

    n_units = W.shape[0]
    samples = []
    state = init_state.copy() if init_state is not None else np.random.choice([0, 1], size=n_units)

    for _ in range(n_samples):
        for _ in range(n_iter):
            for i in range(n_units):
                net_input = np.dot(W[i, :], state) + b[i]
                p = 1.0 / (1.0 + np.exp(-net_input / T))
                state[i] = (np.random.rand() < p).astype(int)
        samples.append(state.copy())

    return np.array(samples)

# 🔁 Egyetlen hosszú Gibbs-lánc rögzített T-n – energia-idő görbéhez
def gibbs_chain_energy(W, b, n_iter=2000, T=1.0, burn_in=200, seed=None, init_state=None):
    if seed is not None:
        np.random.seed(seed)

    n_units = W.shape[0]
    state = init_state.copy() if init_state is not None else np.random.choice([0, 1], size=n_units)

    energies = []
    # burn-in
    for _ in range(burn_in):
        for i in range(n_units):
            net_input = np.dot(W[i, :], state) + b[i]
            p = 1.0 / (1.0 + np.exp(-net_input / T))
            state[i] = (np.random.rand() < p).astype(int)

    # mérési szakasz
    for _ in range(n_iter):
        for i in range(n_units):
            net_input = np.dot(W[i, :], state) + b[i]
            p = 1.0 / (1.0 + np.exp(-net_input / T))
            state[i] = (np.random.rand() < p).astype(int)
        energies.append(energy(state, W, b))

    return np.array(energies)

# 📉 Energia kiszámítása több állapotra
def compute_energies(samples, W, b):
    return np.array([energy(s, W, b) for s in samples])

# 🎯 Vizualizációhoz: bináris állapot → decimális címke
def state_to_int(states):
    return np.dot(states, 1 << np.arange(states.shape[1])[::-1])

# --------------------------------- App ------------------------------------

def run():
    st.title("🧠 Boltzmann-gép – Sztochasztikus neurális háló (sandbox)")

    st.markdown("""
A **Boltzmann-gép** bináris egységekből álló, **generatív** és **sztochasztikus** neurális háló, amely energiafüggvény alapján
valószínűségi eloszlást reprezentál. Itt Gibbs-mintavételezéssel szemléltetjük a viselkedését.
""")

    # 🔧 Paraméterek
    st.sidebar.header("🔧 Paraméterek")
    n_units = st.sidebar.slider("Neuronok száma", 2, 12, 6)
    n_samples = st.sidebar.slider("Minták száma", 10, 500, 100)
    n_iter = st.sidebar.slider("Gibbs iterációk mintánként", 10, 1000, 100)
    temperature = st.sidebar.slider("Hőmérséklet (T)", 0.1, 5.0, 1.0, step=0.1)
    seed = st.sidebar.number_input("Véletlen seed", 0, 10000, 42)

    # ➕ ÚJ: energia-idő görbéhez és T-sweephez
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Energia–idő (egy hosszú lánc)")
    chain_iter = st.sidebar.slider("Iterációk (lánc)", 200, 10000, 2000, step=100)
    chain_burn = st.sidebar.slider("Burn-in (lánc)", 0, 2000, 200, step=50)
    chain_T = st.sidebar.slider("Lánc hőmérséklet (T_chain)", 0.1, 5.0, float(temperature), step=0.1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🌡️ T-sweep (több T összehasonlítása)")
    default_Ts = [0.5, 1.0, 2.0]
    Ts_text = st.sidebar.text_input("Hőmérsékletek (vesszővel)", ", ".join(map(str, default_Ts)))
    try:
        Ts = [float(t.strip()) for t in Ts_text.split(",") if t.strip()]
        Ts = [t for t in Ts if 0.05 <= t <= 10.0]
        if len(Ts) == 0:
            Ts = default_Ts
    except Exception:
        Ts = default_Ts

    # Véletlen W, b (szimmetrikus W, 0 főátló)
    np.random.seed(seed)
    W = np.random.normal(0, 1, (n_units, n_units))
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)
    b = np.random.normal(0, 0.5, n_units)

    st.subheader("📊 Súlymátrix és bias vektor")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**W (kapcsolati súlyok)**")
        st.dataframe(pd.DataFrame(W))
    with c2:
        st.write("**b (bias)**")
        st.dataframe(pd.DataFrame(b.reshape(-1, 1), columns=["b"]))

    # 🔁 Mintagenerálás (eredeti funkció megőrizve)
    st.subheader("🔁 Minták generálása (Gibbs sampling)")
    samples = gibbs_sampling(W, b, n_samples=n_samples, n_iter=n_iter, T=temperature, seed=seed)
    energies = compute_energies(samples, W, b)
    labels = state_to_int(samples)

    # 📈 Energiaeloszlás hisztogram (eredeti)
    st.subheader("📉 Energiaeloszlás hisztogram")
    fig1, ax1 = plt.subplots()
    ax1.hist(energies, bins=20, color='skyblue', edgecolor='black')
    ax1.set_xlabel("Energia")
    ax1.set_ylabel("Előfordulás")
    ax1.set_title(f"Energiaeloszlás – T = {temperature:.2f}")
    st.pyplot(fig1)

    # 🌐 3D Plotly – állapotkódok, energia, előfordulás (eredeti)
    st.subheader("🌐 3D energia–állapot térkép")
    unique_labels, counts = np.unique(labels, return_counts=True)
    # energia a label szerinti bináris vektorra (fix szélesség n_units)
    energies_by_label = []
    for i in unique_labels:
        bits = np.array(list(np.binary_repr(i, width=n_units)), dtype=int)
        energies_by_label.append(energy(bits, W, b))

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
            xaxis_title='Állapot (bin → dec)',
            yaxis_title='Előfordulás',
            zaxis_title='Energia'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=520
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # ---------------------- ÚJ: Energia–idő görbe ----------------------
    st.subheader("⏱️ Energia relaxáció időben (egy hosszú lánc)")
    energies_chain = gibbs_chain_energy(
        W, b, n_iter=chain_iter, T=chain_T, burn_in=chain_burn, seed=seed
    )
    fig_chain, axc = plt.subplots()
    axc.plot(energies_chain, lw=1.2)
    axc.set_xlabel("Iteráció")
    axc.set_ylabel("Energia")
    axc.set_title(f"Energia(t) – T = {chain_T:.2f}, burn-in = {chain_burn}")
    st.pyplot(fig_chain)

    # ----------------------- ÚJ: T-sweep összehasonlítás -----------------------
    st.subheader("🌡️ T-sweep – energiaeloszlások több T-n")
    fig_ts, axts = plt.subplots()
    # hogy gyors maradjon: kevesebb minta / iteráció T-sweephez
    sweep_samples = max(20, n_samples // 2)
    sweep_iters = max(20, n_iter // 2)

    for T in Ts:
        smp = gibbs_sampling(W, b, n_samples=sweep_samples, n_iter=sweep_iters, T=T, seed=seed)
        ens = compute_energies(smp, W, b)
        axts.hist(ens, bins=20, alpha=0.45, label=f"T={T:.2f}")

    axts.set_xlabel("Energia")
    axts.set_ylabel("Előfordulás")
    axts.set_title("Energiaeloszlások különböző hőmérsékleteken")
    axts.legend()
    st.pyplot(fig_ts)

    # 📁 Export (eredeti + kiegészített)
    st.subheader("📁 Adatok exportálása")
    df_export = pd.DataFrame(samples, columns=[f"v{i}" for i in range(n_units)])
    df_export["energia"] = energies
    df_export["állapot (dec)"] = labels

    # energia-lánc export
    df_chain = pd.DataFrame({"iter": np.arange(1, len(energies_chain) + 1), "energia": energies_chain})

    # T-sweep export (összefűzve)
    rows = []
    for T in Ts:
        smp = gibbs_sampling(W, b, n_samples=sweep_samples, n_iter=sweep_iters, T=T, seed=seed)
        ens = compute_energies(smp, W, b)
        for e in ens:
            rows.append({"T": T, "energia": e})
    df_sweep = pd.DataFrame(rows)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ Minták (CSV)", data=df_export.to_csv(index=False).encode("utf-8"),
                           file_name="boltzmann_samples.csv", mime="text/csv")
    with c2:
        st.download_button("⬇️ Energia-lánc (CSV)", data=df_chain.to_csv(index=False).encode("utf-8"),
                           file_name="boltzmann_energy_chain.csv", mime="text/csv")
    with c3:
        st.download_button("⬇️ T-sweep (CSV)", data=df_sweep.to_csv(index=False).encode("utf-8"),
                           file_name="boltzmann_Tsweep.csv", mime="text/csv")

    # 📚 Tudományos háttér (megtartva + kiegészítve)
    st.markdown("### 📚 Tudományos háttér")
    st.latex(r"""
    E(\mathbf{v}) = -\frac{1}{2} \sum_{i,j} w_{ij} v_i v_j - \sum_i b_i v_i
    """)
    st.markdown(r"""
- Bináris egységek: $v_i \in \{0,1\}$, sztochasztikus frissítés **Gibbs-mintavételezéssel**.  
- A **hőmérséklet** $T$ befolyásolja a valószínűségeket: magasabb $T$ → laposabb eloszlás, alacsonyabb $T$ → mély energia-minimumok preferálása.  
- Az **energia–idő** görbe a lánc relaxációját mutatja egy rögzített $T$ mellett (burn-in után).  
- A **T-sweep** hisztogramjai illusztrálják, hogyan változik az energiaeloszlás a hőmérséklettel.
""")

# Kötelező lezárás
app = run
