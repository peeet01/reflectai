import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# 🔬 LIF neuronmodell szimulációja STDP-vel
def snn_simulate(I_ext=1.5, tau_m=20.0, R_m=1.0, V_th=1.0, V_reset=0.0, dt=1.0, T=200, stdp_enabled=True):
    time = np.arange(0, T, dt)
    V = np.zeros_like(time)
    spikes = np.zeros_like(time)
    w = 0.5
    V[0] = V_reset
    pre_spike_time = -np.inf
    post_spike_time = -np.inf

    for i in range(1, len(time)):
        dV = (-(V[i-1]) + R_m * I_ext * w) / tau_m
        V[i] = V[i-1] + dV * dt

        if V[i] >= V_th:
            V[i] = V_reset
            spikes[i] = 1
            post_spike_time = time[i]

            if stdp_enabled and pre_spike_time != -np.inf:
                delta_t = post_spike_time - pre_spike_time
                dw = stdp(delta_t)
                w = np.clip(w + dw, 0.0, 1.5)

        if I_ext > 1.0 and i % 50 == 0:
            pre_spike_time = time[i]

    return time, V, spikes, np.full_like(time, I_ext), w

# 🧠 STDP szabály
def stdp(delta_t):
    A_plus = 0.01
    A_minus = -0.012
    tau_plus = 20.0
    tau_minus = 20.0
    if delta_t > 0:
        return A_plus * np.exp(-delta_t / tau_plus)
    else:
        return A_minus * np.exp(delta_t / tau_minus)

# ⚛️ 3D atomrácsos neuronháló szemléltetés
def atom_lattice_visualization(grid_size=5, spacing=1.5, spike_ratio=0.3):
    x, y, z, color, edges = [], [], [], [], []
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                idx = len(x)
                x.append(i * spacing)
                y.append(j * spacing)
                z.append(k * spacing)
                intensity = np.random.rand()
                color.append(intensity if intensity > (1 - spike_ratio) else 0)

                for dx, dy, dz in [(1,0,0),(0,1,0),(0,0,1)]:
                    ni, nj, nk = i + dx, j + dy, k + dz
                    if ni < grid_size and nj < grid_size and nk < grid_size:
                        edges.append(((i,j,k), (ni,nj,nk)))

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    color = np.array(color)

    fig = go.Figure()

    for (i1, j1, k1), (i2, j2, k2) in edges:
        fig.add_trace(go.Scatter3d(
            x=[i1*spacing, i2*spacing],
            y=[j1*spacing, j2*spacing],
            z=[k1*spacing, k2*spacing],
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False
        ))

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=4,
            color=color,
            colorscale='YlOrRd',
            opacity=0.85,
            colorbar=dict(title="Aktivitás")
        ),
        name='Neuronok'
    ))

    fig.update_layout(
        title="🧠 Atomrács alapú neuronháló – Szemléltetés",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500
    )

    return fig

# 🚀 Streamlit modul
def run():
    st.title("⚡ Spiking Neural Network – LIF Neuron és STDP")

    st.markdown("""
Ez a modul egy **LIF neuronmodell** működését mutatja be, **STDP** (Spike-Timing Dependent Plasticity) tanulással, valamint egy szemléltető 3D atomrácsos neuronhálóval.
    """)

    I_ext = st.slider("Bemeneti áram erőssége (I_ext)", 0.0, 3.0, 1.5, 0.1)
    tau_m = st.slider("Membrán időállandó (τ)", 1.0, 50.0, 20.0, 1.0)
    V_th = st.slider("Tüzelési küszöb (V_th)", 0.1, 2.0, 1.0, 0.1)
    stdp_on = st.checkbox("STDP tanulás engedélyezése", value=True)

    time, V, spikes, I_values, final_w = snn_simulate(
        I_ext=I_ext, tau_m=tau_m, V_th=V_th, stdp_enabled=stdp_on
    )

    st.subheader("📊 Membránpotenciál és tüzelés")
    fig, ax = plt.subplots()
    ax.plot(time, V, label="Membránpotenciál V(t)", color="tab:blue")
    ax.scatter(time[spikes > 0], [V_th] * int(np.sum(spikes)), color="red", marker="|", s=100, label="Spike esemény")
    ax.set_xlabel("Idő (ms)")
    ax.set_ylabel("Feszültség (V)")
    ax.set_title("LIF neuron működése")
    ax.legend()
    st.pyplot(fig)

    st.success(f"📊 Végső szinaptikus súly (w): **{final_w:.3f}**")

    st.subheader("📥 Eredmények letöltése")
    df = pd.DataFrame({
        "idő (ms)": time,
        "V": V,
        "spike": spikes,
        "I_ext": I_values
    })
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("CSV letöltése", data=csv, file_name="snn_simulation.csv")

    st.subheader("🧠 3D neuronháló (atomrács)")
    spike_strength = st.slider("Aktivitás intenzitása", 0.0, 1.0, 0.3, 0.05)
    st.plotly_chart(atom_lattice_visualization(grid_size=6, spike_ratio=spike_strength), use_container_width=True)
    st.caption("💡 A 3D rács csak szemléltetés, nem biológiai valóság. Az aktivitás véletlenszerűen generált.")

    st.markdown("""
# 📚 Tudományos háttér
st.markdown("### 📚 Tudományos háttér")

st.markdown("""
A **Leaky Integrate-and-Fire (LIF)** neuronmodell egy egyszerű, de hatékony biológiai ihletésű modell, amelyet előszeretettel használnak spiking neurális hálókban.
""")

st.markdown("""
**Fő mechanizmusai:**
- A membránpotenciál \( V \) folyamatosan integrálódik a bemeneti áram hatására.
- Ha a potenciál eléri a küszöböt \( V_{th} \), a neuron tüzel (spike-ol), majd visszaáll egy reset értékre.
- A membrán szivárgását egy elsőrendű differenciálegyenlet modellezi:
""")

st.latex(r"\frac{dV}{dt} = \frac{-(V) + R_m \cdot I_{ext}}{\tau_m}")

st.markdown("""
ahol:
- \( V \): membránpotenciál  
- \( R_m \): membránellenállás  
- \( I_{ext} \): bemeneti áram  
- \( \tau_m \): membrán időállandó  
""")

st.markdown("""
A **STDP (Spike-Timing Dependent Plasticity)** szabály az időzítésen alapuló szinaptikus plaszticitást írja le:
- Ha a **preszinaptikus** tüzelés **megelőzi** a posztszinaptikust: erősítés (LTP)
- Ha a **posztszinaptikus** tüzelés **korábbi**, mint a preszinaptikus: gyengítés (LTD)
""")

st.latex(r"\Delta w = \begin{cases} A_+ \cdot e^{-\Delta t / \tau_+}, & \text{ha } \Delta t > 0 \\[5pt] A_- \cdot e^{\Delta t / \tau_-}, & \text{ha } \Delta t < 0 \end{cases}")

st.markdown("""
ahol:
- \( \Delta w \): szinaptikus súlyváltozás  
- \( \Delta t = t_{post} - t_{pre} \): a tüzelési események időbeli különbsége  
- \( A_+ \), \( A_- \): erősítés és gyengítés mértéke  
- \( \tau_+ \), \( \tau_- \): időkonstansok  
""")

st.markdown("""
**Alkalmazások:**
- Neuromorf architektúrák
- Szenzoros tanulás időbeli korrelációkkal
- Energiahatékony beágyazott AI rendszerek
""")

# Kötelező hivatkozás
app = run
