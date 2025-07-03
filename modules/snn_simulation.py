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
    w = 0.5  # szinaptikus súly
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
            pre_spike_time = time[i]  # feltételezünk preszinaptikus aktivitást

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

# 🧊 3D hexagonális neuronháló – szemléltető célú
def draw_hex_neural_activity(grid_size=10, spread=3):
    np.random.seed(0)
    x, y, z = np.meshgrid(
        np.arange(grid_size),
        np.arange(grid_size),
        np.arange(grid_size)
    )

    coords = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    center = np.array([grid_size // 2] * 3)

    distances = np.linalg.norm(coords - center, axis=1)
    activity = np.exp(-((distances / spread)**2)) + 0.1 * np.random.rand(len(coords))
    colors = activity

    fig = go.Figure(data=go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='markers',
        marker=dict(
            size=3 + 10 * activity,
            color=colors,
            colorscale='YlOrRd',
            opacity=0.85,
            colorbar=dict(title='Aktivitás')
        )
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        title="🔺 Neuronaktivitás szemléltetése – Hexagonális 3D rácson",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig

# 🚀 Streamlit app
def run():
    st.title("⚡ Spiking Neural Network – LIF Neuron és STDP")
    st.markdown("""
Ez a modul egy **LIF neuronmodell** működését mutatja be, **STDP** (Spike-Timing Dependent Plasticity) tanulással, valamint egy látványos **3D neuronháló** animációval illusztrálja a tüzelési aktivitást.
""")

    # 🎛️ Paraméterek
    I_ext = st.slider("Bemeneti áram erőssége (I_ext)", 0.0, 3.0, 1.5, 0.1)
    tau_m = st.slider("Membrán időállandó (τ)", 1.0, 50.0, 20.0, 1.0)
    V_th = st.slider("Tüzelési küszöb (V_th)", 0.1, 2.0, 1.0, 0.1)
    spread = st.slider("3D aktivitás terjedése", 1, 10, 3)
    stdp_on = st.checkbox("STDP tanulás engedélyezése", value=True)

    # Szimuláció
    time, V, spikes, I_values, final_w = snn_simulate(
        I_ext=I_ext, tau_m=tau_m, V_th=V_th, stdp_enabled=stdp_on
    )

    # 📈 2D vizualizáció
    st.subheader("📊 Membránpotenciál és tüzelés")
    fig, ax = plt.subplots()
    ax.plot(time, V, label="Membránpotenciál V(t)", color="tab:blue")
    ax.scatter(time[spikes > 0], [V_th] * np.sum(spikes), color="red", marker="|", s=100, label="Spike esemény")
    ax.set_xlabel("Idő (ms)")
    ax.set_ylabel("Feszültség (V)")
    ax.set_title("LIF neuron működése")
    ax.legend()
    st.pyplot(fig)

    st.success(f"📊 Végső szinaptikus súly (w): **{final_w:.3f}**")

    # 🧊 3D neuronháló (szemléltető)
    st.subheader("🧠 3D Hexagonális Neuronháló (Szemléltetés)")
    st.plotly_chart(draw_hex_neural_activity(grid_size=12, spread=spread), use_container_width=True)

    # 📤 CSV export
    st.subheader("📥 Eredmények letöltése")
    df = pd.DataFrame({
        "idő (ms)": time,
        "V": V,
        "spike": spikes,
        "I_ext": I_values
    })
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("CSV letöltése", data=csv, file_name="snn_simulation.csv")

    # 📚 Tudományos háttér
    st.markdown("""
### 📚 Tudományos háttér

A **Leaky Integrate-and-Fire (LIF)** neuronmodell egy egyszerű, de hatékony biológiai modell:

- A membránpotenciál (\( V \)) folyamatosan integrálódik a bemeneti áram hatására.
- Ha \( V \geq V_{th} \), a neuron tüzel, majd visszaáll egy reset szintre.
- A membrán szivárog:  
  \( \frac{dV}{dt} = \frac{-(V) + R_m \cdot I_{ext}}{\tau_m} \)

A **STDP (Spike-Timing Dependent Plasticity)** szabály szerint:

- Ha a **preszinaptikus spike megelőzi** a posztszinaptikust → megerősítés (LTP)
- Ha a **posztszinaptikus spike előbb történik** → gyengítés (LTD)
- Ez modellezi a **szinaptikus plaszticitás időbeli érzékenységét**

A 3D háló szemlélteti a tüzelés térbeli terjedését – nem biológiailag pontos modell, hanem egy **vizuális analógia** az impulzusok dinamikájára.

**Alkalmazások:**

- Neuromorf architektúrák
- Mintázatfelismerés időalapú adatokban
- Energiatakarékos AI rendszerek
""")

# Kötelező Streamlit hivatkozás
app = run
