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

# 🎨 3D hexagonális neuronháló vizualizáció
def hex_grid_3d(n=6, spacing=1.0, spike_intensity=0.3):
    x, y, z = [], [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if (i + j + k) % 2 == 0:  # egyszerűsített hex illeszkedés
                    x.append(i * spacing)
                    y.append(j * spacing * np.sqrt(3)/2)
                    z.append(k * spacing * np.sqrt(2)/3)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    intensity = np.random.rand(len(x))
    intensity = np.where(intensity > (1 - spike_intensity), intensity, 0)

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=4,
                color=intensity,
                colorscale='YlOrRd',
                opacity=0.9,
                colorbar=dict(title="Aktivitás")
            )
        )
    ])
    fig.update_layout(
        title="🔬 Térbeli szemléltető neuronháló",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        height=500
    )
    return fig

# 🚀 Streamlit modul
def run():
    st.title("⚡ Spiking Neural Network – LIF Neuron és STDP")

    st.markdown("""
Ez a modul egy **LIF neuronmodell** működését mutatja be, **STDP** (Spike-Timing Dependent Plasticity) tanulással, valamint egy szemléltető 3D neuronrácsot is.
""")

    # 🎛️ Paraméterek
    I_ext = st.slider("Bemeneti áram erőssége (I_ext)", 0.0, 3.0, 1.5, 0.1)
    tau_m = st.slider("Membrán időállandó (τ)", 1.0, 50.0, 20.0, 1.0)
    V_th = st.slider("Tüzelési küszöb (V_th)", 0.1, 2.0, 1.0, 0.1)
    stdp_on = st.checkbox("STDP tanulás engedélyezése", value=True)

    time, V, spikes, I_values, final_w = snn_simulate(
        I_ext=I_ext, tau_m=tau_m, V_th=V_th, stdp_enabled=stdp_on
    )

    # 📈 Vizualizáció
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

    # 🧠 Hexagonális szemléltetés
    st.subheader("🧠 3D neuronrács (vizualizáció)")
    spike_strength = st.slider("Szimulált aktivitás aránya", 0.0, 1.0, 0.3, 0.05)
    st.plotly_chart(hex_grid_3d(n=8, spike_intensity=spike_strength), use_container_width=True)
    st.caption("💡 A 3D rács csak vizuális szemléltetés, nem matematikai modell alapján működik.")

    # 📚 Tudományos háttér
    st.markdown("""
### 📚 Tudományos háttér

A **Leaky Integrate-and-Fire (LIF)** neuronmodell egy egyszerű, de hatékony biológiai modell:

- A membránpotenciál (\( V \)) folyamatosan integrálódik a bemeneti áram hatására.
- Ha \( V \geq V_{th} \), a neuron tüzel, majd visszaáll a reset szintre.
- A membrán szivárog (leak):  
  \( \frac{dV}{dt} = \frac{-(V) + R_m \cdot I_{ext}}{\tau_m} \)

A **STDP** szabály szerint:
- Ha a **preszinaptikus spike megelőzi** a posztszinaptikust → megerősítés (LTP)
- Ha a **posztszinaptikus spike előbb történik** → gyengítés (LTD)

**Alkalmazások:**
- Neuromorf rendszerek
- Időfüggő mintázatok felismerése
- Alacsony energiaigényű AI rendszerek
""")

# Kötelező streamlit hivatkozás
app = run
