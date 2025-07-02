import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

def stdp_function(delta_t, A_plus, A_minus, tau_plus, tau_minus):
    dw = np.where(delta_t > 0,
                  A_plus * np.exp(-delta_t / tau_plus),
                  -A_minus * np.exp(delta_t / tau_minus))
    return dw

def run():
    st.title("⏱️ STDP – Időzített Szinaptikus Plaszticitás")
    st.markdown("Fedezd fel, hogyan módosul a szinaptikus erő a tüzelési időeltolódás függvényében.")

    # 🔧 Paraméterek
    A_plus = st.slider("Erősítő komponens (A⁺)", 0.01, 1.0, 0.1)
    A_minus = st.slider("Gyengítő komponens (A⁻)", 0.01, 1.0, 0.12)
    tau_plus = st.slider("Időkonstans (τ⁺)", 1.0, 50.0, 20.0)
    tau_minus = st.slider("Időkonstans (τ⁻)", 1.0, 50.0, 20.0)
    t_range = st.slider("Δt tartomány (ms)", -100, 100, 50)

    delta_t = np.linspace(-t_range, t_range, 400)
    delta_w = stdp_function(delta_t, A_plus, A_minus, tau_plus, tau_minus)

    # 📈 2D ábra
    st.subheader("📉 STDP görbe (Δw vs Δt)")
    fig, ax = plt.subplots()
    ax.plot(delta_t, delta_w, color='orange')
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_xlabel("Időeltolódás (Δt, ms)")
    ax.set_ylabel("Súlyváltozás (Δw)")
    ax.set_title("STDP szabály")
    st.pyplot(fig)

    # 🎨 3D vizualizáció
    st.subheader("🧠 Színes 3D vizualizáció neuronpárra")
    fig3d = go.Figure(data=[go.Scatter3d(
        x=delta_t,
        y=delta_w,
        z=np.zeros_like(delta_t),
        mode='lines',
        line=dict(color=delta_w, colorscale='Viridis', width=6),
        marker=dict(size=4)
    )])
    fig3d.update_layout(scene=dict(
        xaxis_title="Δt (ms)",
        yaxis_title="Δw",
        zaxis_title="Neuron tér",
    ), margin=dict(l=0, r=0, b=0, t=30), height=500)
    st.plotly_chart(fig3d)

    # CSV Export
    st.subheader("📥 Export")
    df = pd.DataFrame({"delta_t": delta_t, "delta_w": delta_w})
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Letöltés CSV-ben", data=csv, file_name="stdp_curve.csv")

    # Tudományos háttér
    st.markdown("### 📚 Tudományos háttér")
    st.markdown("""
Az STDP (Spike-Timing Dependent Plasticity) modell egy **biológiailag validált** tanulási szabály, amely a szinaptikus kapcsolatok erősödését vagy gyengülését az **időzítés** alapján szabályozza.

**Matematikai szabály:**

- Ha a pre- és posztszinaptikus aktivitás közötti időkülönbség Δt:
  
  - **Pozitív Δt** → Long-Term Potentiation (LTP)
  - **Negatív Δt** → Long-Term Depression (LTD)

\[
Δw(Δt) = 
\\begin{cases}
A^+ \\, e^{-Δt / τ^+}, & \\text{ha } Δt > 0 \\\\
- A^- \\, e^{Δt / τ^-}, & \\text{ha } Δt < 0
\\end{cases}
\]

A szabály lehetővé teszi, hogy az agy finom időzítési minták alapján tanuljon, és **időben korrelált aktivitásokhoz** igazítsa a kapcsolatok erősségét.

**Felhasználás az alkalmazásban:**

- A **tanulás irányának vizsgálata** időzített neuronpárokon
- Szinaptikus dinamika szimulálása hálózatokban
- Biológiai inspirációjú mesterséges rendszerek fejlesztése

**Tudományos következtetés:**

- Megmagyarázza a tüskék közötti szinaptikus erőváltozást
- Alapja sok **unsupervised** és **neurobiológiai** modellnek
""")
app = run   
