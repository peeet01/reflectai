import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def generate_esn_input(data, delay):
    X, y = [], []
    for i in range(len(data) - delay):
        X.append(data[i:i + delay])
        y.append(data[i + delay])
    return np.array(X), np.array(y)

def esn_predict(X, y, reservoir_size=100, spectral_radius=0.95):
    input_size = X.shape[1]
    Win = np.random.rand(reservoir_size, input_size) - 0.5
    W = np.random.rand(reservoir_size, reservoir_size) - 0.5
    rho = max(abs(np.linalg.eigvals(W)))
    W *= spectral_radius / rho

    states = np.zeros((X.shape[0], reservoir_size))
    x = np.zeros(reservoir_size)
    for t in range(X.shape[0]):
        u = X[t]
        x = np.tanh(np.dot(Win, u) + np.dot(W, x))
        states[t] = x

    Wout = np.dot(np.linalg.pinv(states), y)
    y_pred = np.dot(states, Wout)
    return y_pred

def run():
    st.set_page_config(layout="wide")
    st.title("🔁 Echo State Network (ESN) predikciós szimuláció")

    st.markdown("""
    Az **Echo State Network (ESN)** egy visszacsatolt neurális háló, amelyet időbeli adatok előrejelzésére használnak.  
    Ebben a modulban egy szintetikus, zajjal terhelt szinuszos jelet próbál megjósolni a hálózat.
    """)

    # 🎛️ Paraméterek
    st.sidebar.header("⚙️ Paraméterek")
    series_length = st.sidebar.slider("Idősor hossza", 100, 1000, 300, step=50)
    delay = st.sidebar.slider("Késleltetés (delay)", 2, 50, 10)
    reservoir_size = st.sidebar.slider("Reservoir mérete", 10, 500, 100, step=10)
    spectral_radius = st.sidebar.slider("Spektrális sugár", 0.1, 1.5, 0.95, step=0.05)
    freq = st.sidebar.slider("Szinusz frekvencia", 0.5, 5.0, 1.0, step=0.1)
    noise_level = st.sidebar.slider("Zajszint", 0.0, 0.5, 0.05, step=0.01)

    # 🔊 Adatsor generálása
    t = np.linspace(0, 10, series_length)
    data = np.sin(2 * np.pi * freq * t) + np.random.normal(0, noise_level, series_length)

    # 🧠 Tanításhoz szükséges adatok
    X, y = generate_esn_input(data, delay)
    y_pred = esn_predict(X, y, reservoir_size, spectral_radius)

    # 📈 Plotly predikció
    st.subheader("📈 Predikciós eredmények")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y, mode="lines", name="Valós jel"))
    fig.add_trace(go.Scatter(y=y_pred, mode="lines", name="ESN predikció", line=dict(dash='dash')))
    fig.update_layout(
        xaxis_title="Időlépések",
        yaxis_title="Jel",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 📉 Hibamérték
    mse = np.mean((y - y_pred)**2)
    st.metric("📐 MSE (átlagos négyzetes hiba)", f"{mse:.5f}")

    # 💾 CSV export
    df = pd.DataFrame({"Valós": y, "Predikció": y_pred})
    st.subheader("💾 Eredmények letöltése")
    st.download_button("⬇️ CSV letöltése", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="esn_prediction.csv", mime="text/csv")

    # 📘 Matematikai háttér
    st.markdown("### 📘 ESN elméleti háttér")
    st.latex(r"x(t+1) = \tanh(W_{in} \cdot u(t) + W \cdot x(t))")
    st.markdown("""
    - $x(t)$: belső állapot (reservoir)  
    - $W_{in}$: bemeneti súlymátrix  
    - $W$: belső súlymátrix (rekurrens kapcsolatok)  
    - $u(t)$: bemenet  
    - A **kimenet ($y$)** a belső állapotból lineáris olvasóval történik:  
    $$
    y(t) = W_{out} \cdot x(t)
    $$  
    Az **ESN tanításakor csak a $W_{out}$** kerül optimalizálásra.
    """)

    # 📌 Konklúzió
    st.subheader("📌 Konklúzió")
    st.markdown("""
    - Az ESN képes **nemlineáris időbeli mintázatok** megtanulására, ha a belső dinamika (spectral radius) megfelelően van hangolva.
    - **Túl magas zajszint** rontja az előrejelzést, különösen kis reservoir esetén.
    - A **delay mérete** kulcsszerepet játszik a memóriakapacitás és a prediktív teljesítmény közötti egyensúlyban.
    """)

    # 📝 Felhasználói megfigyelés
    st.subheader("📝 Megfigyelések")
    st.text_area("Írd le, mit tapasztaltál a predikcióval kapcsolatban!", height=120)

# 🔧 Modulregisztrációhoz
app = run
