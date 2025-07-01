import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

def hebbian_learning(X, T, eta, epochs):
    weights = np.zeros(X.shape[1])
    history = []

    for _ in range(epochs):
        for x, t in zip(X, T):
            weights += eta * x * t
            history.append(weights.copy())

    return np.array(history)

def generate_inputs():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    T = np.array([0, 0, 0, 1])  # AND logikai művelet
    return X, T

def main():
    st.title("🧠 Hebbian Learning Szimuláció")
    st.markdown("Fedezd fel a Hebb-szabály működését egy egyszerű példán keresztül.")

    eta = st.slider("Tanulási ráta (η)", 0.01, 1.0, 0.1, step=0.01)
    epochs = st.slider("Epoch-ok száma", 1, 100, 20)

    X, T = generate_inputs()
    history = hebbian_learning(X, T, eta, epochs)

    # 2D plot
    st.subheader("📈 Súlyváltozások 2D-ben")
    fig, ax = plt.subplots()
    ax.plot(history[:, 0], label="w₀")
    ax.plot(history[:, 1], label="w₁")
    ax.set_xlabel("Iteráció")
    ax.set_ylabel("Súly érték")
    ax.set_title("Hebbian súlytanulás")
    ax.legend()
    st.pyplot(fig)

    # 3D plot
    st.subheader("📊 Súlypálya vizualizáció 3D-ben")
    fig3d = go.Figure(data=[go.Scatter3d(
        x=history[:, 0],
        y=history[:, 1],
        z=np.arange(len(history)),
        mode='lines+markers',
        marker=dict(size=4),
        line=dict(width=2)
    )])
    fig3d.update_layout(scene=dict(
        xaxis_title="w₀",
        yaxis_title="w₁",
        zaxis_title="Iteráció"
    ), margin=dict(l=0, r=0, b=0, t=30), height=500)
    st.plotly_chart(fig3d)

    # CSV export
    st.subheader("📥 Export")
    df = pd.DataFrame(history, columns=["w₀", "w₁"])
    csv = df.to_csv(index_label="iteráció").encode("utf-8")
    st.download_button("Súlyok letöltése CSV-ben", data=csv, file_name="hebb_weights.csv")

    # Tudományos háttér
    st.markdown("### 📚 Tudományos háttér")
    st.markdown(r"""
> *"A neuronok, amelyek együtt tüzelnek, együtt huzalozódnak."*

Matematikailag a Hebb-szabály így írható fel:

\[
w_i \leftarrow w_i + \eta \cdot x_i \cdot t
\]

ahol:

- w_i: az i-edik bemenethez tartozó súly  
- \eta: tanulási ráta  
- x_i: bemenet aktuális értéke  
- t: a célérték (posztszinaptikus aktivitás)

Ez a szabály a szinaptikus erősségek változását modellezi az alapján, hogy a bemeneti és kimeneti neuronok **egyszerre aktiválódnak-e**.  
A Hebbian tanulás **nem igényel hibavisszacsatolást**, és jól modellezi az agykéreg szinaptikus plaszticitását.
""")

# App indító
if __name__ == "__main__":
    main()
