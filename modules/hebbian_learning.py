import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def hebbian_learning(inputs, targets, learning_rate):
    n_features = inputs.shape[1]
    weights = np.zeros((n_features,))
    for x, t in zip(inputs, targets):
        weights += learning_rate * x * t
    return weights

def run():
    st.header("🧠 Hebbian tanulás – szinaptikus súlytanulás")
    learning_rate = st.slider("Tanulási ráta (η)", 0.01, 1.0, 0.1)
    num_neurons = st.slider("Bemenetek száma", 2, 10, 3)

    # Bemeneti adatok és célértékek
    inputs = np.random.randint(0, 2, size=(10, num_neurons))
    targets = np.random.choice([-1, 1], size=10)

    st.subheader("🔢 Bemenetek és célértékek")
    st.write("Inputs:", inputs)
    st.write("Célértékek:", targets)

    weights = hebbian_learning(inputs, targets, learning_rate)

    st.subheader("📊 Tanult súlyok")
    fig, ax = plt.subplots()
    ax.bar(range(len(weights)), weights)
    ax.set_xlabel("Bemenet indexe")
    ax.set_ylabel("Súly érték")
    st.pyplot(fig)

    # 📥 Exportálás CSV-be
    st.subheader("📥 Eredmények exportálása")
    df = pd.DataFrame(inputs, columns=[f"x{i}" for i in range(num_neurons)])
    df["target"] = targets
    for i in range(num_neurons):
        df[f"weight_{i}"] = weights[i]
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Letöltés CSV-ben", data=csv, file_name="hebbian_learning_results.csv")

    # 📚 Tudományos háttér
    st.markdown("---")
    st.markdown("### 📚 Tudományos háttér")
    st.markdown(r"""
A **Hebbian tanulás** egy klasszikus, biológiailag motivált tanulási szabály, amely szerint:

> *"A neuronok, amelyek együtt tüzelnek, együtt huzalozódnak."*

Matematikailag a szabály így írható fel:

#### 🧮 Súlyfrissítési képlet:
\[
w_i \leftarrow w_i + \eta \cdot x_i \cdot t
\]

ahol:

- w_i: az i-edik bemenethez tartozó súly  
- \eta: tanulási ráta  
- x_i: bemenet aktuális értéke  
- t: a célérték (vagy a posztszinaptikus aktivitás)

Ez a szabály a szinaptikus erősségek változását modellezi az alapján, hogy a bemeneti és kimeneti neuronok **egyszerre aktiválódnak-e**. A Hebbian tanulás nem igényel hibavisszacsatolást vagy felügyelt tanulást.

---

#### 📌 Alkalmazások:

- **Associative memory** (pl. Hopfield-hálózatok)
- **Unsupervised learning** modellek
- **Neuroplaszticitás** modellezése

A Hebbian tanulás egyszerű, de mély kapcsolatot mutat a biológiai tanulással és a neurális hálózatok stabilizációjával.
    """, unsafe_allow_html=True)

app = run
