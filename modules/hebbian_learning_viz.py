import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

def hebbian_learning(x, y):
    return y @ x.T

def run():
    st.title("🧠 Hebbian Tanulás Vizualizáció")

    st.markdown("""
    A **Hebbian tanulás** alapelve:
    > *"Neurons that fire together, wire together."*  
    Ez a modul szemlélteti a tanult súlymátrixot és a tanulás hatását különböző bemenetekre.
    """)

    st.subheader("📂 Bemenet forrása")
    use_csv = st.checkbox("📥 CSV fájl feltöltése X és Y mátrixokhoz")

    if use_csv:
        uploaded_x = st.file_uploader("Töltsd fel az X mátrixot (bemenet)", type=["csv"])
        uploaded_y = st.file_uploader("Töltsd fel az Y mátrixot (kimenet)", type=["csv"])

        if uploaded_x and uploaded_y:
            x = pd.read_csv(uploaded_x, header=None).to_numpy()
            y = pd.read_csv(uploaded_y, header=None).to_numpy()
        else:
            st.warning("📄 Kérlek, tölts fel mindkét mátrixot.")
            return
    else:
        input_neurons = st.slider("🔢 Bemeneti neuronok száma", 2, 10, 3)
        output_neurons = st.slider("🔢 Kimeneti neuronok száma", 2, 10, 2)
        patterns = st.slider("📊 Minták száma", 3, 20, 5)
        x = np.random.randint(0, 2, (input_neurons, patterns))
        y = np.random.randint(0, 2, (output_neurons, patterns))

    # Tanulás
    w = hebbian_learning(x, y)
    y_pred = w @ x

    # Vizualizáció – súlymátrix
    st.subheader("📘 Súlymátrix $W = Y \\cdot X^T$")
    fig1, ax1 = plt.subplots()
    sns.heatmap(w, annot=True, cmap='coolwarm', ax=ax1)
    ax1.set_xlabel("Bemeneti neuronok")
    ax1.set_ylabel("Kimeneti neuronok")
    st.pyplot(fig1)

    # Vizualizáció – jósolt kimenet
    st.subheader("🔁 Jósolt kimenet: $Y_{pred} = W \\cdot X$")
    fig2, ax2 = plt.subplots()
    sns.heatmap(y_pred, annot=True, cmap='YlGnBu', ax=ax2)
    ax2.set_xlabel("Minták")
    ax2.set_ylabel("Kimeneti neuronok (jósolt)")
    st.pyplot(fig2)

    with st.expander("📊 Mátrixok részletesen"):
        st.write("🧩 Bemeneti mátrix (X):")
        st.dataframe(x)
        st.write("🎯 Célmátrix (Y):")
        st.dataframe(y)
        st.write("🧠 Tanult súlymátrix (W):")
        st.dataframe(w)
        st.write("📤 Jósolt kimenet (Y_pred):")
        st.dataframe(y_pred)

    st.subheader("💾 Exportálás")
    df_w = pd.DataFrame(w)
    df_pred = pd.DataFrame(y_pred)
    csv_w = df_w.to_csv(index=False, header=False).encode("utf-8")
    csv_pred = df_pred.to_csv(index=False, header=False).encode("utf-8")

    st.download_button("⬇️ Súlymátrix letöltése (W)", data=csv_w, file_name="hebbian_weights.csv")
    st.download_button("⬇️ Jósolt kimenet letöltése (Y_pred)", data=csv_pred, file_name="hebbian_output.csv")

    with st.expander("📚 Tudományos háttér"):
        st.markdown("""
        ### Hebbian tanulás
        A Hebbian tanulás az egyik legegyszerűbb és legismertebb szabály a szinaptikus erősségek frissítésére.  
        Lényege, hogy ha egy bemeneti és egy kimeneti neuron gyakran aktiválódik egyszerre, akkor kapcsolatuk erősödik.

        #### Matematikai leírás:
        A súlymátrix frissítése:
\[
        W = Y \cdot X^T
\]
        ahol:
        - X a bemeneti minták mátrixa (dimenzió: bemenet × minták),
        - Y a kívánt kimenet mátrixa (kimenet × minták),
        - W a tanult súlymátrix (kimenet × bemenet).

        Ez a szabály a klasszikus kondicionálás, asszociatív tanulás és egyszerű neurális hálózatok alapját képezi.
        """)
    
# Kötelező ReflectAI-kompatibilitás
app = run
