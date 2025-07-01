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

    with st.expander("📘 Tudományos háttér – Mi az a Hebbian tanulás?"):
    st.markdown("""
    A **Hebbian tanulás** az egyik legegyszerűbb és legismertebb szabály az ideghálózatok tanítására.

    ### 🧠 Alapelv:
    > *„Neurons that fire together, wire together.”*  
    Azaz: ha két neuron egyszerre aktiválódik, akkor megerősödik köztük a kapcsolat.

    ### 🧮 Matematikai modell:
    A súlymátrix kiszámítása:
\[
    W = Y \cdot X^T
\]
    - **X**: bemeneti neuronok aktivitása (bemenet × minták mátrix)
    - **Y**: kimeneti neuronok válasza (kimenet × minták mátrix)
    - **W**: tanult súlymátrix (kimenet × bemenet)

    A modell nem használ tanulási rátát, mivel ez egy egyszeri, lineáris tanulási szabály.

    ### 🔍 Alkalmazás:
    - Biológiai szinapszisok modellezése
    - Egyszerű asszociatív memória rendszerek
    - Hoppfield hálók alapelve
    - Adaptív szűrők és korai gépi tanulásos modellek

    A Hebbian tanulás jól használható oktatási célra, mivel intuitív és jól szemléltethető.
    """)
    
# Kötelező ReflectAI-kompatibilitás
app = run
