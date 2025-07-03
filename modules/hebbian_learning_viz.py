import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

def hebbian_learning(x, y):
    return y @ x.T

def plot_3d_matrix(matrix, title="3D Mátrix vizualizáció"):
    z = matrix
    x, y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
    fig.update_layout(title=title, scene=dict(
        xaxis_title="Bemenet",
        yaxis_title="Kimenet",
        zaxis_title="Súly"
    ))
    return fig

def run():
    st.title("🧠 Hebbian Tanulás Vizualizáció")

    st.markdown("""
A **Hebbian tanulás** az egyik legegyszerűbb tanulási szabály, amely a biológiai idegrendszerek működéséből ered.  
Ez a modul lehetővé teszi, hogy egy bemeneti–kimeneti mátrix alapján vizualizáljuk a tanult súlymátrixot, és megbecsüljük annak hatását.

A tanulási szabály szerint a szinaptikus kapcsolatok megerősödnek, ha a bemeneti és a kimeneti neuron egyszerre aktívak.
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

    # Vizualizáció – súlymátrix (2D)
    st.subheader("📘 Súlymátrix $W = Y \\cdot X^T$")
    fig1, ax1 = plt.subplots()
    sns.heatmap(w, annot=True, cmap='coolwarm', ax=ax1)
    ax1.set_xlabel("Bemeneti neuronok")
    ax1.set_ylabel("Kimeneti neuronok")
    st.pyplot(fig1)

    if st.checkbox("🌐 3D súlymátrix Plotly"):
        fig3d = plot_3d_matrix(w, title="Tanult súlymátrix 3D-ben")
        st.plotly_chart(fig3d, use_container_width=True)

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

    st.markdown("### 📚 Tudományos háttér")
    st.markdown(r"""
A **Hebbian-tanulás** az egyik legismertebb biológiai ihletésű tanulási szabály, amely az agy szinaptikus plaszticitását írja le.  
A modell célja, hogy a bemeneti minták és a megfelelő kimenetek alapján erősítse a releváns súlyokat.

**Tanulási szabály:**

$$
W = Y \cdot X^T
$$

Ahol:
- \( X \in \mathbb{R}^{n \times p} \): bemeneti minták (n bemenet, p minta)
- \( Y \in \mathbb{R}^{m \times p} \): kívánt kimenetek (m kimenet, p minta)
- \( W \in \mathbb{R}^{m \times n} \): tanult súlymátrix

Ez azt jelenti, hogy a kimeneti aktivitás súlyozott módon visszahat a bemenetekre, megerősítve azokat a kapcsolatokat, amelyek együttes aktivitást mutatnak.

**Jelentősége:**
- Biológiai idegrendszerek tanulmányozása
- Asszociatív memória (pl. Hopfield-hálózat)
- Szinaptikus erősítés elméleti alapja

A Hebbian szabály a **korrelációtanulás** alapvető példája, amely világosan illusztrálja, hogyan alakulhatnak ki neurális asszociációk.
""")

# ReflectAI kompatibilitás
app = run
