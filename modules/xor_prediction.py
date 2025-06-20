import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

def run(hidden_size=2, learning_rate=0.1, epochs=1000, note=""):
    st.subheader("🧠 XOR predikció statisztikai összegzéssel")

    # XOR input és output
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_true = np.array([0,1,1,0])

    # Véletlen súlyok inicializálása
    np.random.seed(42)
    weights1 = np.random.randn(2, hidden_size)
    weights2 = np.random.randn(hidden_size)

    # Tanítási ciklus
    for epoch in range(epochs):
        z1 = X @ weights1
        a1 = np.tanh(z1)
        z2 = a1 @ weights2
        y_pred = (z2 > 0).astype(int)
        error = y_true - y_pred
        weights2 += learning_rate * a1.T @ error

    # Végső predikciók újraszámolása
    z1 = X @ weights1
    a1 = np.tanh(z1)
    z2 = a1 @ weights2
    y_pred = (z2 > 0).astype(int)

    # Táblázat készítés
    df = pd.DataFrame(X, columns=["Input 1", "Input 2"])
    df["Valós érték"] = y_true
    df["Predikció"] = y_pred

    # Pontosság
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    st.markdown(f"### 🎯 Pontosság: `{acc:.2f}`")

    # Konfúziós mátrix megjelenítés
    st.markdown("#### 📊 Konfúziós mátrix:")
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=[0, 1], yticklabels=[0, 1], ax=ax)
    ax.set_xlabel("Predikció")
    ax.set_ylabel("Valós érték")
    st.pyplot(fig_cm)

    # Plotly predikciós scatter ábra
    st.markdown("#### 🧩 Predikció vizualizáció:")
    fig = px.scatter(df, x="Input 1", y="Input 2",
                     color=df["Predikció"].astype(str),
                     symbol=df["Valós érték"].astype(str),
                     title="XOR predikciós térkép",
                     labels={"color": "Predikció", "symbol": "Valós érték"})
    st.plotly_chart(fig, use_container_width=True)

    # Letöltés
    st.download_button("📥 Eredmények letöltése CSV-ben",
                       data=df.to_csv(index=False).encode('utf-8'),
                       file_name="xor_predikcio.csv",
                       mime="text/csv")

    # Jegyzet
    if note:
        st.markdown(f"#### 📝 Jegyzet:\n> {note}")
