import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

def run():
    st.title("🔁 XOR Prediction – Scientific Neural Network Playground")
    st.markdown("""
    Ez a modul egy neurális hálózatot alkalmaz az XOR logikai művelet megtanulására.  
    A modell beállításai testreszabhatók, és a teljes tanulási folyamat nyomon követhető vizuálisan is.
    """)

    # Dataset
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Beállítások
    hidden_layer_size = st.slider("🔧 Rejtett réteg mérete", 2, 20, 4)
    max_iter = st.slider("🔁 Iterációk száma", 100, 2000, 500, step=100)
    activation = st.selectbox("📐 Aktivációs függvény", ["tanh", "relu", "logistic"])
    solver = st.selectbox("🧮 Solver algoritmus", ["adam", "sgd", "lbfgs"])
    show_3d = st.checkbox("🌐 3D vizualizáció", value=True)

    # Modell betanítása
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_layer_size,),
        activation=activation,
        solver=solver,
        max_iter=1,
        warm_start=True
    )

    losses = []
    progress = st.progress(0)
    status = st.empty()

    for i in range(max_iter):
        model.fit(X_train, y_train)
        losses.append(model.loss_)
        progress.progress((i+1)/max_iter)
        status.text(f"Iteráció: {i+1}/{max_iter} | Loss: {model.loss_:.4f}")

    # Előrejelzés és értékelés
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"🎯 Modell pontossága: {acc:.2f}")

    # Konfúziós mátrix
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
    st.pyplot(fig_cm)

    # Loss-görbe
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(losses)
    ax_loss.set_xlabel("Iteráció")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("📉 Loss-görbe")
    st.pyplot(fig_loss)

    # 3D vizualizáció (fejlesztett)
   if show_3d:
    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
   if show_3d:
        xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
        zz = np.array([model.predict([[x, y]])[0] for x, y in zip(np.ravel(xx), np.ravel(yy))]).reshape(xx.shape)

        fig3d = go.Figure(data=[go.Surface(
            z=zz, x=xx, y=yy,
            colorscale='Electric',
            lighting=dict(ambient=0.4, diffuse=0.9, roughness=0.3, specular=1),
            lightposition=dict(x=100, y=200, z=100),
            showscale=True,
            opacity=0.95
        )])

        fig3d.update_layout(
            title="🧠 3D Előrejelzési Felszín (XOR)",
            scene=dict(
                xaxis_title='X1',
                yaxis_title='X2',
                zaxis_title='Kimenet',
                xaxis=dict(nticks=6, backgroundcolor="rgb(240,240,255)"),
                yaxis=dict(nticks=6, backgroundcolor="rgb(240,240,255)"),
                zaxis=dict(nticks=4, backgroundcolor="rgb(230,230,250)"),
                camera=dict(eye=dict(x=1.3, y=1.2, z=1.0))
            ),
            margin=dict(l=10, r=10, b=10, t=40)
        )

        st.plotly_chart(fig3d)

# Kötelező ReflectAI kompatibilitás
app = run
