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
    import numpy as np
import plotly.graph_objects as go

# 3D predikciós mátrix létrehozása
xx, yy = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
X = np.c_[xx.ravel(), yy.ravel()]
Z = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5).astype(float).reshape(xx.shape)

# 3D ábra kombinált surface és kontúros megjelenítéssel
fig = go.Figure()

# Felület hozzáadása
fig.add_trace(go.Surface(
    z=Z, x=xx, y=yy,
    colorscale='Viridis',
    opacity=0.85,
    showscale=False
))

# Kontúr réteg hozzáadása
fig.add_trace(go.Contour(
    z=Z, x=np.linspace(0, 1, 50), y=np.linspace(0, 1, 50),
    contours=dict(start=0.0, end=1.0, size=0.5),
    line=dict(width=3, color='black'),
    showscale=False,
    opacity=0.4
))

# Megjelenítési beállítások
fig.update_layout(
    title="🧠 XOR – 3D Surface & Contour",
    scene=dict(
        xaxis_title='X1',
        yaxis_title='X2',
        zaxis_title='Output',
        zaxis=dict(nticks=4, range=[0, 1])
    ),
    margin=dict(l=0, r=0, t=60, b=0)
)

# Streamlit-kompatibilis
st.plotly_chart(fig, use_container_width=True) 
   

# Kötelező ReflectAI kompatibilitás
app = run
