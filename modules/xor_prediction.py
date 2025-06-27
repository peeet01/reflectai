import numpy as np import matplotlib.pyplot as plt import streamlit as st import time from sklearn.neural_network import MLPClassifier from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay import plotly.graph_objects as go from sklearn.model_selection import train_test_split

def run(): st.title("🔁 XOR Prediction – Scientific Neural Network Playground") st.markdown(""" Ez a modul egy neurális hálózatot alkalmaz az XOR logikai művelet megtanulására.
A modell beállításai testreszabhatók, és a teljes tanulási folyamat nyomon követhető vizuálisan is. """)

# Dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Beállítások
hidden_layer_size = st.slider("🔧 Rejtett réteg mérete", 2, 20, 4)
max_iter = st.slider("🔁 Iterációk száma", 100, 2000, 500, step=100)
activation = st.selectbox("📊 Aktivációs függvény", ["tanh", "relu", "logistic"])
solver = st.selectbox("🧲 Solver algoritmus", ["adam", "sgd", "lbfgs"])
show_3d = st.checkbox("🌐 3D vizualizáció", value=True)

# Modell tanítása (tanulásképes változat)
model = MLPClassifier(
    hidden_layer_sizes=(hidden_layer_size,),
    activation=activation,
    solver=solver,
    max_iter=max_iter,
    random_state=42
)

start = time.time()
model.fit(X_train, y_train)
end = time.time()

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.success(f"🌟 Modell pontossága: {acc:.2f} | Tanítás ideje: {end-start:.3f} sec")

# Loss-görbe megjelenítése
if hasattr(model, 'loss_curve_'):
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(model.loss_curve_)
    ax_loss.set_xlabel("Iteráció")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("📉 Loss-görbe")
    st.pyplot(fig_loss)

# Konfúzós mátrix
fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
st.pyplot(fig_cm)

# 3D vizualizáció (vizuálisan finomabb)
if show_3d:
    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    zz = np.array([
        model.predict([[x, y]])[0] for x, y in zip(np.ravel(xx), np.ravel(yy))
    ]).reshape(xx.shape)

    fig = go.Figure(data=[
        go.Surface(
            z=zz,
            x=xx,
            y=yy,
            colorscale='Viridis',
            opacity=0.9,
            showscale=True,
            contours=dict(
                x=dict(show=False),
                y=dict(show=False),
                z=dict(show=False)
            )
        )
    ])

    fig.update_layout(
        title="🧠 XOR – 3D Surface from Neural Network",
        scene=dict(
            xaxis=dict(title='X1'),
            yaxis=dict(title='X2'),
            zaxis=dict(title='Output', nticks=4, range=[0, 1])
        ),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

# 📖 Működésmagyárat
with st.expander("🔎 Hogyan működik?"):
    st.markdown("""
    - Az XOR probléma kizárólag nemlineáris módszerekkel tanulható meg.
    - A modul egy MLPClassifier-t alkalmaz, melynek tanulása visszaterjesztéses algoritmussal történik.
    - A rejtett réteg mérete, aktiváció, és a solver algoritmus is testreszabható.
    - A tanulás hatékonysága loss és 3D vizualizáció által mérhető.
    """)

ReflectAI integrációhoz

app = run

