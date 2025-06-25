import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification

def create_data(n_samples=500, noise=0.2):
    X = np.random.uniform(-1, 1, (n_samples, 2))
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    return X, y

def train_model(X, y, hidden_layer_sizes=(10,), alpha=0.01, max_iter=1000):
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='tanh',
                        solver='adam', alpha=alpha, max_iter=max_iter, random_state=42)
    clf.fit(X, y)
    return clf

def plot_3d_decision_boundary(model, X, y, resolution=50):
    xx, yy = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=y,
        mode='markers',
        marker=dict(size=4, color=y, colorscale='Viridis'),
        name='Adatpontok'
    ))

    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz,
        colorscale='RdBu', opacity=0.7, showscale=False,
        name='Döntési határ'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X1',
            yaxis_title='X2',
            zaxis_title='Kimenet',
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

def run():
    st.header("🧩 XOR Predikció – 3D Döntési Határ Vizualizáció")
    st.markdown("Ez a szimuláció egy neurális hálóval tanulja meg az XOR logikai kaput, és vizualizálja a döntési felületet 3D-ben.")

    n_samples = st.slider("Minták száma", 100, 1000, 500, 50)
    hidden_neurons = st.slider("Rejtett neuronok száma", 2, 20, 10, 1)
    alpha = st.slider("Regularizációs erő (alpha)", 0.0001, 0.1, 0.01, 0.0001)

    X, y = create_data(n_samples)
    model = train_model(X, y, hidden_layer_sizes=(hidden_neurons,), alpha=alpha)
    plot_3d_decision_boundary(model, X, y)

# Kötelező az app dinamikus betöltéséhez:
app = run
