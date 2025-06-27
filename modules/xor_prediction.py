import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score
import pandas as pd

# ----- Adatok (XOR) -----
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]], dtype=torch.float32)

y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]], dtype=torch.float32)

# ----- Hálózat -----
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ----- Tanítás -----
def train_model(model, X, y, epochs=2000, lr=0.1):
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

# ----- 3D Vizualizáció -----
def plot_decision_surface(model):
    xx, yy = torch.meshgrid(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100), indexing='ij')
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    with torch.no_grad():
        zz = model(grid).reshape(100, 100).numpy()

    fig = go.Figure(data=[
        go.Surface(z=zz, x=xx.numpy(), y=yy.numpy(), colorscale='Viridis', showscale=True),
        go.Scatter3d(x=X[:, 0], y=X[:, 1], z=y.flatten(), mode='markers',
                     marker=dict(size=5, color='red'), name='XOR pontok')
    ])
    fig.update_layout(title="XOR – 3D döntési felület", scene=dict(
        xaxis_title='x₁',
        yaxis_title='x₂',
        zaxis_title='output'
    ))
    st.plotly_chart(fig)

# ----- Streamlit App -----
def run():
    st.title("🔀 XOR Prediction – Pro Modul")
    st.markdown("""
    A **XOR probléma** egy klasszikus példa a nemlineáris osztályozási feladatra, amit egy 
    egyszerű MLP hálózat képes megtanulni. Itt egy 3D döntési felület is látható.
    """)

    model = XORNet()
    losses = train_model(model, X, y)

    # Eredmények
    with torch.no_grad():
        preds = (model(X) > 0.5).float()
    acc = accuracy_score(y, preds)

    st.success(f"🎯 Pontosság: {acc * 100:.2f}%")

    # 3D Plot
    st.subheader("📊 3D döntési felület")
    plot_decision_surface(model)

    # Loss görbe
    st.subheader("📉 Tanulási görbe")
    st.line_chart(losses)

    # Eredmények táblázat
    df = pd.DataFrame(torch.cat([X, preds], dim=1).numpy(), columns=["x₁", "x₂", "Predikció"])
    st.dataframe(df)

# ReflectAI kompatibilitás
app = run
