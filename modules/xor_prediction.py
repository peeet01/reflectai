import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score

# XOR adatok
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]], dtype=torch.float32)
y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]], dtype=torch.float32)

# Háló
class XORNet(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Tanítás
def train_model(model, X, y, epochs=2000, lr=0.1, progress_bar=False):
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    if progress_bar:
        bar = st.progress(0)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if progress_bar and epoch % (epochs // 100) == 0:
            bar.progress(epoch / epochs)
    if progress_bar:
        bar.progress(1.0)
    return losses

# 3D döntési felület
def plot_decision_surface(model):
    xx, yy = torch.meshgrid(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100), indexing='ij')
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    with torch.no_grad():
        zz = model(grid).reshape(100, 100).numpy()

    fig = go.Figure(data=[
        go.Surface(z=zz, x=xx.numpy(), y=yy.numpy(), colorscale='Viridis', showscale=True),
        go.Scatter3d(x=X[:, 0], y=X[:, 1], z=y.flatten(), mode='markers',
                     marker=dict(size=5, color='red'), name='XOR points')
    ])
    fig.update_layout(title="XOR – 3D Döntési Felület", scene=dict(
        xaxis_title='x₁', yaxis_title='x₂', zaxis_title='output'))
    st.plotly_chart(fig)

# Streamlit app
def run():
    st.title("🔀 XOR Prediction – Pro Verzió")

    st.sidebar.header("Modul Beállítások")
    hidden_size = st.sidebar.slider("Rejtett neuronok száma", 2, 10, 4)
    epochs = st.sidebar.slider("Epochok száma", 100, 5000, 2000, step=100)
    lr = st.sidebar.select_slider("Tanulási ráta", options=[0.001, 0.01, 0.05, 0.1], value=0.1)
    show_3d = st.sidebar.checkbox("3D Döntési Felület", True)
    show_loss = st.sidebar.checkbox("Tanulási Görbe", True)
    show_table = st.sidebar.checkbox("Predikciós Tábla", True)
    progress = st.sidebar.checkbox("Progress Bar", False)

    model = XORNet(hidden_size=hidden_size)
    losses = train_model(model, X, y, epochs=epochs, lr=lr, progress_bar=progress)

    with torch.no_grad():
        preds = (model(X) > 0.5).float()
    acc = accuracy_score(y, preds)
    st.success(f"🎯 Pontosság: {acc * 100:.2f}%")

    if show_3d:
        plot_decision_surface(model)
    if show_loss:
        st.line_chart(losses)
    if show_table:
        df = pd.DataFrame(torch.cat([X, preds], dim=1).numpy(), columns=["x₁", "x₂", "Predikció"])
        st.dataframe(df)

# ReflectAI integráció
app = run
