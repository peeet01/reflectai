import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd

def app():
    st.title("🧠 XOR Prediction - Pro Version")
    st.markdown("Klasszikus XOR probléma modern, mélytanulás-alapú megoldása vizualizációval és elemzéssel.")

    # Adatok
    X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
    Y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

    # UI
    hidden_size = st.slider("Rejtett réteg méret", 2, 16, 4)
    activation_name = st.selectbox("Aktivációs függvény", ["Sigmoid", "ReLU", "Tanh"])
    loss_fn_name = st.selectbox("Loss függvény", ["MSE", "Binary Crossentropy"])
    epochs = st.slider("Epoch-ok száma", 100, 5000, 1000)
    learning_rate = st.number_input("Tanulási ráta", 0.001, 1.0, 0.1)

    # Aktiváció kiválasztása
    if activation_name == "Sigmoid":
        activation = nn.Sigmoid()
    elif activation_name == "ReLU":
        activation = nn.ReLU()
    else:
        activation = nn.Tanh()

    class XORModel(nn.Module):
        def __init__(self):
            super(XORModel, self).__init__()
            self.fc1 = nn.Linear(2, hidden_size)
            self.fc2 = nn.Linear(hidden_size, 1)

        def forward(self, x):
            x = activation(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
            return x

    # Loss kiválasztása
    if loss_fn_name == "MSE":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.BCELoss()

    model = XORModel()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(epochs):
        y_pred = model(X)
        loss = loss_fn(y_pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Előrejelzés
    predicted = model(X).detach().numpy().round()
    accuracy = (predicted == Y.numpy()).mean()
    st.success(f"✅ Pontosság: {accuracy*100:.2f}%")

    # Loss plot
    fig1, ax1 = plt.subplots()
    ax1.plot(losses)
    ax1.set_title("Loss görbe")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    st.pyplot(fig1)

    # Confusion matrix
    cm = confusion_matrix(Y.numpy(), predicted)
    fig2, ax2 = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax2)
    st.pyplot(fig2)

    # Eredmények táblázatban
    df = pd.DataFrame({
        "Input 1": X[:,0],
        "Input 2": X[:,1],
        "Célérték": Y.flatten(),
        "Előrejelzés": predicted.flatten()
    })
    st.dataframe(df)

# Kötelező ReflectAI kompatibilitáshoz
app()
