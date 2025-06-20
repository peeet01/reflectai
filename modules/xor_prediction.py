import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Neurális háló osztály
class XORNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.out(self.fc2(x))
        return x

def run(hidden_size, learning_rate, epochs, note):
    st.subheader("🧠 XOR predikció neurális hálóval (Pro verzió)")

    # Felhasználói zaj szint beállítása
    noise_level = st.slider("Zaj szint (0.0 = nincs zaj, 1.0 = teljes)", 0.0, 1.0, 0.1, step=0.01)

    # XOR bemenetek és kimenetek
    X_raw = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
    Y_raw = np.array([[0], [1], [1], [0]], dtype=np.float32)

    # Zaj hozzáadása a bemenethez
    noise = noise_level * np.random.randn(*X_raw.shape).astype(np.float32)
    X_noisy = X_raw + noise

    X = torch.from_numpy(X_noisy)
    Y = torch.from_numpy(Y_raw)

    # Modell inicializálás
    model = XORNet(input_size=2, hidden_size=hidden_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    accuracies = []

    # Tanítás
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == Y).float().mean().item()
            losses.append(loss.item())
            accuracies.append(accuracy)

    # Előrejelzések
    with torch.no_grad():
        final_outputs = model(X)
        final_preds = (final_outputs > 0.5).float()
        final_acc = (final_preds == Y).float().mean().item()

    # Megjegyzés
    if note:
        st.markdown(f"📌 **Megjegyzés:** _{note}_")

    # Pontosság, konfidencia
    st.markdown(f"✅ **Pontosság:** `{final_acc*100:.2f}%`")
    st.markdown("### 🔍 Előrejelzések részletezve:")

    results_df = pd.DataFrame({
        "Bemenet 1": X_raw[:, 0],
        "Bemenet 2": X_raw[:, 1],
        "Valós kimenet": Y_raw[:, 0],
        "Predikció": final_preds.numpy().flatten(),
        "Konfidencia": final_outputs.numpy().flatten()
    })

    st.dataframe(results_df.style.background_gradient(cmap="RdYlGn", subset=["Konfidencia"]))

    # Loss és pontosság grafikon
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(losses, label="Veszteség")
    ax[0].set_title("Tanulási veszteség")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(accuracies, label="Pontosság", color="green")
    ax[1].set_title("Tanulási pontosság")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Pontosság")
    ax[1].legend()

    st.pyplot(fig)

    # Hőtérkép a bemenet és konfidencia viszonyáról
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(results_df.pivot_table(index="Bemenet 1", columns="Bemenet 2", values="Konfidencia"),
                annot=True, fmt=".2f", cmap="viridis", ax=ax2)
    ax2.set_title("📊 Bemenet ↔ Konfidencia hőtérkép")
    st.pyplot(fig2)
