import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim

# Egyszerű XOR háló modell
class XORNet(nn.Module):
    def __init__(self, hidden_size):
        super(XORNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def run(hidden_size, learning_rate, epochs, note=""):
    st.write(f"🔧 Háló tanítása: hidden_size={hidden_size}, lr={learning_rate}, epochs={epochs}")
    if note:
        st.info(f"📝 Megjegyzés: {note}")

    # XOR adat
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    model = XORNet(hidden_size)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    losses = []
    for epoch in range(epochs):
        output = model(X)
        loss = loss_fn(output, Y)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % (epochs // 5) == 0:
            st.write(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # Eredmény kiírás
    st.subheader("📊 Tanulás befejezve. Eredmények:")
    with torch.no_grad():
        preds = model(X)
        for i in range(4):
            a, b = X[i].tolist()
            pred = preds[i].item()
            st.write(f"{int(a)} XOR {int(b)} ≈ {pred:.2f}")

    # Loss görbe (opcionálisan matplotlibtel is mehetne)
    st.line_chart(losses)
