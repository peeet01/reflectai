import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# --- Egyszerű Autoencoder ---
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def plot_reconstruction(model, device):
    model.eval()
    with torch.no_grad():
        sample = next(iter(DataLoader(
            datasets.MNIST('./data', train=False, download=True,
                           transform=transforms.ToTensor()),
            batch_size=8
        )))[0].to(device)
        original = sample.view(-1, 28*28)
        reconstructed = model(original).view(-1, 1, 28, 28).cpu()
        grid = make_grid(reconstructed, nrow=4, normalize=True)
        fig, ax = plt.subplots()
        ax.imshow(grid.permute(1, 2, 0))
        ax.axis("off")
        st.pyplot(fig)

# --- Streamlit modul ---
def run():
    st.set_page_config(layout="wide")
    st.title("🧠 Autoencoder – Dimenziócsökkentés és rekonstrukció")

    st.markdown("""
    Az **autoencoder** egy neurális háló, amely megtanulja **tömöríteni** a bemeneti adatot és utána **rekonstruálni** azt.  
    Célja, hogy a tömörített reprezentáció (kód) elegendő információt tartalmazzon a bemenet visszaállításához.

    - Encoder: bemeneti adat → tömörített reprezentáció  
    - Decoder: reprezentáció → rekonstrukció

    Hasznos **dimenziócsökkentésre**, **zajszűrésre** és **adatfeltárásra**.
    """)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = st.sidebar.slider("Epochok száma", 1, 20, 5)
    lr = st.sidebar.select_slider("Tanulási ráta", [1e-4, 5e-4, 1e-3], value=1e-3)
    batch_size = st.sidebar.slider("Batch méret", 32, 256, 128, step=32)

    if st.button("🔁 Autoencoder tanítása"):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = Autoencoder().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0
            for imgs, _ in loader:
                imgs = imgs.view(-1, 28*28).to(device)
                outputs = model(imgs)
                loss = criterion(outputs, imgs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            st.write(f"📊 Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

        st.subheader("🔍 Rekonstruált képek")
        plot_reconstruction(model, device)

# ReflectAI-kompatibilitás
app = run
