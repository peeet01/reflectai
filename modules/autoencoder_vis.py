import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# ----- Inicializálás -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# ----- Adatok betöltése -----
transform = transforms.ToTensor()
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=128, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transform),
    batch_size=1000, shuffle=False
)

# ----- Autoencoder modell -----
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3D latent space
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# Streamlit UI
def app():
    st.title("🧠 Autoencoder Vizualizáció – 3D Latens Tér")

    epochs = st.sidebar.slider("Epochok száma", 1, 30, 10)
    learning_rate = st.sidebar.select_slider("Tanulási ráta", options=[1e-4, 5e-4, 1e-3, 2e-3], value=1e-3)

    if st.button("🚀 Tanítás indítása"):
        model = Autoencoder().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        loss_history = []
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for data, _ in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, data.view(-1, 28 * 28).to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            loss_history.append(avg_loss)
            st.write(f"📊 Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        # Loss ábra
        st.subheader("📉 Veszteség alakulása")
        fig1, ax1 = plt.subplots()
        ax1.plot(loss_history)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("MSE Loss")
        st.pyplot(fig1)

        # Rekonstrukciók
        st.subheader("🖼️ Rekonstruált képek")
        model.eval()
        with torch.no_grad():
            test_imgs, test_labels = next(iter(test_loader))
            test_imgs = test_imgs.to(device)
            encoded = model.encoder(test_imgs)
            decoded = model.decoder(encoded).view(-1, 1, 28, 28).cpu()

        grid = make_grid(decoded[:10], nrow=5, normalize=True)
        st.image(grid.permute(1, 2, 0).numpy(), clamp=True)

        # Latens tér
        st.subheader("🌌 3D Latens tér")
        encoded_np = encoded.cpu().numpy()
        labels_np = test_labels.numpy()

        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')
        ax.scatter(encoded_np[:, 0], encoded_np[:, 1], encoded_np[:, 2], c=labels_np, cmap='tab10', s=10)
        ax.set_title("MNIST rejtett reprezentáció (3D)")
        st.pyplot(fig2)

# ReflectAI kompatibilitás
app = app
