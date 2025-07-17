import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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

# Modell inicializálás
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Streamlit UI
def app():
    st.title("🧠 Autoencoder Vizualizáció – 3D Latens Tér")
    loss_history = []

    # Tréning
    epochs = 12
    if st.button("🚀 Tanítás indítása"):
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for data, _ in train_loader:
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, data.view(-1, 28 * 28))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            loss_history.append(avg_loss)
            st.write(f"📊 Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        # Rekonstrukciók megjelenítése
        model.eval()
        with torch.no_grad():
            test_imgs, test_labels = next(iter(test_loader))
            encoded = model.encoder(test_imgs)
            decoded = model.decoder(encoded).view(-1, 1, 28, 28)

        st.markdown("### 🔍 Rekonstruált képek")
        grid = make_grid(decoded[:10], nrow=5)
        st.image(grid.permute(1, 2, 0).numpy(), clamp=True)

        # 3D vizualizáció
        st.markdown("### 🌌 3D Latens tér")
        encoded_np = encoded.numpy()
        labels_np = test_labels.numpy()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(encoded_np[:, 0], encoded_np[:, 1], encoded_np[:, 2],
                             c=labels_np, cmap='tab10', s=10)
        ax.set_title("MNIST Rejtett reprezentáció (3D)")
        st.pyplot(fig)
