import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pandas as pd

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=28*28):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_dim=28*28):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Helper: Show images
def show_generated_images(generator, z_dim, device):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(16, z_dim).to(device)
        generated = generator(z).view(-1, 1, 28, 28).cpu()
        grid = make_grid(generated, nrow=4, normalize=True)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(grid.permute(1, 2, 0))
        ax.axis("off")
        st.pyplot(fig)

# Main app
def run():
    st.title("🧪 GAN Lab – Generative Adversarial Network")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    z_dim = st.sidebar.slider("Z dimenzió", 64, 256, 100, step=16)
    lr = st.sidebar.select_slider("Tanulási ráta", options=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3], value=2e-4)
    epochs = st.sidebar.slider("Epochok", 1, 30, 5)
    batch_size = st.sidebar.slider("Batch méret", 32, 256, 128, step=32)
    seed = st.sidebar.number_input("Seed", 0, 9999, 42)

    if st.button("Tanítás indítása"):
        torch.manual_seed(seed)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        G = Generator(z_dim).to(device)
        D = Discriminator().to(device)
        opt_G = optim.Adam(G.parameters(), lr=lr)
        opt_D = optim.Adam(D.parameters(), lr=lr)
        criterion = nn.BCELoss()

        g_losses, d_losses = [], []

        for epoch in range(epochs):
            for real_imgs, _ in loader:
                real_imgs = real_imgs.view(-1, 28*28).to(device)
                batch_size = real_imgs.size(0)
                real = torch.ones(batch_size, 1).to(device)
                fake = torch.zeros(batch_size, 1).to(device)

                # Discriminator
                z = torch.randn(batch_size, z_dim).to(device)
                fake_imgs = G(z)
                d_real = D(real_imgs)
                d_fake = D(fake_imgs.detach())
                loss_D = (criterion(d_real, real) + criterion(d_fake, fake)) / 2
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

                # Generator
                z = torch.randn(batch_size, z_dim).to(device)
                fake_imgs = G(z)
                d_output = D(fake_imgs)
                loss_G = criterion(d_output, real)
                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()

            g_losses.append(loss_G.item())
            d_losses.append(loss_D.item())
            st.write(f"Epoch {epoch+1}/{epochs} | G: {loss_G.item():.4f} | D: {loss_D.item():.4f}")

        # Loss plot
        st.subheader("📉 Loss görbék")
        fig, ax = plt.subplots()
        ax.plot(g_losses, label="Generátor")
        ax.plot(d_losses, label="Diszkriminátor")
        ax.legend()
        st.pyplot(fig)

        # Generate images
        st.subheader("🖼️ Generált képek")
        show_generated_images(G, z_dim, device)

        # CSV export
        z = torch.randn(100, z_dim).to(device)
        samples = G(z).view(-1, 28*28).cpu().detach().numpy()
        df = pd.DataFrame(samples)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Minták letöltése CSV-ben", data=csv, file_name="gan_samples.csv")

# Export
app = run
