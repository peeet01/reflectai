import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pandas as pd

# Generátor hálózat
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# Diszkriminátor hálózat
class Discriminator(nn.Module):
    def __init__(self, img_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Minták kirajzolása
def plot_samples(generator, z_dim, device):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(16, z_dim).to(device)
        samples = generator(z).view(-1, 1, 28, 28).cpu()
        grid = make_grid(samples, nrow=4, normalize=True)
        fig, ax = plt.subplots()
        ax.imshow(grid.permute(1, 2, 0))
        ax.axis("off")
        st.pyplot(fig)

# Fő modul
def run():
    st.set_page_config(layout="wide")
    st.title("🧪 GAN Lab")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_dim = st.sidebar.slider("Z dimenzió", 64, 128, 100)
    lr = st.sidebar.select_slider("Tanulási ráta", options=[1e-4, 2e-4, 5e-4], value=2e-4)
    epochs = st.sidebar.slider("Epochok", 1, 10, 3)
    batch_size = st.sidebar.slider("Batch méret", 32, 128, 64)
    seed = st.sidebar.number_input("Seed", 0, 9999, 42)

    if st.button("Tréning indítása"):
        torch.manual_seed(seed)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        gen = Generator(z_dim).to(device)
        disc = Discriminator().to(device)
        opt_g = optim.Adam(gen.parameters(), lr=lr)
        opt_d = optim.Adam(disc.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        for epoch in range(epochs):
            for real_imgs, _ in loader:
                real_imgs = real_imgs.view(-1, 28*28).to(device)
                b_size = real_imgs.size(0)
                real = torch.ones(b_size, 1).to(device)
                fake = torch.zeros(b_size, 1).to(device)

                # Train D
                z = torch.randn(b_size, z_dim).to(device)
                fake_imgs = gen(z)
                loss_real = loss_fn(disc(real_imgs), real)
                loss_fake = loss_fn(disc(fake_imgs.detach()), fake)
                loss_d = (loss_real + loss_fake) / 2
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

                # Train G
                z = torch.randn(b_size, z_dim).to(device)
                fake_imgs = gen(z)
                loss_g = loss_fn(disc(fake_imgs), real)
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()

            st.write(f"Epoch {epoch+1}/{epochs} | G: {loss_g.item():.4f} | D: {loss_d.item():.4f}")

        st.subheader("🖼️ Generált minták")
        plot_samples(gen, z_dim, device)

        # Export
        z = torch.randn(100, z_dim).to(device)
        samples = gen(z).view(-1, 28*28).cpu().detach().numpy()
        df = pd.DataFrame(samples)
        st.download_button("⬇️ Letöltés CSV-ben", data=df.to_csv(index=False).encode("utf-8"), file_name="gan_samples.csv")

# ReflectAI kompatibilitás
app = run
