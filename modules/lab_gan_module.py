import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

# Generator hálózat
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

# Discriminator hálózat
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

# Képek megjelenítése
def show_images(images):
    grid = make_grid(images.view(-1, 1, 28, 28), nrow=4, normalize=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
    ax.axis("off")
    st.pyplot(fig)

# Egyetlen tanítási lépés
def training_step(generator, discriminator, optim_g, optim_d, real_imgs, z_dim, device):
    batch = real_imgs.size(0)
    real_imgs = real_imgs.view(batch, -1).to(device)

    real_labels = torch.ones(batch, 1).to(device)
    fake_labels = torch.zeros(batch, 1).to(device)

    # Diszkriminátor
    z = torch.randn(batch, z_dim).to(device)
    fake_imgs = generator(z)
    real_loss = nn.BCELoss()(discriminator(real_imgs), real_labels)
    fake_loss = nn.BCELoss()(discriminator(fake_imgs.detach()), fake_labels)
    loss_d = (real_loss + fake_loss) / 2

    optim_d.zero_grad()
    loss_d.backward()
    optim_d.step()

    # Generátor
    z = torch.randn(batch, z_dim).to(device)
    fake_imgs = generator(z)
    loss_g = nn.BCELoss()(discriminator(fake_imgs), real_labels)

    optim_g.zero_grad()
    loss_g.backward()
    optim_g.step()

    return loss_d.item(), loss_g.item(), fake_imgs

# Fő Streamlit alkalmazás
def run():
    st.set_page_config(layout="wide")
    st.title("🧪 GAN Lab – simple test")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Init state
    if "losses" not in st.session_state:
        st.session_state["losses"] = []
    if "batch_iter" not in st.session_state:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)
        st.session_state["loader"] = loader
        st.session_state["batch_iter"] = iter(loader)

    # Modellek
    z_dim = 100
    generator = Generator(z_dim).to(device)
    discriminator = Discriminator().to(device)
    optim_g = optim.Adam(generator.parameters(), lr=2e-4)
    optim_d = optim.Adam(discriminator.parameters(), lr=2e-4)

    st.success("✅ Modul betöltődött, egy tanítási lépés elérhető.")

    st.markdown("## 🎯 GAN lépések")
    steps = st.slider("Hány tanítási lépés fusson?", 1, 50, 5)

    if st.button(f"Run {steps} training step"):
        for _ in range(steps):
            try:
                real_imgs, _ = next(st.session_state["batch_iter"])
            except StopIteration:
                st.session_state["batch_iter"] = iter(st.session_state["loader"])
                real_imgs, _ = next(st.session_state["batch_iter"])

            loss_d, loss_g, fake_imgs = training_step(
                generator, discriminator, optim_g, optim_d,
                real_imgs, z_dim, device
            )
            st.session_state["losses"].append((loss_d, loss_g))

        st.write(f"🧪 Prediction (fake): `{discriminator(fake_imgs[0].view(1, -1)).item():.12f}`")
        st.write(f"📉 Loss: `{loss_d:.12f}`")

        show_images(fake_imgs[:16])

    # Loss grafikon
    if st.session_state["losses"]:
        st.subheader("📊 Loss görbék")
        d_loss, g_loss = zip(*st.session_state["losses"])
        fig, ax = plt.subplots()
        ax.plot(d_loss, label="Diszkriminátor")
        ax.plot(g_loss, label="Generátor")
        ax.set_xlabel("Lépés")
        ax.set_ylabel("Loss")
        ax.legend()
        st.pyplot(fig)

# ReflectAI kompatibilitás
app = run
