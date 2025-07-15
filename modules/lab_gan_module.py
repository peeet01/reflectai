import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Generator hálózat
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# Discriminator hálózat
class Discriminator(nn.Module):
    def __init__(self, img_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Globális állapot (egyszeri inicializálás)
@st.cache_resource
def init_models(z_dim, lr, device):
    generator = Generator(z_dim).to(device)
    discriminator = Discriminator().to(device)
    optim_g = optim.Adam(generator.parameters(), lr=lr)
    optim_d = optim.Adam(discriminator.parameters(), lr=lr)
    return generator, discriminator, optim_g, optim_d

@st.cache_data
def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    return dataset

# Egy lépés tréning
def training_step(generator, discriminator, optim_g, optim_d, real_imgs, z_dim, device):
    criterion = nn.BCELoss()
    real_imgs = real_imgs.view(-1, 28*28).to(device)
    batch = real_imgs.size(0)
    real = torch.ones(batch, 1).to(device)
    fake = torch.zeros(batch, 1).to(device)

    # Train D
    z = torch.randn(batch, z_dim).to(device)
    fake_imgs = generator(z)
    loss_real = criterion(discriminator(real_imgs), real)
    loss_fake = criterion(discriminator(fake_imgs.detach()), fake)
    loss_d = (loss_real + loss_fake) / 2
    optim_d.zero_grad()
    loss_d.backward()
    optim_d.step()

    # Train G
    z = torch.randn(batch, z_dim).to(device)
    fake_imgs = generator(z)
    loss_g = criterion(discriminator(fake_imgs), real)
    optim_g.zero_grad()
    loss_g.backward()
    optim_g.step()

    return loss_d.item(), loss_g.item(), fake_imgs.detach()

# Képek megjelenítése
def show_images(images, nrow=4):
    grid = make_grid(images.view(-1, 1, 28, 28), nrow=nrow, normalize=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(grid.permute(1, 2, 0))
    ax.axis("off")
    st.pyplot(fig)

# Streamlit app
def run():
    st.set_page_config(layout="wide")
    st.title("🧪 GAN Lab — Trainable GAN")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    z_dim = st.sidebar.slider("Z dimenzió", 64, 256, 100, step=16)
    lr = st.sidebar.select_slider("Tanulási ráta", options=[1e-4, 2e-4, 5e-4, 1e-3], value=2e-4)
    batch_size = st.sidebar.slider("Batch méret", 32, 256, 128, step=32)

    generator, discriminator, optim_g, optim_d = init_models(z_dim, lr, device)
    dataset = load_mnist()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch_iter = iter(loader)

    if "losses" not in st.session_state:
        st.session_state["losses"] = []

    if st.button("🎯 Futtass egy tréning lépést"):
        try:
            real_imgs, _ = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            real_imgs, _ = next(batch_iter)

        loss_d, loss_g, generated = training_step(
            generator, discriminator, optim_g, optim_d, real_imgs, z_dim, device
        )

        st.session_state["losses"].append((loss_d, loss_g))

        st.success("Tanítás sikeres ✅")
        st.write(f"Diszkriminátor Loss: `{loss_d:.4f}` | Generátor Loss: `{loss_g:.4f}`")
        st.subheader("🖼 Generált képek")
        show_images(generated)

    if st.session_state["losses"]:
        d, g = zip(*st.session_state["losses"])
        fig, ax = plt.subplots()
        ax.plot(d, label="Diszkriminátor")
        ax.plot(g, label="Generátor")
        ax.set_title("Loss alakulása")
        ax.legend()
        st.pyplot(fig)

# ReflectAI kompatibilitás
app = run
