import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import io

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim=64, img_dim=28*28):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, img_dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_dim=28*28):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

@st.cache_data
def load_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def show_images(generator, z_dim, device):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(16, z_dim).to(device)
        fake_imgs = generator(z).view(-1, 1, 28, 28).cpu()
        grid = make_grid(fake_imgs, nrow=4, normalize=True)
        fig, ax = plt.subplots()
        ax.imshow(grid.permute(1, 2, 0))
        ax.axis("off")
        st.pyplot(fig)

        buffer = io.BytesIO()
        save_image(fake_imgs, buffer, format='png')
        st.download_button("⬇️ Minták letöltése (PNG)", data=buffer.getvalue(), file_name="samples.png", mime="image/png")

def run():
    st.set_page_config(layout="wide")
    st.title("🧪 GAN – Generative Adversarial Network")

    st.sidebar.header("⚙️ Paraméterek")
    z_dim = st.sidebar.slider("Z dimenzió", 32, 128, 64, step=16)
    lr = st.sidebar.select_slider("Tanulási ráta", options=[1e-5, 5e-5, 1e-4, 2e-4], value=2e-4)
    epochs = st.sidebar.slider("Epochok száma", 1, 5, 2)
    batch_size = st.sidebar.slider("Batch méret", 32, 128, 64, step=32)
    seed = st.sidebar.number_input("Seed", 0, 9999, 42)
    show_outputs = st.sidebar.checkbox("📊 Ábrák és minták megjelenítése", value=True)

    if st.button("🚀 Tanítás indítása"):
        torch.manual_seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loader = load_data(batch_size)
        generator = Generator(z_dim).to(device)
        discriminator = Discriminator().to(device)
        optim_g = optim.Adam(generator.parameters(), lr=lr)
        optim_d = optim.Adam(discriminator.parameters(), lr=lr)
        criterion = nn.BCELoss()
        g_losses, d_losses = [], []

        progress_bar = st.progress(0)
        status = st.empty()

        total_batches = len(loader)
        for epoch in range(epochs):
            for i, (real_imgs, _) in enumerate(loader):
                real_imgs = real_imgs.view(-1, 28*28).to(device)
                batch = real_imgs.size(0)
                real_labels = torch.ones(batch, 1).to(device)
                fake_labels = torch.zeros(batch, 1).to(device)

                # Diszkriminátor
                z = torch.randn(batch, z_dim).to(device)
                fake_imgs = generator(z)
                d_real = discriminator(real_imgs)
                d_fake = discriminator(fake_imgs.detach())
                loss_d = (criterion(d_real, real_labels) + criterion(d_fake, fake_labels)) / 2
                optim_d.zero_grad()
                loss_d.backward()
                optim_d.step()

                # Generátor
                z = torch.randn(batch, z_dim).to(device)
                fake_imgs = generator(z)
                d_fake = discriminator(fake_imgs)
                loss_g = criterion(d_fake, real_labels)
                optim_g.zero_grad()
                loss_g.backward()
                optim_g.step()

                g_losses.append(loss_g.item())
                d_losses.append(loss_d.item())

                progress = (epoch * total_batches + i + 1) / (epochs * total_batches)
                progress_bar.progress(progress)
                status.text(f"Epoch {epoch+1}/{epochs} | Batch {i+1}/{total_batches} | G_loss: {loss_g.item():.4f} | D_loss: {loss_d.item():.4f}")

        if show_outputs:
            st.subheader("📉 Loss görbe")
            fig, ax = plt.subplots()
            ax.plot(g_losses, label="Generátor")
            ax.plot(d_losses, label="Diszkriminátor")
            ax.set_xlabel("Iteráció")
            ax.set_ylabel("Loss")
            ax.legend()
            st.pyplot(fig)

            st.subheader("🖼️ Generált minták")
            show_images(generator, z_dim, device)

            # 3D Plotly
            z = torch.randn(1000, z_dim).to(device)
            with torch.no_grad():
                fake = generator(z).cpu().view(-1, 28*28)
            df = pd.DataFrame(fake[:, :3].numpy(), columns=["x", "y", "z"])
            st.subheader("🌈 3D Latens tér")
            fig3d = px.scatter_3d(df, x="x", y="y", z="z", color=df["z"], opacity=0.7)
            st.plotly_chart(fig3d, use_container_width=True)

        # CSV mentés (1000 mintára korlátozva)
        z = torch.randn(1000, z_dim).to(device)
        samples = generator(z).view(-1, 28*28).cpu().detach().numpy()
        df = pd.DataFrame(samples)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Minták mentése (CSV)", data=csv, file_name="gan_samples.csv")

        st.subheader("📚 Tudományos értelmezés")
        st.markdown("""
A GAN-modellek tanítási folyamata során a két hálózat (generátor és diszkriminátor) egymást próbálja túljárni – ez stabil vagy instabil tanulási görbéket eredményezhet.  
A 3D latens tér vizualizációja segít látni, hogyan szerveződnek a generált minták egy eloszlás szerint.

- A veszteségek ingadozása normális.
- A vizuálisan értékelhető minták fontosabbak, mint az abszolút loss érték.
""")

# ReflectAI-kompatibilitás
app = run
