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
    def __init__(self, z_dim=100, img_dim=28*28):
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

# Adatbetöltés
@st.cache_data
def load_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Képek megjelenítése
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

# Fő alkalmazás
def run():
    st.set_page_config(layout="wide")
    st.title("🧪 GAN – Generative Adversarial Network")

    st.markdown(r"""
A **Generative Adversarial Network (GAN)** egy két komponensből álló neurális hálózat:
- **Generátor** – új adatmintákat állít elő.
- **Diszkriminátor** – megkülönbözteti a valódi és hamis mintákat.

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_\text{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$
""")

    st.sidebar.header("🛠️ Paraméterek")
    z_dim = st.sidebar.slider("Z dimenzió", 32, 256, 64, step=16)
    lr = st.sidebar.select_slider("Tanulási ráta", options=[1e-5, 5e-5, 1e-4, 2e-4], value=2e-4)
    epochs = st.sidebar.slider("Epochok száma", 1, 10, 3)
    batch_size = st.sidebar.slider("Batch méret", 32, 256, 64, step=32)
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

        # Tréning
        for epoch in range(epochs):
            g_loss_val, d_loss_val = 0, 0
            for real_imgs, _ in loader:
                real_imgs = real_imgs.view(-1, 28*28).to(device)
                batch = real_imgs.size(0)
                real_labels = torch.ones(batch, 1).to(device)
                fake_labels = torch.zeros(batch, 1).to(device)

                z = torch.randn(batch, z_dim).to(device)
                fake_imgs = generator(z)

                # Diszkriminátor
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

                g_loss_val += loss_g.item()
                d_loss_val += loss_d.item()

            g_losses.append(g_loss_val / len(loader))
            d_losses.append(d_loss_val / len(loader))
            st.write(f"📊 Epoch {epoch+1}/{epochs} | Generator: {g_losses[-1]:.4f} | Discriminator: {d_losses[-1]:.4f}")

        if show_outputs:
            st.subheader("📉 Loss görbe")
            fig, ax = plt.subplots()
            ax.plot(g_losses, label="Generátor")
            ax.plot(d_losses, label="Diszkriminátor")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            st.pyplot(fig)

            st.subheader("🖼️ Generált minták")
            show_images(generator, z_dim, device)

            st.subheader("🌈 3D Latens tér")
            z = torch.randn(1000, z_dim).to(device)
            with torch.no_grad():
                fake = generator(z).cpu().view(-1, 28*28)
            df = pd.DataFrame(fake[:, :3].numpy(), columns=["x", "y", "z"])
            fig3d = px.scatter_3d(df, x="x", y="y", z="z", color=df["z"], opacity=0.7,
                                  title="Latens tér 3 dimenzióban")
            st.plotly_chart(fig3d, use_container_width=True)

        # CSV mentés
        z = torch.randn(1000, z_dim).to(device)
        samples = generator(z).view(-1, 28*28).cpu().detach().numpy()
        df = pd.DataFrame(samples)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Minták mentése (CSV)", data=csv, file_name="gan_samples.csv")

        st.subheader("📚 Tudományos értelmezés")
        st.markdown("""
A GAN-modellek iteratív tanítása során a diszkriminátor és a generátor „versengve” fejlődnek.  
A veszteségértékek és a vizuális minőség is mutatja, mennyire sikeres a hamis minták létrehozása.

- A generátor tanulási görbéje csökkenő trendet mutat → sikeres mintagenerálás.
- A diszkriminátor vesztesége emelkedhet, ha nehézséget okoz a valódi és generált adatok megkülönböztetése.
- A 3D scatter plot a minták sűrűségét és szerkezetét szemlélteti a latens tér reprezentációjában.
        """)

# ReflectAI-kompatibilitás
app = run
