import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pandas as pd

# --- GAN komponensek ---
class Generator(nn.Module):
    def __init__(self, z_dim=64, img_dim=28 * 28):
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

class Discriminator(nn.Module):
    def __init__(self, img_dim=28 * 28):
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

def show_generated_images(generator, z_dim, device):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(16, z_dim).to(device)
        fake_imgs = generator(z).view(-1, 1, 28, 28).cpu()
        grid = make_grid(fake_imgs, nrow=4, normalize=True)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(grid.permute(1, 2, 0))
        ax.axis("off")
        st.pyplot(fig)

def run():
    st.set_page_config(layout="wide")
    st.title("🧪 GAN – Generative Adversarial Network")

    st.markdown("""
    A Generative Adversarial Network (GAN) két modellből áll:

    - **Generátor**: új adatminta generálása a bemeneti zajból  
    - **Diszkriminátor**: megpróbálja eldönteni, hogy egy minta valós vagy hamis

    A cél, hogy a generátor olyan jól tanuljon, hogy a diszkriminátor ne tudjon különbséget tenni.

    A GAN célfüggvénye:  
    $$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))] $$
    """)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    st.sidebar.header("Beállítások")
    z_dim = st.sidebar.slider("Z dimenzió", 32, 128, 64, step=16)
    lr = st.sidebar.select_slider("Tanulási ráta", options=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3], value=2e-4)
    epochs = st.sidebar.slider("Epochok száma", 1, 20, 5)
    batch_size = st.sidebar.slider("Batch méret", 16, 128, 32, step=16)
    seed = st.sidebar.number_input("Seed", 0, 9999, 42)

    if st.button("🚀 Tanítás indítása"):
        torch.manual_seed(seed)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        subset = Subset(dataset, range(2000))  # csak 2000 minta
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        generator = Generator(z_dim).to(device)
        discriminator = Discriminator().to(device)
        optim_g = optim.Adam(generator.parameters(), lr=lr)
        optim_d = optim.Adam(discriminator.parameters(), lr=lr)
        criterion = nn.BCELoss()

        g_losses, d_losses = [], []

        for epoch in range(epochs):
            for real_imgs, _ in loader:
                real_imgs = real_imgs.view(-1, 28 * 28).to(device)
                batch = real_imgs.size(0)
                real = torch.ones(batch, 1).to(device)
                fake = torch.zeros(batch, 1).to(device)

                # Diszkriminátor
                z = torch.randn(batch, z_dim).to(device)
                fake_imgs = generator(z)
                loss_real = criterion(discriminator(real_imgs), real)
                loss_fake = criterion(discriminator(fake_imgs.detach()), fake)
                loss_d = (loss_real + loss_fake) / 2
                optim_d.zero_grad()
                loss_d.backward()
                optim_d.step()

                # Generátor
                z = torch.randn(batch, z_dim).to(device)
                fake_imgs = generator(z)
                loss_g = criterion(discriminator(fake_imgs), real)
                optim_g.zero_grad()
                loss_g.backward()
                optim_g.step()

            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())
            st.write(f"📊 Epoch {epoch+1}/{epochs} | Generator: {loss_g.item():.4f} | Diszkriminátor: {loss_d.item():.4f}")

        st.subheader("📉 Loss alakulása")
        fig, ax = plt.subplots()
        ax.plot(g_losses, label="Generátor")
        ax.plot(d_losses, label="Diszkriminátor")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss érték")
        ax.legend()
        st.pyplot(fig)

        st.subheader("🖼️ Generált minták")
        show_generated_images(generator, z_dim, device)

        z = torch.randn(100, z_dim).to(device)
        samples = generator(z).view(-1, 28 * 28).cpu().detach().numpy()
        df = pd.DataFrame(samples)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Minták letöltése (CSV)", data=csv, file_name="gan_samples.csv")

        st.subheader("🧠 Tudományos magyarázat")
        st.markdown("""
        A GAN célja, hogy egy generátor modell megtanuljon a bemeneti zajból valósághű adatmintákat előállítani, miközben a diszkriminátor próbálja felismerni, hogy mi valódi, mi hamis.

        A két hálózat egymással versengve fejlődik. Ha a diszkriminátor túl jó, a generátor nem tanul. Ha a generátor túljár az eszén, a diszkriminátor tanulása gyengül.

        Az egyensúlyi állapot célja: a generátor olyan jó, hogy a diszkriminátor 50%-os arányban téved – tehát *nem tud különbséget tenni valós és generált között*.
        """)

app = run
