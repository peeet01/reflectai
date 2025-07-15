import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pandas as pd

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

    - **Generátor**: új adatminta generálása a zajból  
    - **Diszkriminátor**: eldönti, hogy a bemenő kép valódi vagy hamis

    A cél: a generátor megtanuljon olyan jól hamisítani, hogy a diszkriminátor ne tudjon különbséget tenni.

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
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        generator = Generator(z_dim).to(device)
        discriminator = Discriminator().to(device)
        optim_g = optim.Adam(generator.parameters(), lr=lr)
        optim_d = optim.Adam(discriminator.parameters(), lr=lr)
        criterion = nn.BCELoss()

        g_losses, d_losses = [], []
        progress = st.progress(0.0, text="Tanítás indult...")

        for epoch in range(epochs):
            for real_imgs, _ in loader:
                real_imgs = real_imgs.view(-1, 28 * 28).to(device)
                batch = real_imgs.size(0)
                real = torch.ones(batch, 1).to(device)
                fake = torch.zeros(batch, 1).to(device)

                z = torch.randn(batch, z_dim).to(device)
                fake_imgs = generator(z)
                loss_real = criterion(discriminator(real_imgs), real)
                loss_fake = criterion(discriminator(fake_imgs.detach()), fake)
                loss_d = (loss_real + loss_fake) / 2
                optim_d.zero_grad()
                loss_d.backward()
                optim_d.step()

                z = torch.randn(batch, z_dim).to(device)
                fake_imgs = generator(z)
                loss_g = criterion(discriminator(fake_imgs), real)
                optim_g.zero_grad()
                loss_g.backward()
                optim_g.step()

            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())
            st.markdown(f"📊 **Epoch {epoch+1}/{epochs}** | Generator: {loss_g.item():.4f} | Diszkriminátor: {loss_d.item():.4f}")
            progress.progress((epoch + 1) / epochs, text=f"Folyamat: {epoch+1}/{epochs}")

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

        st.subheader("🧠 Tudományos értékelés")
        st.markdown("""
        A veszteségértékek változása alapján látható, hogy a generátor és diszkriminátor kiegyensúlyozottan fejlődnek.

        A generált minták még nem tökéletesek, de bizonyos mintázatok kezdődnek kialakulni, ami azt jelenti, hogy a hálózat tanulási folyamata elindult.

        A további epochok és finomhangolás valószínűleg tovább javítja majd a minőséget.
        """)

app = run
