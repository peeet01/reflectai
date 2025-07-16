import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

# --- GAN komponensek ---
class Generator(nn.Module):
    def __init__(self, z_dim=64, img_dim=28 * 28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, img_dim=28 * 28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# --- Mintagenerálás ---
def show_generated_images(generator, z_dim, device, num_samples=16):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, z_dim).to(device)
        samples = generator(z).view(-1, 28, 28).cpu()
    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i], cmap="gray")
        ax.axis("off")
    st.pyplot(fig)

# --- App futás ---
def run():
    st.markdown("""
        # 🧪 GAN – Generative Adversarial Network

        A Generative Adversarial Network (GAN) két modellből áll:

        - **Generátor**: új adatmintákat generál a zajból  
        - **Diszkriminátor**: eldönti, hogy a bemenő kép valós vagy hamis

        A cél: a generátor megtanuljon olyan jól hamisítani, hogy a diszkriminátor ne tudjon különbséget tenni.

        $$
        \\min_G \\max_D V(D, G) = \\mathbb{E}_{x \\sim p_{data}}[\\log D(x)] + \\mathbb{E}_{z \\sim p_z}[\\log(1 - D(G(z)))]
        $$

        A veszteségértékek alapján a generátor és a diszkriminátor fokozatosan tanulnak.
        Bár a generált minták még nem élethűek, a hálózat elindult a jó irányba.
        Több epoch és finomhangolás segítene a minőség javításában.
    """)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Beállítások ---
    st.sidebar.header("Beállítások")
    z_dim = st.sidebar.slider("Z dimenzió", 32, 128, 64, step=16)
    lr = st.sidebar.select_slider("Tanulási ráta", options=[1e-5, 5e-5, 1e-4, 2e-4], value=2e-4)
    epochs = st.sidebar.slider("Epochok száma", 1, 20, 5)
    batch_size = st.sidebar.slider("Batch méret", 16, 128, 32, step=16)
    seed = st.sidebar.number_input("Seed", 0, 9999, 42)

    if st.button("🚀 Tanítás indítása"):
        torch.manual_seed(seed)
        generator = Generator(z_dim=z_dim).to(device)
        discriminator = Discriminator().to(device)

        optim_g = torch.optim.Adam(generator.parameters(), lr=lr)
        optim_d = torch.optim.Adam(discriminator.parameters(), lr=lr)
        criterion = nn.BCELoss()

        g_losses, d_losses = [], []

        for epoch in range(epochs):
            real = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

            # --- Diszkriminátor ---
            z = torch.randn(batch_size, z_dim).to(device)
            fake_imgs = generator(z)
            real_imgs = torch.randn(batch_size, 28 * 28).to(device)

            loss_real = criterion(discriminator(real_imgs), real)
            loss_fake = criterion(discriminator(fake_imgs.detach()), fake)
            loss_d = (loss_real + loss_fake) / 2

            optim_d.zero_grad()
            loss_d.backward()
            optim_d.step()

            # --- Generátor ---
            z = torch.randn(batch_size, z_dim).to(device)
            fake_imgs = generator(z)
            loss_g = criterion(discriminator(fake_imgs), real)

            optim_g.zero_grad()
            loss_g.backward()
            optim_g.step()

            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())

            st.markdown(f"📊 **Epoch {epoch+1}/{epochs}** | Generator: {loss_g.item():.4f} | Diszkriminátor: {loss_d.item():.4f}")

        # --- Loss ábra ---
        st.subheader("📉 Loss alakulása")
        fig, ax = plt.subplots()
        ax.plot(g_losses, label="Generátor")
        ax.plot(d_losses, label="Diszkriminátor")
        ax.legend()
        st.pyplot(fig)

        # --- Minták ---
        st.subheader("🖼️ Generált minták")
        show_generated_images(generator, z_dim, device)

app = run
