import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pandas as pd

# Generator hálózat
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

# Discriminator hálózat
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

# Generált képek megjelenítése
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

# Fő alkalmazás
def run():
    st.set_page_config(layout="wide")
    st.title("🧪 GAN – Generative Adversarial Network")

    st.markdown(r"""
A Generative Adversarial Network (GAN) két modellből áll:

- **Generátor**: új adatmintákat generál a bemeneti zajból.
- **Diszkriminátor**: megpróbálja eldönteni, hogy egy minta valós vagy hamis.

A cél, hogy a generátor olyan jól tanuljon, hogy a diszkriminátor ne tudjon különbséget tenni.

A GAN célfüggvénye:

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_\text{data}} [\log D(x)] + \mathbb{E}_{z \sim p_z} [\log(1 - D(G(z)))]
$$
""")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    st.sidebar.header("🛠️ Paraméterek")
    z_dim = st.sidebar.slider("Z dimenzió", 32, 256, 64, step=16)
    lr = st.sidebar.select_slider("Tanulási ráta", options=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4], value=2e-4)
    epochs = st.sidebar.slider("Epochok száma", 1, 20, 3)
    batch_size = st.sidebar.slider("Batch méret", 32, 256, 64, step=32)
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

        for epoch in range(epochs):
            for real_imgs, _ in loader:
                real_imgs = real_imgs.view(-1, 28*28).to(device)
                batch = real_imgs.size(0)
                real_labels = torch.ones(batch, 1).to(device)
                fake_labels = torch.zeros(batch, 1).to(device)

                # Diszkriminátor lépés
                z = torch.randn(batch, z_dim).to(device)
                fake_imgs = generator(z)
                d_real = discriminator(real_imgs)
                d_fake = discriminator(fake_imgs.detach())
                loss_d = (criterion(d_real, real_labels) + criterion(d_fake, fake_labels)) / 2

                optim_d.zero_grad()
                loss_d.backward()
                optim_d.step()

                # Generátor lépés
                z = torch.randn(batch, z_dim).to(device)
                fake_imgs = generator(z)
                d_fake = discriminator(fake_imgs)
                loss_g = criterion(d_fake, real_labels)

                optim_g.zero_grad()
                loss_g.backward()
                optim_g.step()

            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())
            st.write(f"📊 Epoch {epoch+1}/{epochs} | G: {loss_g.item():.4f} | D: {loss_d.item():.4f}")

        # Loss görbék
        st.subheader("📉 Loss")
        fig, ax = plt.subplots()
        ax.plot(g_losses, label="Generátor")
        ax.plot(d_losses, label="Diszkriminátor")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss érték")
        ax.legend()
        st.pyplot(fig)

        # Minták
        st.subheader("🖼️ Minták")
        show_images(generator, z_dim, device)

        # 🎲 Új minták külön gombbal
        st.subheader("🔁 Új minták generálása")
        if st.button("🎲 Generálj új mintákat"):
            show_images(generator, z_dim, device)

        # 💾 Loss értékek mentése CSV-be
        st.subheader("📁 Loss értékek letöltése")
        df_loss = pd.DataFrame({
            "Generátor loss": g_losses,
            "Diszkriminátor loss": d_losses
        })
        csv_loss = df_loss.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Loss letöltése (CSV)",
            data=csv_loss,
            file_name="gan_loss.csv"
        )

        # 🧠 Tudományos megjegyzés
        st.subheader("🧠 Tudományos megjegyzés")
        st.markdown("""
A GAN egyensúlyi játéka nehezen tanítható, de már néhány epoch után is megfigyelhetők mintázatok.  
A diszkriminátor loss értékének stabilizálódása és a generált minták minősége mutatja a tanulás előrehaladását.
        """)

# ReflectAI-kompatibilis
app = run
