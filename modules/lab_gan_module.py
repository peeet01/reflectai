import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import pandas as pd
import io

# -----------------------------
# Generator hálózat
# -----------------------------
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

# -----------------------------
# Discriminator hálózat
# -----------------------------
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

# -----------------------------
# Generált képek megjelenítése
# -----------------------------
@st.cache_resource
def show_images(_generator, z_dim, device):
    _generator.eval()
    with torch.no_grad():
        z = torch.randn(16, z_dim).to(device)
        fake_imgs = _generator(z).view(-1, 1, 28, 28).cpu()
        grid = make_grid(fake_imgs, nrow=4, normalize=True)
        fig, ax = plt.subplots()
        ax.imshow(grid.permute(1, 2, 0))
        ax.axis("off")
        st.pyplot(fig)

        buffer = io.BytesIO()
        save_image(fake_imgs, buffer, format='png')
        st.download_button("⬇️ Minták letöltése (PNG)", data=buffer.getvalue(), file_name="samples.png", mime="image/png")

# -----------------------------
# Adatok betöltése
# -----------------------------
@st.cache_data
def load_data(batch_size, limit=1000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    subset = torch.utils.data.Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(subset, batch_size=batch_size, shuffle=True)

# -----------------------------
# Fő alkalmazás
# -----------------------------
def run():
    st.set_page_config(layout="wide")
    st.title("🧪 GAN – Generative Adversarial Network")

    st.markdown(r"""
A **Generative Adversarial Network (GAN)** egy neurális architektúra, amely két egymással versengő hálózatot – egy *generátort* és egy *diszkriminátort* – használ.  
Cél: olyan hamis mintákat generálni, melyek megkülönböztethetetlenek a valódiaktól.

### 🎓 Matematikai háttér:
$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

GAN-ok használata:
- Képgenerálás
- Stílusátvitel
- Szuperfelbontás
- Deepfake technológiák
""")

    # Paraméterek
    st.sidebar.header("🛠️ Paraméterek")
    z_dim = st.sidebar.slider("Z dimenzió", 32, 256, 64, step=16)
    lr = st.sidebar.select_slider("Tanulási ráta", options=[1e-5, 5e-5, 1e-4, 2e-4], value=2e-4)
    epochs = st.sidebar.slider("Epochok száma", 1, 20, 3)
    batch_size = st.sidebar.slider("Batch méret", 32, 256, 64, step=32)
    seed = st.sidebar.number_input("Seed", 0, 9999, 42)
    show_outputs = st.sidebar.checkbox("📊 Eredmények megjelenítése", value=True)

    if st.button("🚀 Tanítás indítása"):
        torch.manual_seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loader = load_data(batch_size=batch_size, limit=1000)  # ❗ max 1000 kép

        generator = Generator(z_dim).to(device)
        discriminator = Discriminator().to(device)
        optim_g = optim.Adam(generator.parameters(), lr=lr)
        optim_d = optim.Adam(discriminator.parameters(), lr=lr)
        criterion = nn.BCELoss()

        g_losses, d_losses = [], []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for epoch in range(epochs):
            for real_imgs, _ in loader:
                real_imgs = real_imgs.view(-1, 28*28).to(device)
                batch = real_imgs.size(0)
                real_labels = torch.ones(batch, 1).to(device)
                fake_labels = torch.zeros(batch, 1).to(device)

                # Diszkriminátor tanítása
                z = torch.randn(batch, z_dim).to(device)
                fake_imgs = generator(z)
                d_real = discriminator(real_imgs)
                d_fake = discriminator(fake_imgs.detach())
                loss_d = (criterion(d_real, real_labels) + criterion(d_fake, fake_labels)) / 2

                optim_d.zero_grad()
                loss_d.backward()
                optim_d.step()

                # Generátor tanítása
                z = torch.randn(batch, z_dim).to(device)
                fake_imgs = generator(z)
                d_fake = discriminator(fake_imgs)
                loss_g = criterion(d_fake, real_labels)

                optim_g.zero_grad()
                loss_g.backward()
                optim_g.step()

            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"Epoch {epoch+1}/{epochs} | Generator: {loss_g.item():.4f} | Discriminator: {loss_d.item():.4f}")

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
            show_images(_generator=generator, z_dim=z_dim, device=device)

        # 🔻 CSV export (minták)
        z = torch.randn(100, z_dim).to(device)
        samples = generator(z).view(-1, 28*28).cpu().detach().numpy()
        df = pd.DataFrame(samples)
        filename = f"gan_samples_z{z_dim}_e{epochs}_b{batch_size}.csv"
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Minták mentése (CSV)", data=csv, file_name=filename)

        # 📚 Tudományos értelmezés
        st.subheader("📚 Tudományos értelmezés")
        st.markdown("""
A veszteséggörbék alapján megfigyelhető a tanulási dinamika:
- A generátor loss csökkenése a minta minőség javulását mutatja.
- A diszkriminátor loss növekedése a hamis minták megtévesztőbbé válását jelzi.
- A kiegyenlített fejlődés stabil tanulásra utal.

A GAN tanítása érzékeny a hiperparaméterekre, és nem garantált a konvergencia.  
A jelen példa célja az **alapelvek demonstrálása 1000 mintán**.
""")


# ✅ ReflectAI kompatibilitás
app = run
