import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Generátor
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Diszkriminátor
class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Egy tanítási lépés
def train_step(generator, discriminator, g_opt, d_opt, real_imgs, z_dim, device):
    batch_size = real_imgs.size(0)
    real_imgs = real_imgs.view(batch_size, -1).to(device)
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    # Diszkriminátor
    z = torch.randn(batch_size, z_dim).to(device)
    fake_imgs = generator(z)
    real_loss = nn.BCELoss()(discriminator(real_imgs), real_labels)
    fake_loss = nn.BCELoss()(discriminator(fake_imgs.detach()), fake_labels)
    d_loss = (real_loss + fake_loss) / 2
    d_opt.zero_grad()
    d_loss.backward()
    d_opt.step()

    # Generátor
    z = torch.randn(batch_size, z_dim).to(device)
    fake_imgs = generator(z)
    g_loss = nn.BCELoss()(discriminator(fake_imgs), real_labels)
    g_opt.zero_grad()
    g_loss.backward()
    g_opt.step()

    return d_loss.item(), g_loss.item(), fake_imgs

# Fő app
def run():
    st.title("🧪 GAN Lab — Mini")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_dim = 100

    # Modellek és optimizálók
    generator = Generator(z_dim).to(device)
    discriminator = Discriminator().to(device)
    g_opt = optim.Adam(generator.parameters(), lr=0.0002)
    d_opt = optim.Adam(discriminator.parameters(), lr=0.0002)

    # Adatok
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(data, batch_size=64, shuffle=True)
    real_imgs, _ = next(iter(loader))

    if st.button("Run 1 train step"):
        d_loss, g_loss, samples = train_step(generator, discriminator, g_opt, d_opt, real_imgs, z_dim, device)

        st.write(f"📉 Loss D: {d_loss:.4f} | G: {g_loss:.4f}")

        # Képek megjelenítése
        samples = samples[:16].view(-1, 1, 28, 28).cpu().detach()
        grid = make_grid(samples, nrow=4, normalize=True)
        fig, ax = plt.subplots()
        ax.imshow(grid.permute(1, 2, 0))
        ax.axis("off")
        st.pyplot(fig)

# ReflectAI kompatibilitás
app = run
