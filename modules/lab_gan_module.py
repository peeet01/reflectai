import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Egyszerű Generátor
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

# Futás
def run():
    st.title("🧪 GAN – Ultra Light Test")

    z_dim = 100
    generator = Generator(z_dim)
    z = torch.randn(16, z_dim)
    fake_imgs = generator(z).view(-1, 1, 28, 28)

    # Megjelenítés
    grid = make_grid(fake_imgs, nrow=4, normalize=True)
    fig, ax = plt.subplots()
    ax.imshow(grid.permute(1, 2, 0))
    ax.axis("off")
    st.pyplot(fig)

# ReflectAI-kompatibilis
app = run
