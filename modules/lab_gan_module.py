import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Generator hálózat
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# Discriminator hálózat
class Discriminator(nn.Module):
    def __init__(self, img_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def run():
    st.title("🧪 GAN Lab – simple test")
    st.markdown("✅ Modul betöltődött, egy tanítási lépés elérhető.")

    z_dim = 100
    img_dim = 28 * 28
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = Generator(z_dim, img_dim).to(device)
    discriminator = Discriminator(img_dim).to(device)

    if st.button("Run one training step"):
        z = torch.randn((1, z_dim)).to(device)
        fake_img = generator(z)

        prediction = discriminator(fake_img)
        loss = nn.BCELoss()(prediction, torch.ones_like(prediction))

        st.write("🧪 Prediction:", prediction.item())
        st.write("📉 Loss:", loss.item())

        # Vizualizáció
        fake_img_reshaped = fake_img.view(28, 28).detach().cpu().numpy()
        fig, ax = plt.subplots()
        ax.imshow(fake_img_reshaped, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

app = run
