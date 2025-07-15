import streamlit as st import torch import torch.nn as nn import torch.optim as optim from torchvision import datasets, transforms from torch.utils.data import DataLoader from torchvision.utils import make_grid import matplotlib.pyplot as plt import pandas as pd

--- 🎯 Generátor ---

class Generator(nn.Module): def init(self, z_dim=100, img_dim=28*28): super().init() self.net = nn.Sequential( nn.Linear(z_dim, 256), nn.ReLU(True), nn.Linear(256, 512), nn.ReLU(True), nn.Linear(512, img_dim), nn.Tanh() )

def forward(self, x):
    return self.net(x)

--- ❌ Diszkriminátor ---

class Discriminator(nn.Module): def init(self, img_dim=28*28): super().init() self.net = nn.Sequential( nn.Linear(img_dim, 512), nn.LeakyReLU(0.2), nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1), nn.Sigmoid() )

def forward(self, x):
    return self.net(x)

--- 🖼️ Képmegjelenítő ---

def show_generated_images(generator, z_dim, device): generator.eval() with torch.no_grad(): z = torch.randn(16, z_dim).to(device) fake_imgs = generator(z).view(-1, 1, 28, 28).cpu() grid = make_grid(fake_imgs, nrow=4, normalize=True) fig, ax = plt.subplots(figsize=(4, 4)) ax.imshow(grid.permute(1, 2, 0)) ax.axis("off") st.pyplot(fig)

🚀 Streamlit futtatás

def run(): st.set_page_config(layout="wide") st.title("🤪 GAN Lab – Generative Adversarial Network")

st.markdown("""

A GAN egy generatív modell, amely két hálózatot tanít egymás ellen:

Generátor (G): képeket próbál generálni.

Diszkriminátor (D): megpróbálja eldönteni, mi igazi vagy hamis. """)

device = "cuda" if torch.cuda.is_available() else "cpu"

🛠️ Paraméterek

st.sidebar.header("🛠️ Paraméterek") z_dim = st.sidebar.slider("Z dimenzió", 64, 256, 100, step=16) lr = st.sidebar.select_slider("Tanulási ráta", options=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3], value=2e-4) epochs = st.sidebar.slider("Epochok", 1, 30, 5) batch_size = st.sidebar.slider("Batch méret", 32, 256, 128, step=32) seed = st.sidebar.number_input("Seed", 0, 9999, 42)

if st.button("▶️ Tanítás indítása"): torch.manual_seed(seed)

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
      st.text(f"Epoch {epoch+1}/{epochs} | G: {loss_g.item():.4f} | D: {loss_d.item():.4f}")

  st.subheader("📉 Veszteség")
  fig, ax = plt.subplots()
  ax.plot(g_losses, label="Generátor")
  ax.plot(d_losses, label="Diszkriminátor")
  ax.legend()
  st.pyplot(fig)

  st.subheader("🔼 Generált képek")
  show_generated_images(generator, z_dim, device)

  st.subheader("📅 Export")
  z = torch.randn(100, z_dim).to(device)
  samples = generator(z).view(-1, 28*28).cpu().detach().numpy()
  df = pd.DataFrame(samples)
  csv = df.to_csv(index=False).encode("utf-8")
  st.download_button("Letöltés CSV-ben", data=csv, file_name="gan_samples.csv")


ReflectAI kompatibilitás

app = run

