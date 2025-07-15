import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class G(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def plot_generated(generator, z_dim, device):
    generator.eval()
    z = torch.randn(16, z_dim).to(device)
    imgs = generator(z).view(-1,1,28,28).cpu()
    grid = make_grid(imgs, nrow=4, normalize=True)
    fig = plt.figure(figsize=(4,4))
    plt.axis('off')
    plt.imshow(grid.permute(1,2,0))
    st.pyplot(fig)

def run():
    st.title("🧪 GAN Lab (MINI)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_dim = st.sidebar.slider("Z-dim", 64, 128, 100)
    lr = st.sidebar.select_slider("Learning rate", [1e-4,2e-4,5e-4], value=2e-4)
    epochs = st.sidebar.slider("Epochs",1,10,3)
    batch = st.sidebar.slider("Batch",32,128,64)
    seed = st.sidebar.number_input("Seed",0,9999,42)

    if st.button("Train"):
        torch.manual_seed(seed)
        ds = datasets.MNIST(".", True, transform=transforms.ToTensor(), download=True)
        loader = DataLoader(ds, batch_size=batch, shuffle=True)

        gen = G(z_dim).to(device)
        disc = D().to(device)
        opt_g = optim.Adam(gen.parameters(), lr=lr)
        opt_d = optim.Adam(disc.parameters(), lr=lr)
        crit = nn.BCELoss()

        for ep in range(1, epochs+1):
            for real,_ in loader:
                real = real.view(-1,28*28).to(device)
                bs = real.size(0)
                real_lbl = torch.ones(bs,1).to(device)
                fake_lbl = torch.zeros(bs,1).to(device)

                # D lépés
                z = torch.randn(bs, z_dim).to(device)
                fake = gen(z)
                loss_d = (crit(disc(real), real_lbl) + crit(disc(fake.detach()), fake_lbl))/2
                opt_d.zero_grad(); loss_d.backward(); opt_d.step()

                # G lépés
                z = torch.randn(bs, z_dim).to(device)
                fake = gen(z)
                loss_g = crit(disc(fake), real_lbl)
                opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            st.write(f"Epoch {ep}/{epochs} | G: {loss_g.item():.4f} | D: {loss_d.item():.4f}")

        st.subheader("Generated samples")
        plot_generated(gen, z_dim, device)

app = run
