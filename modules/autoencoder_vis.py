import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import numpy as np
from torch.utils.data import DataLoader, Subset

# -----------------------------
# Hasznos segédfüggvények
# -----------------------------
@st.cache_data
def load_mnist_datasets(download=True):
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST('./data', train=True, download=download, transform=transform)
    test_ds  = datasets.MNIST('./data', train=False, download=download, transform=transform)
    return len(train_ds), len(test_ds)

def subset_loader(train=True, n_items=3000, batch_size=128, shuffle=True, seed=42):
    transform = transforms.ToTensor()
    ds = datasets.MNIST('./data', train=train, download=True, transform=transform)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n_items, len(ds)), replace=False)
    sub = Subset(ds, idx)
    return DataLoader(sub, batch_size=batch_size, shuffle=shuffle)

def psnr_from_mse(mse):
    if mse <= 0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)

def to_cpu_numpy(t):
    return t.detach().cpu().numpy()

def pca_to_3d(Z):
    Z = np.asarray(Z, dtype=np.float64)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    if Vt.shape[0] < 3:
        pad = np.zeros((3 - Vt.shape[0], Vt.shape[1]), dtype=Vt.dtype)
        Vt = np.vstack([Vt, pad])
    Z3 = Zc @ Vt[:3].T
    return Z3

# -----------------------------
# Modellek
# -----------------------------
class MLP_AE(nn.Module):
    def __init__(self, latent_dim=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 28*28), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        xr = self.decoder(z).view(-1,1,28,28)
        return xr, z

class Conv_AE(nn.Module):
    def __init__(self, latent_dim=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), nn.ReLU(),
            nn.Conv2d(32,64,4,2,1), nn.ReLU()
        )
        self.enc_lin = nn.Linear(64*7*7, latent_dim)
        self.dec_lin = nn.Linear(latent_dim, 64*7*7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,1,4,2,1), nn.Sigmoid()
        )

    def forward(self, x):
        h = self.enc(x).view(x.size(0), -1)
        z = self.enc_lin(h)
        h = self.dec_lin(z).view(-1,64,7,7)
        xr = self.dec(h)
        return xr, z

# --------- Variational AE (MLP alapú) ---------
class MLP_VAE(nn.Module):
    def __init__(self, latent_dim=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 28*28), nn.Sigmoid()
        )

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparam(mu, logvar)
        xr = self.dec(z).view(-1,1,28,28)
        return xr, z, mu, logvar

class Conv_VAE(nn.Module):
    def __init__(self, latent_dim=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), nn.ReLU(),
            nn.Conv2d(32,64,4,2,1), nn.ReLU()
        )
        self.enc_lin = nn.Linear(64*7*7,128)
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        self.dec_lin = nn.Linear(latent_dim, 64*7*7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,1,4,2,1), nn.Sigmoid()
        )

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h = self.enc(x).view(x.size(0), -1)
        h = self.enc_lin(h)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparam(mu, logvar)
        h = self.dec_lin(z).view(-1,64,7,7)
        xr = self.dec(h)
        return xr, z, mu, logvar

# -----------------------------
# Loss: AE vagy VAE
# -----------------------------
def vae_loss(xr, x, mu, logvar, beta=1.0):
    recon = nn.functional.mse_loss(xr, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon + beta*kl)/x.size(0), recon.item()/x.size(0), kl.item()/x.size(0)

def ae_loss(xr, x):
    return nn.functional.mse_loss(xr, x)

# -----------------------------
# Streamlit modul
# -----------------------------
def app():
    st.set_page_config(layout="wide")
    st.title("🧠 Autoencoder & Variational Autoencoder – 3D latens tér")

    n_train_full, n_test_full = load_mnist_datasets()

    st.sidebar.header("⚙️ Beállítások")
    arch = st.sidebar.selectbox("Architektúra", ["MLP AE", "Conv AE", "MLP VAE", "Conv VAE"])
    latent_dim = st.sidebar.slider("Latens dimenzió", 2, 8, 3)
    beta = st.sidebar.slider("β (KL súly, csak VAE esetén)", 0.1, 5.0, 1.0, 0.1)
    lr = st.sidebar.select_slider("Tanulási ráta", options=[5e-4, 1e-3, 2e-3], value=1e-3)
    batch_size = st.sidebar.slider("Batch méret", 64, 512, 128, 64)
    epochs = st.sidebar.slider("Epochok", 1, 15, 6)

    n_train = st.sidebar.number_input("Train képek", 500, n_train_full, 3000, step=500)
    n_test = st.sidebar.number_input("Test képek", 500, n_test_full, 1000, step=500)
    seed = st.sidebar.number_input("Seed", 0, 9999, 42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    if arch=="MLP AE":
        model = MLP_AE(latent_dim).to(device)
        use_vae=False
    elif arch=="Conv AE":
        model = Conv_AE(latent_dim).to(device)
        use_vae=False
    elif arch=="MLP VAE":
        model = MLP_VAE(latent_dim).to(device)
        use_vae=True
    else:
        model = Conv_VAE(latent_dim).to(device)
        use_vae=True

    train_loader = subset_loader(True, n_train, batch_size, True, seed)
    test_loader = subset_loader(False, n_test, batch_size, False, seed)

    if st.button("🚀 Tanítás és kiértékelés"):
        opt = optim.Adam(model.parameters(), lr=lr)
        history=[]
        for ep in range(epochs):
            model.train()
            run_loss=0
            for x,_ in train_loader:
                x=x.to(device)
                opt.zero_grad()
                if use_vae:
                    xr,z,mu,logvar = model(x)
                    loss,recon,kl = vae_loss(xr,x,mu,logvar,beta)
                else:
                    xr,z = model(x)
                    loss = ae_loss(xr,x)
                loss.backward()
                opt.step()
                run_loss+=loss.item()
            history.append(run_loss/len(train_loader))
            st.write(f"Epoch {ep+1}/{epochs} Loss: {history[-1]:.4f}")

        # Rekonstrukciós hiba görbe
        fig,ax=plt.subplots()
        ax.plot(history,marker='o')
        ax.set_xlabel("Epoch"); ax.set_ylabel("Train Loss")
        st.pyplot(fig)

        # Első batch kiértékelés
        x,y = next(iter(test_loader))
        x=x.to(device)
        with torch.no_grad():
            if use_vae:
                xr,z,mu,logvar = model(x)
            else:
                xr,z = model(x)

        st.subheader("🖼️ Eredeti vs. Rekonstrukció")
        n_show=10
        grid_orig = make_grid(x[:n_show].cpu(),nrow=n_show,normalize=True)
        grid_reco = make_grid(xr[:n_show].cpu(),nrow=n_show,normalize=True)
        fig,axes=plt.subplots(2,1,figsize=(n_show,3))
        axes[0].imshow(grid_orig.permute(1,2,0)); axes[0].axis("off"); axes[0].set_title("Eredeti")
        axes[1].imshow(grid_reco.permute(1,2,0)); axes[1].axis("off"); axes[1].set_title("Rekonstrukció")
        st.pyplot(fig)

        # Latens tér vizualizáció
        st.subheader("🌌 Latens tér")
        Z = to_cpu_numpy(z)
        if Z.shape[1]==3:
            Z3=Z
        else:
            Z3=pca_to_3d(Z)
        df=pd.DataFrame(Z3,columns=["z1","z2","z3"])
        df["label"]=y.numpy()
        fig3d=px.scatter_3d(df,x="z1",y="z2",z="z3",color=df["label"].astype(str))
        st.plotly_chart(fig3d,use_container_width=True)

        st.markdown("### 📚 Tudományos háttér")
        if use_vae:
            st.latex(r"\mathcal{L} = \|x-\hat{x}\|^2 + \beta D_{KL}(q(z|x)\,\|\,p(z))")
            st.markdown("A VAE folytonos, szabályozott latens teret tanul → jobb generatív képesség.")
        else:
            st.latex(r"\mathcal{L} = \|x-\hat{x}\|^2")
            st.markdown("Az AE dimenziócsökkentésre és vizualizációra alkalmas, de kevésbé generatív.")

# ReflectAI-kompatibilis
app = app
