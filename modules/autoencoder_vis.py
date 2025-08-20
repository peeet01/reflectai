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
    return DataLoader(sub, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)

def psnr_from_mse(mse):
    # képek 0..1 skálán → MAX_I = 1
    if mse <= 0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)

def to_cpu_numpy(t):
    return t.detach().cpu().numpy()

def pca_to_3d(Z):
    # Z: [N, d] → PCA 3 dimenzióra (numpy SVD)
    Z = np.asarray(Z, dtype=np.float64)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    if Vt.shape[0] < 3:
        # kevés dimenzió esetén nullákkal töltünk
        pad = np.zeros((3 - Vt.shape[0], Vt.shape[1]), dtype=Vt.dtype)
        Vt = np.vstack([Vt, pad])
    Z3 = Zc @ Vt[:3].T
    return Z3

# -----------------------------
# Modellek
# -----------------------------
class MLP_Autoencoder(nn.Module):
    def __init__(self, latent_dim=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 28*28),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        x_flat = self.decoder(z)
        return x_flat.view(-1, 1, 28, 28)

    def forward(self, x):
        z = self.encode(x)
        xr = self.decode(z)
        return xr, z

class Conv_Autoencoder(nn.Module):
    def __init__(self, latent_dim=3):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder: 28x28 -> 14x14 -> 7x7
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 28->14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 14->7
            nn.ReLU(inplace=True),
        )
        self.enc_lin = nn.Linear(64*7*7, latent_dim)
        # Decoder
        self.dec_lin = nn.Linear(latent_dim, 64*7*7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 7->14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 14->28
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(x.size(0), -1)
        z = self.enc_lin(h)
        return z

    def decode(self, z):
        h = self.dec_lin(z).view(-1, 64, 7, 7)
        xr = self.dec(h)
        return xr

    def forward(self, x):
        z = self.encode(x)
        xr = self.decode(z)
        return xr, z

# -----------------------------
# Tanítás
# -----------------------------
def train_autoencoder(model, train_loader, device, epochs=6, lr=1e-3, progress_place=None):
    model.train()
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    history = []

    for ep in range(1, epochs+1):
        running = 0.0
        n = 0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            xr, _ = model(x)
            loss = criterion(xr, x)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        avg = running / max(1, n)
        history.append(avg)
        if progress_place is not None:
            progress_place.write(f"📊 Epoch {ep}/{epochs} — Train MSE: **{avg:.5f}**")
    return history

@torch.no_grad()
def evaluate_autoencoder(model, test_loader, device, max_batches=10):
    model.eval()
    criterion = nn.MSELoss(reduction='sum')
    total_mse = 0.0
    total_n = 0
    first_batch = None
    first_labels = None
    all_Z = []
    all_y = []

    for b_idx, (x, y) in enumerate(test_loader):
        x = x.to(device)
        xr, z = model(x)
        total_mse += criterion(xr, x).item()
        total_n += x.numel()
        if first_batch is None:
            first_batch = (x.cpu().clone(), xr.cpu().clone())
            first_labels = y.clone()
        all_Z.append(to_cpu_numpy(z))
        all_y.append(to_cpu_numpy(y))
        if b_idx+1 >= max_batches:
            break

    mse = total_mse / total_n
    psnr = psnr_from_mse(mse)
    Z = np.concatenate(all_Z, axis=0) if all_Z else np.zeros((0, model.latent_dim))
    y = np.concatenate(all_y, axis=0) if all_y else np.zeros((0,))
    return mse, psnr, first_batch, first_labels, Z, y

@torch.no_grad()
def latent_traversal(model, x, span=2.0, steps=9, device="cpu"):
    """Egydarab input képet (x[0]) kódolunk, majd a z tér mindhárom dimenziójában
       egy-egy vonalat bejárunk. Visszaadunk egy [3*steps] rekonstr. rácsot."""
    model.eval()
    x = x[:1].to(device)  # első minta
    _, z0 = model(x)
    z0 = z0[0].cpu().numpy()
    lat_dim = z0.shape[0]
    if lat_dim < 1:
        return None

    grid_imgs = []
    axes = min(3, lat_dim)  # max 3 tengely a bemutatóhoz
    for d in range(axes):
        vals = np.linspace(-span, span, steps)
        for v in vals:
            z = z0.copy()
            z[d] = z0[d] + v
            zt = torch.from_numpy(z).float().unsqueeze(0).to(device)
            xr = model.decode(zt)
            grid_imgs.append(xr.squeeze(0).cpu())
    # [axes*steps, 1, 28, 28]
    return torch.stack(grid_imgs, dim=0)

# -----------------------------
# Streamlit modul
# -----------------------------
def app():
    st.set_page_config(layout="wide")
    st.title("🧠 Autoencoder – 3D latens tér, gyors demó és mélyebb diagnosztika")

    # Gyors infó a teljes MNIST méretről (cache-elt)
    n_train_full, n_test_full = load_mnist_datasets()

    st.sidebar.header("⚙️ Beállítások")
    arch = st.sidebar.selectbox("Architektúra", ["Konvolúciós AE", "MLP AE"])
    latent_dim = st.sidebar.slider("Latens dimenzió (vizualizációhoz 3 javasolt)", 2, 8, 3)
    lr = st.sidebar.select_slider("Tanulási ráta", options=[5e-4, 1e-3, 2e-3], value=1e-3)
    batch_size = st.sidebar.slider("Batch méret", 64, 512, 128, 64)
    epochs = st.sidebar.slider("Epochok", 1, 20, 6)

    st.sidebar.subheader("⏱️ Gyors demó / Adat-budget")
    quick_demo = st.sidebar.checkbox("Gyors demó mód (ajánlott elsőre)", value=True)
    if quick_demo:
        n_train = st.sidebar.number_input("Train budget (képek)", 500, n_train_full, 2000, step=500)
        n_test  = st.sidebar.number_input("Test budget (képek)", 500, n_test_full, 2000, step=500)
        epochs  = min(epochs, 8)  # demóban ne legyen túl hosszú
    else:
        n_train = st.sidebar.number_input("Train budget (képek)", 1000, n_train_full, 20000, step=1000)
        n_test  = st.sidebar.number_input("Test budget (képek)", 1000, n_test_full, 5000, step=500)

    seed = st.sidebar.number_input("Seed", 0, 9999, 42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Modell létrehozása
    torch.manual_seed(seed)
    if arch == "Konvolúciós AE":
        model = Conv_Autoencoder(latent_dim=latent_dim).to(device)
    else:
        model = MLP_Autoencoder(latent_dim=latent_dim).to(device)

    # Adatbetöltők
    train_loader = subset_loader(train=True,  n_items=int(n_train), batch_size=int(batch_size), shuffle=True, seed=seed)
    test_loader  = subset_loader(train=False, n_items=int(n_test),  batch_size=int(batch_size), shuffle=False, seed=seed)

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        st.markdown("**Adat-budget**")
        st.write(f"Train: **{n_train}** / {n_train_full} • Test: **{n_test}** / {n_test_full}")
        st.write(f"Eszköz: **{device.upper()}** • Arch: **{arch}** • Latens dim: **{latent_dim}**")

    start = st.button("🚀 Tanítás és kiértékelés")

    if start:
        log_area = st.empty()
        # Tanítás
        history = train_autoencoder(model, train_loader, device, epochs=epochs, lr=lr, progress_place=log_area)

        # Loss görbe
        st.subheader("📉 Rekonstrukciós hiba (MSE) alakulása")
        fig_loss, ax = plt.subplots()
        ax.plot(history, marker='o')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train MSE")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig_loss)

        # Kiértékelés + első batch rekonstrukció
        mse, psnr, first_batch, first_labels, Z, y = evaluate_autoencoder(model, test_loader, device, max_batches=10)
        st.success(f"✅ Test MSE: **{mse:.6f}**  |  PSNR: **{psnr:.2f} dB**  (0–1 skálán)")

        # Eredeti vs Recon rács
        st.subheader("🖼️ Eredeti vs. Rekonstrukció")
        if first_batch is not None:
            x0, xr0 = first_batch
            # 10-10 kép rácsban
            n_show = min(10, x0.shape[0])
            grid_orig = make_grid(x0[:n_show], nrow=n_show, normalize=True, pad_value=1.0)
            grid_reco = make_grid(xr0[:n_show], nrow=n_show, normalize=True, pad_value=1.0)

            fig, axes = plt.subplots(2, 1, figsize=(n_show*1.0, 3), dpi=120)
            axes[0].imshow(grid_orig.permute(1,2,0), cmap='gray')
            axes[0].set_title("Eredeti")
            axes[0].axis('off')
            axes[1].imshow(grid_reco.permute(1,2,0), cmap='gray')
            axes[1].set_title("Rekonstrukció")
            axes[1].axis('off')
            st.pyplot(fig)

        # Latens tér 3D
        st.subheader("🌌 3D latens tér")
        if Z.shape[0] > 0:
            if Z.shape[1] == 3:
                Z3 = Z
            else:
                st.caption("Latens dim != 3 → PCA-val vetítve 3D-be a vizualizációhoz.")
                Z3 = pca_to_3d(Z)
            df = pd.DataFrame(Z3, columns=["z1","z2","z3"])
            df["label"] = y.astype(int)
            fig3d = px.scatter_3d(df, x="z1", y="z2", z="z3",
                                  color=df["label"].astype(str),
                                  opacity=0.85,
                                  title="Latens reprezentáció (3D)")
            st.plotly_chart(fig3d, use_container_width=True)

            # Export latensek
            st.download_button("⬇️ Latens vektorok (CSV)",
                               df.to_csv(index=False).encode("utf-8"),
                               file_name="latent_vectors.csv",
                               mime="text/csv")

        # Latens traversálás (csak ha van első batch)
        st.subheader("🧭 Latens traversálás (első minta mentén max. 3 tengely)")
        span = st.slider("Eltérés (±)", 0.5, 4.0, 2.0, 0.5)
        steps = st.slider("Lépések tengelyenként", 5, 13, 9, 2)
        if first_batch is not None:
            with torch.no_grad():
                grid_lat = latent_traversal(model, first_batch[0], span=span, steps=steps, device=device)
            if grid_lat is not None:
                # rács: axes sorok egymás alatt
                grid = make_grid(grid_lat, nrow=steps, normalize=True, pad_value=1.0)
                plt.figure(figsize=(steps*0.7, 6))
                plt.imshow(grid.permute(1,2,0), cmap='gray')
                plt.axis('off')
                plt.title("Latens tengelyek menti változások")
                st.pyplot(plt.gcf())

        # Rövid tudományos összefoglaló
        st.markdown("### 📚 Tudományos háttér (rövid)")
        st.markdown(
            "- Rekonstrukciós cél: $\\min \\|x-\\hat{x}\\|^2$; **PSNR** a vizuális hűség durva mértéke.\n"
            "- **Konvolúciós AE** jobb lokális mintázat-megőrzést ad az MNIST-hez, mint a tiszta MLP.\n"
            "- A latens tér klasztereződése jelzi, hogy a reprezentáció **diszkriminatív** az osztályokra.\n"
            "- **Latens traversálás**: lokális manifoldszerkezetet szemléltet a $z$ térben."
        )

# ReflectAI-kompatibilitás
app = app
