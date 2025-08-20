import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Subset

# ==============================
# Hasznos segédfüggvények
# ==============================
@st.cache_data
def load_mnist_sizes():
    tfm = transforms.ToTensor()
    tr = datasets.MNIST('./data', train=True, download=True, transform=tfm)
    te = datasets.MNIST('./data', train=False, download=True, transform=tfm)
    return len(tr), len(te)

def subset_loader(train=True, n_items=3000, batch_size=128, shuffle=True, seed=42):
    tfm = transforms.ToTensor()
    ds = datasets.MNIST('./data', train=train, download=True, transform=tfm)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(int(n_items), len(ds)), replace=False)
    sub = Subset(ds, idx)
    return DataLoader(sub, batch_size=int(batch_size), shuffle=shuffle, num_workers=0, pin_memory=False)

def psnr_from_mse(mse):
    # képek 0..1 skálán → MAX_I=1
    if mse <= 0: return float('inf')
    return 10.0 * np.log10(1.0 / mse)

def pca_to_3d(Z):
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[0] == 0:
        return np.zeros((0,3))
    Zc = Z - Z.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    if Vt.shape[0] < 3:
        pad = np.zeros((3 - Vt.shape[0], Vt.shape[1]), dtype=Vt.dtype)
        Vt = np.vstack([Vt, pad])
    return Zc @ Vt[:3].T

def to_cpu(x):
    return x.detach().cpu()

# ==============================
# Modellek
# ==============================
# --- AE: MLP ---
class MLP_AE(nn.Module):
    def __init__(self, zdim=3):
        super().__init__()
        self.zdim = zdim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128),   nn.ReLU(inplace=True),
            nn.Linear(128, zdim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(zdim, 128),  nn.ReLU(inplace=True),
            nn.Linear(128, 256),   nn.ReLU(inplace=True),
            nn.Linear(256, 28*28), nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        xflat = self.decoder(z)
        return xflat.view(-1,1,28,28)

    def forward(self, x):
        z = self.encode(x)
        xr = self.decode(z)
        return xr, z

# --- AE: Convolutional ---
class Conv_AE(nn.Module):
    def __init__(self, zdim=3):
        super().__init__()
        self.zdim = zdim
        self.enc = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), nn.ReLU(inplace=True),  # 28->14
            nn.Conv2d(32,64,4,2,1), nn.ReLU(inplace=True), # 14->7
        )
        self.enc_lin = nn.Linear(64*7*7, zdim)
        self.dec_lin = nn.Linear(zdim, 64*7*7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(inplace=True), # 7->14
            nn.ConvTranspose2d(32,1,4,2,1), nn.Sigmoid()            # 14->28
        )

    def encode(self, x):
        h = self.enc(x)
        z = self.enc_lin(h.view(x.size(0), -1))
        return z

    def decode(self, z):
        h = self.dec_lin(z).view(-1,64,7,7)
        return self.dec(h)

    def forward(self, x):
        z = self.encode(x)
        xr = self.decode(z)
        return xr, z

# --- VAE: MLP ---
class MLP_VAE(nn.Module):
    def __init__(self, zdim=3):
        super().__init__()
        self.zdim = zdim
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128),   nn.ReLU(inplace=True)
        )
        self.mu    = nn.Linear(128, zdim)
        self.logv  = nn.Linear(128, zdim)
        self.dec   = nn.Sequential(
            nn.Linear(zdim, 128),  nn.ReLU(inplace=True),
            nn.Linear(128, 256),   nn.ReLU(inplace=True),
            nn.Linear(256, 28*28), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logv(h)

    def reparam(self, mu, logv):
        std = torch.exp(0.5*logv)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        xflat = self.dec(z)
        return xflat.view(-1,1,28,28)

    def forward(self, x):
        mu, logv = self.encode(x)
        z = self.reparam(mu, logv)
        xr = self.decode(z)
        return xr, z, mu, logv

# --- VAE: Convolutional ---
class Conv_VAE(nn.Module):
    def __init__(self, zdim=3):
        super().__init__()
        self.zdim = zdim
        self.enc = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), nn.ReLU(inplace=True),  # 28->14
            nn.Conv2d(32,64,4,2,1), nn.ReLU(inplace=True), # 14->7
        )
        self.enc_lin = nn.Linear(64*7*7, 128)
        self.mu    = nn.Linear(128, zdim)
        self.logv  = nn.Linear(128, zdim)
        self.dec_lin = nn.Linear(zdim, 64*7*7)
        self.dec  = nn.Sequential(
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,1,4,2,1), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x).view(x.size(0), -1)
        h = F.relu(self.enc_lin(h), inplace=True)
        return self.mu(h), self.logv(h)

    def reparam(self, mu, logv):
        std = torch.exp(0.5*logv)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_lin(z).view(-1,64,7,7)
        return self.dec(h)

    def forward(self, x):
        mu, logv = self.encode(x)
        z = self.reparam(mu, logv)
        xr = self.decode(z)
        return xr, z, mu, logv

# ==============================
# Tanítás és értékelés
# ==============================
def train_epoch_AE(model, loader, opt, device, loss_type="MSE"):
    model.train()
    total = 0.0
    n_pix = 0
    for x, _ in loader:
        x = x.to(device)
        opt.zero_grad()
        xr, _ = model(x)
        if loss_type == "BCE":
            loss = F.binary_cross_entropy(xr, x, reduction='sum') / x.size(0)
        else:
            loss = F.mse_loss(xr, x, reduction='mean')
        loss.backward()
        opt.step()
        if loss_type == "BCE":
            total += loss.item() * x.size(0)
            n_pix += x.size(0)
        else:
            total += loss.item() * x.size(0)
            n_pix += x.size(0)
    return total / max(1, n_pix)

def train_epoch_VAE(model, loader, opt, device, beta=1.0, loss_type="BCE"):
    model.train()
    recon_total = 0.0
    kl_total = 0.0
    n_samp = 0
    for x, _ in loader:
        x = x.to(device)
        opt.zero_grad()
        xr, z, mu, logv = model(x)
        if loss_type == "BCE":
            recon = F.binary_cross_entropy(xr, x, reduction='sum') / x.size(0)
        else:
            recon = F.mse_loss(xr, x, reduction='sum') / x.size(0)
        # KL = -0.5 * sum(1 + logσ^2 - μ^2 - σ^2), átlag minta/fő
        kl = -0.5 * torch.sum(1 + logv - mu.pow(2) - logv.exp()) / x.size(0)
        loss = recon + beta * kl
        loss.backward()
        opt.step()
        recon_total += recon.item() * x.size(0)
        kl_total    += kl.item()    * x.size(0)
        n_samp      += x.size(0)
    return (recon_total/max(1,n_samp)), (kl_total/max(1,n_samp)), (recon_total+beta*kl_total)/max(1,n_samp)

@torch.no_grad()
def evaluate_AE(model, loader, device, max_batches=8):
    model.eval()
    sum_sq = 0.0
    count  = 0
    first_batch = None
    first_labels = None
    Z_all, y_all = [], []
    for bi, (x,y) in enumerate(loader):
        x = x.to(device)
        xr, z = model(x)
        sum_sq += F.mse_loss(xr, x, reduction='sum').item()
        count  += x.numel()
        if first_batch is None:
            first_batch = (to_cpu(x), to_cpu(xr))
            first_labels = y.clone()
        Z_all.append(to_cpu(z).numpy())
        y_all.append(y.numpy())
        if bi+1 >= max_batches: break
    mse = sum_sq / max(1,count)
    psnr = psnr_from_mse(mse)
    Z = np.concatenate(Z_all, axis=0) if Z_all else np.zeros((0, model.zdim))
    y = np.concatenate(y_all, axis=0) if y_all else np.zeros((0,))
    return mse, psnr, first_batch, first_labels, Z, y

@torch.no_grad()
def evaluate_VAE(model, loader, device, max_batches=8, use_mu=True):
    model.eval()
    sum_sq = 0.0
    count  = 0
    first_batch = None
    first_labels = None
    Z_all, y_all = [], []
    for bi, (x,y) in enumerate(loader):
        x = x.to(device)
        xr, z, mu, logv = model(x)
        sum_sq += F.mse_loss(xr, x, reduction='sum').item()
        count  += x.numel()
        if first_batch is None:
            first_batch = (to_cpu(x), to_cpu(xr))
            first_labels = y.clone()
        embed = mu if use_mu else z
        Z_all.append(to_cpu(embed).numpy())
        y_all.append(y.numpy())
        if bi+1 >= max_batches: break
    mse = sum_sq / max(1,count)
    psnr = psnr_from_mse(mse)
    Z = np.concatenate(Z_all, axis=0) if Z_all else np.zeros((0, model.zdim))
    y = np.concatenate(y_all, axis=0) if y_all else np.zeros((0,))
    return mse, psnr, first_batch, first_labels, Z, y

@torch.no_grad()
def latent_traversal(model, x, is_vae=False, span=2.0, steps=9, device="cpu", use_mu=True):
    """Első minta körül 3 tengely menti bejárás."""
    if x is None or x.size(0) == 0: return None
    x = x[:1].to(device)
    if is_vae:
        _, _, mu, logv = model(x)
        center = mu if use_mu else model.reparam(mu, logv)
        z0 = center[0].cpu().numpy()
        dec = model.decode
        zdim = model.zdim
    else:
        _, z = model(x)
        z0 = z[0].cpu().numpy()
        dec = model.decode
        zdim = z0.shape[0]
    axes = min(3, zdim)
    vals = np.linspace(-span, span, steps)
    imgs = []
    for d in range(axes):
        for v in vals:
            z = z0.copy()
            z[d] = z0[d] + v
            zt = torch.from_numpy(z).float().unsqueeze(0).to(device)
            xr = dec(zt)
            imgs.append(xr.squeeze(0).cpu())
    if not imgs: return None
    return torch.stack(imgs, dim=0)  # [axes*steps,1,28,28]

@torch.no_grad()
def sample_prior_vae(model, n=16, device="cpu"):
    z = torch.randn(n, model.zdim, device=device)
    xr = model.decode(z)
    return xr.cpu()

# ==============================
# Streamlit modul (AE + VAE)
# ==============================
def app():
    st.set_page_config(layout="wide")
    st.title("🧠 Autoencoder + Variational Autoencoder – 3D latens tér, diagnosztika, export")

    # --- Bevezető + képletek (nem expanderben, egységesen a modul tetején) ---
    st.markdown("""
Az **autoencoder (AE)** determinisztikus latens reprezentációt tanul:  
**Encoder** \(x \\to z\), **Decoder** \(z \\to \\hat{x}\), a cél a **rekonstrukciós hiba minimalizálása**.

A **variációs autoencoder (VAE)** ugyanerre az alapra épül, de a latens térre **valószínűségi modellt** tanul,  
és az **ELBO**-t maximalizálja (rekonstrukció + **KL-divergencia** a normál priorhoz).
    """)

    st.latex(r"\textbf{AE:}\;\; \min_\theta \; \mathbb{E}\big[\|x-\hat{x}\|^2\big]\quad\text{vagy}\quad \min \; \mathrm{BCE}(x,\hat{x})")
    st.latex(r"\textbf{VAE:}\;\; \max \mathrm{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \mathrm{KL}\big(q_\phi(z|x)\,\|\,p(z)\big)")
    st.latex(r"z = \mu(x) + \sigma(x)\odot \epsilon,\;\; \epsilon\sim\mathcal{N}(0,I) \quad\text{(reparametrizáció)}")

    # --- Oldalsáv: beállítások ---
    n_train_full, n_test_full = load_mnist_sizes()
    st.sidebar.header("⚙️ Beállítások")

    family = st.sidebar.selectbox("Modell", ["Autoencoder (AE)", "Variational Autoencoder (VAE)"])
    arch = st.sidebar.selectbox("Architektúra", ["Konvolúciós", "MLP"])
    zdim = st.sidebar.slider("Latens dimenzió", 2, 8, 3)
    loss_type = st.sidebar.selectbox("Rekonstrukciós veszteség", ["MSE", "BCE"])
    lr = st.sidebar.select_slider("Tanulási ráta", options=[5e-4, 1e-3, 2e-3], value=1e-3)
    batch = st.sidebar.slider("Batch méret", 64, 512, 128, 64)
    epochs = st.sidebar.slider("Epochok", 1, 30, 8)

    if family.startswith("Variational"):
        beta = st.sidebar.slider("β (KL-súly)", 0.1, 4.0, 1.0, 0.1)
    else:
        beta = None

    st.sidebar.subheader("⏱️ Gyors demó / adat-budget")
    quick = st.sidebar.checkbox("Gyors demó mód", value=True)
    if quick:
        n_train = st.sidebar.number_input("Train képek", 500, n_train_full, 3000, step=500)
        n_test  = st.sidebar.number_input("Test képek",  500, n_test_full, 2000, step=500)
        epochs  = min(epochs, 12)
    else:
        n_train = st.sidebar.number_input("Train képek", 1000, n_train_full, 20000, step=1000)
        n_test  = st.sidebar.number_input("Test képek",  1000, n_test_full, 5000,  step=500)

    seed = st.sidebar.number_input("Seed", 0, 9999, 42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Modell létrehozás ---
    torch.manual_seed(seed)
    if family.startswith("Autoencoder"):
        model = (Conv_AE(zdim) if arch=="Konvolúciós" else MLP_AE(zdim)).to(device)
    else:
        model = (Conv_VAE(zdim) if arch=="Konvolúciós" else MLP_VAE(zdim)).to(device)

    # --- Dataloader-ek (budget) ---
    train_loader = subset_loader(True,  n_train, batch, True, seed)
    test_loader  = subset_loader(False, n_test,  batch, False, seed)

    # --- Meta infó ---
    c1, c2, c3 = st.columns([1.5,1,1])
    with c1:
        st.markdown("**Adat-budget**")
        st.write(f"Train: **{n_train}** / {n_train_full} • Test: **{n_test}** / {n_test_full}")
    with c2:
        st.write(f"Eszköz: **{device.upper()}**")
        st.write(f"Arch: **{arch}**, Latens: **{zdim}**")
    with c3:
        if family.startswith("Variational"):
            st.write(f"β (KL): **{beta:.2f}**")
        st.write(f"Loss: **{loss_type}**, LR: **{lr}**")

    # --- Tanítás ---
    start = st.button("🚀 Tanítás és kiértékelés")
    if start:
        prog = st.empty()
        opt = optim.Adam(model.parameters(), lr=lr)
        hist = {"epoch":[], "train_mse_or_bce":[], "kl":[], "total":[]}

        for ep in range(1, epochs+1):
            if family.startswith("Autoencoder"):
                train_metric = train_epoch_AE(model, train_loader, opt, device, loss_type=loss_type)
                hist["epoch"].append(ep)
                hist["train_mse_or_bce"].append(train_metric)
                hist["kl"].append(0.0)
                hist["total"].append(train_metric)
                prog.write(f"📊 Epoch {ep}/{epochs} — Train {loss_type}: **{train_metric:.6f}**")
            else:
                recon_m, kl_m, tot_m = train_epoch_VAE(model, train_loader, opt, device, beta=beta, loss_type=loss_type)
                hist["epoch"].append(ep)
                hist["train_mse_or_bce"].append(recon_m)
                hist["kl"].append(kl_m)
                hist["total"].append(tot_m)
                prog.write(f"📊 Epoch {ep}/{epochs} — Recon({loss_type}): **{recon_m:.6f}** | KL: **{kl_m:.6f}** | Total: **{tot_m:.6f}**")

        # --- Loss-görbék ---
        st.subheader("📉 Tanulási görbe")
        figL, axL = plt.subplots()
        axL.plot(hist["epoch"], hist["train_mse_or_bce"], label=f"Train {loss_type}")
        if family.startswith("Variational"):
            axL.plot(hist["epoch"], hist["kl"], label="KL")
            axL.plot(hist["epoch"], hist["total"], label="Total")
        axL.grid(alpha=0.3); axL.set_xlabel("Epoch"); axL.legend()
        st.pyplot(figL)

        # --- Értékelés + rekonstrukciók ---
        if family.startswith("Autoencoder"):
            mse, psnr, first_batch, first_labels, Z, y = evaluate_AE(model, test_loader, device)
        else:
            mse, psnr, first_batch, first_labels, Z, y = evaluate_VAE(model, test_loader, device, use_mu=True)

        st.success(f"✅ Test MSE: **{mse:.6f}** | PSNR: **{psnr:.2f} dB** (0–1 skála)")

        st.subheader("🖼️ Eredeti vs. rekonstrukció")
        if first_batch is not None:
            x0, xr0 = first_batch
            n_show = min(12, x0.shape[0])
            grid_o  = make_grid(x0[:n_show],  nrow=n_show, normalize=True, pad_value=1.0)
            grid_r  = make_grid(xr0[:n_show], nrow=n_show, normalize=True, pad_value=1.0)
            figG, axes = plt.subplots(2,1, figsize=(n_show*0.9, 3), dpi=120)
            axes[0].imshow(grid_o.permute(1,2,0), cmap='gray'); axes[0].set_title("Eredeti");      axes[0].axis('off')
            axes[1].imshow(grid_r.permute(1,2,0), cmap='gray'); axes[1].set_title("Rekonstrukció"); axes[1].axis('off')
            st.pyplot(figG)

        # --- Generatív minták (csak VAE) ---
        if family.startswith("Variational"):
            st.subheader("🎲 VAE – generált minták a priorból (z~N(0,I))")
            samples = sample_prior_vae(model, n=16, device=device)
            grid_s = make_grid(samples, nrow=8, normalize=True, pad_value=1.0)
            plt.figure(figsize=(8,3)); plt.imshow(grid_s.permute(1,2,0), cmap='gray'); plt.axis('off')
            st.pyplot(plt.gcf())

        # --- 3D latens tér (ha nem 3D, PCA vetítés) ---
        st.subheader("🌌 3D latens tér")
        if Z.shape[0] > 0:
            Z3 = Z if Z.shape[1]==3 else pca_to_3d(Z)
            df = pd.DataFrame(Z3, columns=["z1","z2","z3"])
            df["label"] = y.astype(int)
            fig3d = px.scatter_3d(df, x="z1", y="z2", z="z3", color=df["label"].astype(str),
                                  opacity=0.85, title="Latens reprezentáció (3D)")
            st.plotly_chart(fig3d, use_container_width=True)

            # --- CSV exportok ---
            st.subheader("⬇️ Exportok")
            st.download_button("Latens vektorok (CSV)", df.to_csv(index=False).encode("utf-8"),
                               file_name="latent_vectors.csv", mime="text/csv")
            hist_df = pd.DataFrame(hist)
            st.download_button("Tanulási görbe (CSV)", hist_df.to_csv(index=False).encode("utf-8"),
                               file_name="training_history.csv", mime="text/csv")

        # --- Latens traversálás (első minta körül, 3 tengely) ---
        st.subheader("🧭 Latens traversálás (első mintán)")
        span = st.slider("Eltérés ±", 0.5, 4.0, 2.0, 0.5)
        steps = st.slider("Lépések tengelyenként", 5, 13, 9, 2)
        grid_lat = latent_traversal(model, first_batch[0] if first_batch else None,
                                    is_vae=family.startswith("Variational"),
                                    span=span, steps=steps, device=device, use_mu=True)
        if grid_lat is not None:
            g = make_grid(grid_lat, nrow=steps, normalize=True, pad_value=1.0)
            plt.figure(figsize=(steps*0.7, 6))
            plt.imshow(g.permute(1,2,0), cmap='gray'); plt.axis('off')
            plt.title("Latens tengelyek menti elmozdulás")
            st.pyplot(plt.gcf())

        # --- Rövid tudományos összegzés ---
        st.markdown("### 📚 Tudományos összegzés")
        if family.startswith("Autoencoder"):
            st.markdown(
                "- **AE cél**: $\\min \\|x-\\hat{x}\\|^2$ vagy BCE — determinisztikus $z$.\n"
                "- A latens klasztereződés azt jelzi, hogy a reprezentáció **diszkriminatív** az osztályokra.\n"
                "- **PSNR** a rekonstrukció vizuális hűségének durva indikátora.\n"
                "- **Latens traversálás**: lokális manifoldszerkezet szemléltetése."
            )
        else:
            st.markdown(
                "- **VAE cél**: ELBO maximalizálása (rekonstrukció + **KL**), $\\beta$ a diszperzió/sűrűség kompromisszumát szabályozza.\n"
                "- **Reparametrizáció** stabil backprop a sztochasztikus $z$-n.\n"
                "- **Prior mintavétel**: valódi generatív képesség, nem csak másolás."
            )

# ReflectAI kompatibilitás
app = app
