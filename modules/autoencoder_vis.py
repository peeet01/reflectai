# modules/vae_module.py
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image

# =============== Hasznos segédfüggvények ===============

@st.cache_data
def mnist_sizes():
    tr = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    te = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    return len(tr), len(te)

def subset_loader(train: bool, n_items: int, batch_size: int, seed: int = 42, shuffle: bool = True):
    ds = datasets.MNIST("./data", train=train, download=True, transform=transforms.ToTensor())
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n_items, len(ds)), replace=False)
    sub = Subset(ds, idx)
    return DataLoader(sub, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)

def psnr_from_mse(mse):
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)

def pca_to_3d(Z: np.ndarray):
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        return np.zeros((0, 3))
    Zc = Z - Z.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    if Vt.shape[0] < 3:
        pad = np.zeros((3 - Vt.shape[0], Vt.shape[1]), dtype=Vt.dtype)
        Vt = np.vstack([Vt, pad])
    return Zc @ Vt[:3].T

# =============== Konvolúciós VAE ===============

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=3):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder: 28x28x1 -> 14x14x32 -> 7x7x64 -> FC -> (mu, logvar)
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 28->14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), # 14->7
            nn.ReLU(inplace=True),
        )
        self.enc_lin = nn.Linear(64*7*7, 256)
        self.mu_head = nn.Linear(256, latent_dim)
        self.logvar_head = nn.Linear(256, latent_dim)

        # Decoder: z -> FC -> 7x7x64 -> 14x14x32 -> 28x28x1 (sigmoid)
        self.dec_lin = nn.Linear(latent_dim, 64*7*7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # 7->14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),  # 14->28
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(x.size(0), -1)
        h = F.relu(self.enc_lin(h), inplace=True)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_lin(z).view(-1, 64, 7, 7)
        xr = self.dec(h)
        return xr

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xr = self.decode(z)
        return xr, mu, logvar, z

# =============== Tanítás/értékelés ===============

def vae_loss(x, xr, mu, logvar, beta=1.0, recon="bce"):
    if recon == "mse":
        recon_loss = F.mse_loss(xr, x, reduction="mean")
    else:
        # BCE az [0,1] képekre; átlag a batch felett
        recon_loss = F.binary_cross_entropy(xr, x, reduction="mean")
    # KL(q||p) az N(mu, sigma^2) és N(0, I) között, képenként átlag
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    elbo = recon_loss + beta * kl
    return elbo, recon_loss, kl

def train_vae(model, loader, device, epochs=6, lr=1e-3, beta=1.0, recon="bce", progress=None):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"elbo": [], "recon": [], "kl": []}
    for ep in range(1, epochs+1):
        elbo_sum = recon_sum = kl_sum = 0.0
        n_batches = 0
        for x, _ in loader:
            x = x.to(device)
            opt.zero_grad()
            xr, mu, logvar, z = model(x)
            loss, r, k = vae_loss(x, xr, mu, logvar, beta=beta, recon=recon)
            loss.backward()
            opt.step()
            elbo_sum += loss.item()
            recon_sum += r.item()
            kl_sum += k.item()
            n_batches += 1
        elbo_avg = elbo_sum / n_batches
        recon_avg = recon_sum / n_batches
        kl_avg = kl_sum / n_batches
        hist["elbo"].append(elbo_avg)
        hist["recon"].append(recon_avg)
        hist["kl"].append(kl_avg)
        if progress is not None:
            progress.write(f"📊 Epoch {ep}/{epochs} — ELBO: **{elbo_avg:.5f}**, Recon: {recon_avg:.5f}, KL: {kl_avg:.5f}")
    return hist

@torch.no_grad()
def evaluate_vae(model, loader, device, max_batches=10, recon_metric="mse"):
    model.eval()
    # MSE a PSNR-hez
    mse_sum = 0.0
    n_pix = 0
    first_batch = None
    first_labels = None
    all_mu = []
    all_z = []
    all_y = []
    b_count = 0
    for x, y in loader:
        x = x.to(device)
        xr, mu, logvar, z = model(x)
        mse_sum += F.mse_loss(xr, x, reduction="sum").item()
        n_pix += x.numel()
        if first_batch is None:
            first_batch = (x.cpu().clone(), xr.cpu().clone())
            first_labels = y.clone()
        all_mu.append(mu.cpu().numpy())
        all_z.append(z.cpu().numpy())
        all_y.append(y.cpu().numpy())
        b_count += 1
        if b_count >= max_batches:
            break
    mse = mse_sum / n_pix if n_pix > 0 else np.nan
    psnr = psnr_from_mse(mse)
    Mu = np.concatenate(all_mu, axis=0) if all_mu else np.zeros((0, model.latent_dim))
    Z  = np.concatenate(all_z,  axis=0) if all_z  else np.zeros((0, model.latent_dim))
    yy = np.concatenate(all_y,  axis=0) if all_y  else np.zeros((0,))
    return mse, psnr, first_batch, first_labels, Mu, Z, yy

@torch.no_grad()
def sample_prior(model, n_samples=32, device="cpu"):
    z = torch.randn(n_samples, model.latent_dim, device=device)
    xr = model.decode(z)
    return xr.cpu()

@torch.no_grad()
def latent_traversal(model, x, span=2.0, steps=9, device="cpu"):
    model.eval()
    x = x[:1].to(device)
    _, mu, logvar, _ = model(x)
    z0 = mu[0].cpu().numpy()
    axes = min(3, z0.shape[0])
    imgs = []
    for d in range(axes):
        vals = np.linspace(-span, span, steps)
        for v in vals:
            z = z0.copy()
            z[d] = z0[d] + v
            zt = torch.from_numpy(z).float().unsqueeze(0).to(device)
            xr = model.decode(zt)
            imgs.append(xr.squeeze(0).cpu())
    return torch.stack(imgs, dim=0)  # [axes*steps, 1, 28, 28]

# =============== Streamlit modul ===============

def run():
    st.set_page_config(layout="wide")
    st.title("🧪 Variational Autoencoder (VAE) – Latens tér, generálás, diagnosztika")

    with st.expander("📘 Mi ez a modul? (tudományos bevezető)", expanded=True):
        st.markdown(
            "A **Variational Autoencoder (VAE)** egy generatív modell, amely a bemenetet "
            "egy **valószínűségi latens térre** kódolja. A dekóder innen mintát generál a bemeneti térbe. "
            "A VAE-t **rekonstrukciós veszteség** és a **KL-divergencia** együttesével tanítjuk, "
            "ami a latens eloszlást egy **normál priorhoz** köti."
        )
        st.latex(r"\textbf{ELBO:}\quad \mathcal{L}_\text{VAE} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta\, D_{\text{KL}}(q_\phi(z|x)\,\|\,p(z))")
        st.latex(r"\textbf{Reparametrizáció:}\quad z = \mu(x) + \sigma(x)\odot\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)")
        st.markdown(
            "- **β-VAE**: a β paraméter nagyobb súlyt ad a disentangling-nak, de rontja a rekonstrukciót.\n"
            "- **Vizualizációk**: 3D (vagy PCA-vetített) latens tér, priorból mintavételezett képek, latens-tengely bejárás."
        )

    # Paraméterek
    n_train_full, n_test_full = mnist_sizes()
    st.sidebar.header("⚙️ Beállítások")
    latent_dim = st.sidebar.slider("Latens dimenzió", 2, 16, 3)
    beta = st.sidebar.slider("β (KL-súly)", 0.1, 8.0, 1.0, 0.1)
    recon_kind = st.sidebar.selectbox("Rekonstrukciós veszteség", ["bce", "mse"])
    lr = st.sidebar.select_slider("Tanulási ráta", [5e-4, 1e-3, 2e-3], 1e-3)
    batch_size = st.sidebar.slider("Batch méret", 64, 512, 128, 64)
    epochs = st.sidebar.slider("Epochok", 1, 30, 8)
    seed = st.sidebar.number_input("Seed", 0, 9999, 42)

    st.sidebar.subheader("⏱️ Gyors demó / adat-budget")
    quick = st.sidebar.checkbox("Gyors demó mód", value=True)
    if quick:
        n_train = st.sidebar.number_input("Train képek", 500, n_train_full, 3000, step=500)
        n_test  = st.sidebar.number_input("Test képek",  500, n_test_full, 2000, step=500)
        epochs = min(epochs, 10)
    else:
        n_train = st.sidebar.number_input("Train képek", 1000, n_train_full, 20000, step=1000)
        n_test  = st.sidebar.number_input("Test képek",  1000, n_test_full, 5000,  step=500)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Adatbetöltők
    train_loader = subset_loader(True, int(n_train), int(batch_size), seed=seed, shuffle=True)
    test_loader  = subset_loader(False, int(n_test), int(batch_size), seed=seed, shuffle=False)

    # Modell
    model = ConvVAE(latent_dim=latent_dim).to(device)

    st.markdown(
        f"**Eszköz:** {device.upper()} • **Latens dim:** {latent_dim} • **β:** {beta} • "
        f"**Rekonst.:** {recon_kind.upper()} • **Train/Test:** {n_train}/{n_test}"
    )

    start = st.button("🚀 Tanítás és kiértékelés")
    if not start:
        return

    log = st.empty()
    history = train_vae(model, train_loader, device, epochs=epochs, lr=lr, beta=beta, recon=recon_kind, progress=log)

    # Loss görbék + CSV export
    st.subheader("📉 Tanulási görbék (ELBO, Recon, KL)")
    fig, ax = plt.subplots()
    ax.plot(history["elbo"], label="ELBO")
    ax.plot(history["recon"], label="Recon")
    ax.plot(history["kl"], label="KL")
    ax.set_xlabel("Epoch")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig)

    hist_df = pd.DataFrame({"epoch": np.arange(1, len(history["elbo"])+1),
                            "elbo": history["elbo"],
                            "recon": history["recon"],
                            "kl": history["kl"]})
    st.download_button("⬇️ Loss history (CSV)",
                       hist_df.to_csv(index=False).encode("utf-8"),
                       "vae_loss_history.csv", "text/csv")

    # Értékelés
    mse, psnr, first_batch, first_labels, Mu, Z, y = evaluate_vae(model, test_loader, device, max_batches=10)
    st.success(f"✅ Test PSNR: **{psnr:.2f} dB**  |  MSE: **{mse:.6f}**")

    # Eredeti vs rekonstrukció grid + letöltés
    st.subheader("🖼️ Eredeti vs. rekonstrukció")
    if first_batch is not None:
        x0, xr0 = first_batch
        n_show = min(12, x0.shape[0])
        grid_orig = make_grid(x0[:n_show], nrow=n_show, normalize=True, pad_value=1.0)
        grid_reco = make_grid(xr0[:n_show], nrow=n_show, normalize=True, pad_value=1.0)

        fig2, axes = plt.subplots(2, 1, figsize=(n_show*0.8, 3), dpi=120)
        axes[0].imshow(grid_orig.permute(1,2,0), cmap="gray"); axes[0].axis("off"); axes[0].set_title("Eredeti")
        axes[1].imshow(grid_reco.permute(1,2,0), cmap="gray"); axes[1].axis("off"); axes[1].set_title("Rekonstrukció")
        st.pyplot(fig2)

        # PNG letöltés
        buf = io.BytesIO()
        save_image(torch.cat([x0[:n_show], xr0[:n_show]], dim=0), buf, format="png", nrow=n_show, padding=2)
        st.download_button("⬇️ Grid letöltése (PNG)", data=buf.getvalue(), file_name="vae_recon_grid.png", mime="image/png")

    # Latens tér 3D (mu vagy z)
    st.subheader("🌌 3D latens tér (μ)")
    if Mu.shape[0] > 0:
        if Mu.shape[1] == 3:
            Z3 = Mu
        else:
            st.caption("Latens dim ≠ 3 → PCA-val 3D-be vetítve a vizualizációhoz.")
            Z3 = pca_to_3d(Mu)
        df_lat = pd.DataFrame(Z3, columns=["z1","z2","z3"])
        df_lat["label"] = y.astype(int)
        fig3d = px.scatter_3d(df_lat, x="z1", y="z2", z="z3", color=df_lat["label"].astype(str),
                              opacity=0.85, title="Latens reprezentáció (μ)")
        st.plotly_chart(fig3d, use_container_width=True)

        # CSV export (μ és címkék)
        out_df = pd.DataFrame(Mu, columns=[f"mu_{i}" for i in range(Mu.shape[1])])
        out_df["label"] = y.astype(int)
        st.download_button("⬇️ Latens μ vektorok (CSV)",
                           out_df.to_csv(index=False).encode("utf-8"),
                           "vae_latent_mu.csv", "text/csv")

    # Priorból mintavétel + letöltés
    st.subheader("✨ Mintavételezés a priorból p(z)=N(0,I)")
    n_gen = st.slider("Generált minták száma", 16, 64, 32, 16)
    xr_prior = sample_prior(model, n_samples=n_gen, device=device)
    grid_gen = make_grid(xr_prior, nrow=int(np.sqrt(n_gen)), normalize=True, pad_value=1.0)
    st.image(grid_gen.permute(1,2,0).numpy(), caption="Prior minták", clamp=True)
    buf2 = io.BytesIO(); save_image(xr_prior, buf2, format="png", nrow=int(np.sqrt(n_gen)))
    st.download_button("⬇️ Generált minták (PNG)", data=buf2.getvalue(), file_name="vae_prior_samples.png", mime="image/png")

    # Latens tengely bejárás (első teszt képen)
    st.subheader("🧭 Latens tengely bejárás (z-mentén)")
    span = st.slider("Eltérés (±)", 0.5, 4.0, 2.0, 0.5)
    steps = st.slider("Lépések tengelyenként", 5, 15, 9, 2)
    if first_batch is not None:
        grid_lat = latent_traversal(model, first_batch[0], span=span, steps=steps, device=device)
        grid = make_grid(grid_lat, nrow=steps, normalize=True, pad_value=1.0)
        plt.figure(figsize=(steps*0.7, 6)); plt.imshow(grid.permute(1,2,0), cmap="gray"); plt.axis("off")
        st.pyplot(plt.gcf())
        buf3 = io.BytesIO(); save_image(grid_lat, buf3, format="png", nrow=steps)
        st.download_button("⬇️ Latens bejárás (PNG)", data=buf3.getvalue(), file_name="vae_latent_traversal.png", mime="image/png")

    # Záró tudományos összefoglaló
    st.markdown("### 📚 Tudományos összefoglaló")
    st.markdown(
        "- **ELBO** minimalizálás: rekonstrukció (BCE/MSE) + β·KL — a β állítja a rekonstrukció vs. disentangling arányt.\n"
        "- **PSNR** a vizuális hűség durva, de hasznos mérőszáma 0–1 skálán normalizált képeknél.\n"
        "- A 3D latens tér klasztereződése jelzi, hogy a reprezentáció **diszkriminatív** a számjegyosztályok szerint.\n"
        "- **Prior minták** a generatív képességet mutatják; a latens-tengely bejárás a manifold lokális szerkezetét szemlélteti."
    )

# ReflectAI kompatibilitás
app = run
