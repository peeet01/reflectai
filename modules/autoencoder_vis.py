import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

# ----- Adatok betöltése -----
transform = transforms.ToTensor()
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=128, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transform),
    batch_size=1000, shuffle=False
)

# ----- Autoencoder modell -----
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# ----- App -----
def app():
    st.set_page_config(layout="wide")
    st.title("🧠 Autoencoder Vizualizáció – 3D Latens tér")

    with st.expander("📘 Mi történik ebben a modulban?", expanded=True):
        st.markdown("""
        Egy **autoencoder** célja, hogy a bemeneti adatot **egy alacsony dimenziós térbe leképezze**, majd onnan visszaállítsa azt.  
        A középső 3-dimenziós kódolt reprezentáció segítségével **vizualizálni tudjuk az adatokat**.

        #### 🧠 Alapstruktúra:
        - **Encoder**: $x \\rightarrow z$
        - **Decoder**: $z \\rightarrow \\hat{x}$

        #### 💡 Célfüggvény:
        A modell célja, hogy minimalizálja a rekonstrukciós hibát:

        $$
        \\mathcal{L} = \\frac{1}{N} \\sum_i \\| x_i - \\hat{x}_i \\|^2
        $$

        Ahol:
        - $x_i$: eredeti kép
        - $\\hat{x}_i$: rekonstruált kép
        - $z$: latens reprezentáció

        A 3D latens tér lehetővé teszi, hogy a **képek kategóriái külön klaszterekként** jelenjenek meg.

        """)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if st.button("🚀 Tanítás indítása (12 epoch)"):
        loss_history = []

        for epoch in range(12):
            model.train()
            total_loss = 0
            for data, _ in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, data.view(-1, 28*28))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            loss_history.append(avg_loss)
            st.write(f"📊 Epoch {epoch+1}/12 | Loss: {avg_loss:.4f}")

        # Loss görbe
        st.subheader("📉 Rekonstrukciós hiba alakulása")
        fig, ax = plt.subplots()
        ax.plot(loss_history)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Rekonstrukciós hiba")
        st.pyplot(fig)

        # Képek rekonstruálása
        st.subheader("🖼️ Rekonstruált képek")
        model.eval()
        with torch.no_grad():
            test_imgs, test_labels = next(iter(test_loader))
            test_imgs = test_imgs.to(device)
            encoded = model.encoder(test_imgs)
            decoded = model.decoder(encoded).view(-1, 1, 28, 28).cpu()

        grid = make_grid(decoded[:10], nrow=5, normalize=True)
        st.image(grid.permute(1, 2, 0).numpy(), clamp=True)

        # 3D Plotly scatter
        st.subheader("🌌 3D Latens tér (Plotly)")
        z = encoded.cpu().numpy()
        labels = test_labels.numpy()
        df = pd.DataFrame(z, columns=["x", "y", "z"])
        df["label"] = labels
        fig = px.scatter_3d(df, x="x", y="y", z="z", color=df["label"].astype(str),
                            title="MNIST latens reprezentáció", width=800, height=600)
        st.plotly_chart(fig)

        # Tudományos értékelés
        st.subheader("🧪 Tudományos értékelés")
        st.markdown("""
        Az autoencoder sikeresen leképezte az MNIST számjegyeket egy **3D latens térbe**, ahol az egyes osztályok láthatóan klasztereződnek.

        Ez azt mutatja, hogy a modell képes **kompakt reprezentációkat** kialakítani, amelyek **megőrzik a kategória-információt**.

        #### 🔬 Következtetés:
        - A kódolt tér **szerkezetet tükröz**, nem véletlenszerű
        - Alkalmazható **dimenziócsökkentésre**, **adatvizualizációra** és **előfeldolgozásra**
        """)

# ReflectAI-kompatibilis
app = app
