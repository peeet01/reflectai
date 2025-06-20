import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D  # 3D vizualizációhoz

# 🔧 Modell definiálása
class XORNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=4):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return x

# 🌪️ Zaj hozzáadása
def add_noise(data, noise_level):
    noise = noise_level * np.random.randn(*data.shape)
    return data + noise

# 💾 Modell mentése
def save_model(model, path="xor_model.pth"):
    torch.save(model.state_dict(), path)

# 📊 Pontosság kiértékelése
def evaluate(model, inputs, targets):
    with torch.no_grad():
        predictions = model(inputs).round()
        accuracy = (predictions.eq(targets).sum().item()) / targets.size(0)
    return accuracy, predictions

# 🚀 Fő Streamlit modul
def run(hidden_size=4, learning_rate=0.1, epochs=1000, note=""):
    st.subheader("🔁 XOR predikció neurális hálóval")
    st.markdown("Ez a modul egy egyszerű neurális háló segítségével tanítja meg az XOR függvényt, zajjal, mentéssel, exporttal és 3D vizualizációval kiegészítve.")

    # 📋 Paraméterek
    noise_level = st.slider("Zaj szintje", 0.0, 0.5, 0.1, 0.01)
    export_results = st.checkbox("📤 Eredmények exportálása CSV-be")
    save_model_flag = st.checkbox("💾 Modell mentése")
    custom_input = st.checkbox("🎛️ Egyéni input kipróbálása tanítás után")
    user_note = st.text_area("📝 Megjegyzésed", value=note)

    # 🧠 GPU támogatás
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔢 XOR adatok
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    X_noisy = add_noise(X, noise_level)

    X_tensor = torch.tensor(X_noisy, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    # 📐 Modell, veszteség, optimalizáló
    model = XORNet(hidden_size=hidden_size).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ⏱️ Tanítás
    start_time = time.time()
    progress = st.progress(0)
    progress_text = st.empty()
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % max(1, epochs // 100) == 0 or epoch == epochs - 1:
            percent = int((epoch + 1) / epochs * 100)
            progress.progress((epoch + 1) / epochs)
            progress_text.text(f"Tanítás folyamatban... {percent}%")

    train_time = time.time() - start_time

    # 📉 Loss görbe
    st.markdown("### 📉 Veszteség alakulása")
    st.line_chart(losses)

    # 🌊 3D hullámszerű predikciós felület
    st.markdown("### 🌊 Predikciós felület (3D hullámként)")
    grid_size = 100
    x_vals = np.linspace(0, 1, grid_size)
    y_vals = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)

    with torch.no_grad():
        zz = model(grid_tensor).cpu().numpy().reshape(xx.shape)

    fig_wave = plt.figure(figsize=(8, 6))
    ax = fig_wave.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xx, yy, zz, cmap="coolwarm", edgecolor='k', linewidth=0.3, alpha=0.9)
    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    ax.set_zlabel("Predikció")
    ax.set_title("🌐 Hullámszerű predikciós felület")
    fig_wave.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    st.pyplot(fig_wave)

    # 📈 Eredmények
    accuracy, predictions = evaluate(model, X_tensor, y_tensor)
    st.success(f"✅ Tanítás kész! Pontosság: {accuracy * 100:.2f}%")
    st.info(f"🕒 Tanítás ideje: {train_time:.2f} másodperc")

    # 🧮 Konfúziós mátrix
    st.markdown("### 🧮 Konfúziós mátrix")
    cm = confusion_matrix(y_tensor.cpu().numpy(), predictions.cpu().numpy())
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    ax_cm.set_xlabel("Predikció")
    ax_cm.set_ylabel("Valós érték")
    st.pyplot(fig_cm)

    # 📤 CSV export
    if export_results:
        results_df = pd.DataFrame({
            "Input1": X[:, 0],
            "Input2": X[:, 1],
            "Zaj": noise_level,
            "Valós kimenet": y.flatten(),
            "Predikció": predictions.cpu().numpy().flatten()
        })
        csv_path = "xor_results.csv"
        results_df.to_csv(csv_path, index=False)
        with open(csv_path, "rb") as f:
            csv_bytes = f.read()
            st.download_button("📁 CSV letöltése", data=csv_bytes, file_name="xor_results.csv", mime="text/csv")

    # 💾 Modell mentése
    if save_model_flag:
        save_model(model)
        st.success("💾 Modell elmentve `xor_model.pth` néven.")

    # 🎛️ Egyéni input predikció
    if custom_input:
        st.markdown("### 🧪 Egyéni input kipróbálása")
        input1 = st.slider("Input 1", 0.0, 1.0, 0.0)
        input2 = st.slider("Input 2", 0.0, 1.0, 0.0)
        input_tensor = torch.tensor([[input1, input2]], dtype=torch.float32).to(device)
        with torch.no_grad():
            prediction = model(input_tensor).item()
        st.write(f"🔮 Predikció valószínűség: {prediction:.4f}")
        st.write(f"🧾 Kategória: {'1' if prediction > 0.5 else '0'}")

    # 📝 Jegyzet megjelenítése
    if user_note:
        st.markdown("### 📝 Felhasználói megjegyzés")
        st.write(user_note)
