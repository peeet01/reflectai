import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 🔧 Többrétegű háló dinamikusan
class DeepXORNet(nn.Module):
    def __init__(self, input_size=2, hidden_sizes=[4]):
        super(DeepXORNet, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.Tanh())
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

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
def run(learning_rate=0.1, epochs=1000, note=""):
    st.subheader("🔁 XOR predikció neurális hálóval")
    st.markdown("Ez a modul egy többrétegű neurális háló segítségével tanítja meg az XOR függvényt, zajjal, mentéssel és exporttal kiegészítve.")

    # 📋 Paraméterek
    noise_level = st.slider("Zaj szintje", 0.0, 0.5, 0.1, 0.01)
    export_results = st.checkbox("📤 Eredmények exportálása CSV-be")
    save_model_flag = st.checkbox("💾 Modell mentése")
    custom_input = st.checkbox("🎛️ Egyéni input kipróbálása tanítás után")
    user_note = st.text_area("📝 Megjegyzésed", value=note)

    # 🧠 Többrétegű architektúra kiválasztása
    st.markdown("### 🔧 Rejtett rétegek")
    hidden_sizes = st.multiselect("Válassz rétegeket", options=[2, 4, 8, 16, 32], default=[4, 4])
    if not hidden_sizes:
        st.warning("⚠️ Legalább egy rejtett réteg szükséges!")
        return

    # 🧠 GPU támogatás
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔢 Alap adatok
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    y = np.array([[0],[1],[1],[0]], dtype=np.float32)
    X_noisy = add_noise(X, noise_level)

    X_tensor = torch.tensor(X_noisy, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    # 🧠 Modell létrehozása
    model = DeepXORNet(hidden_sizes=hidden_sizes).to(device)
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
