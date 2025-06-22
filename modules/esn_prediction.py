import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go

from modules.data_upload import get_uploaded_data, show_data_overview


# 🔧 Neurális háló definíció
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


# 📊 Kiértékelés
def evaluate(model, inputs, targets):
    with torch.no_grad():
        predictions = model(inputs).round()
        accuracy = (predictions.eq(targets).sum().item()) / targets.size(0)
    return accuracy, predictions


# 🚀 Fő modul
def run(hidden_size=4, learning_rate=0.1, epochs=1000, note=""):
    st.subheader("🔁 XOR predikció neurális hálóval")

    noise_level = st.slider("Zaj szintje", 0.0, 0.5, 0.1, 0.01)
    export_results = st.checkbox("📤 Eredmények exportálása CSV-be")
    save_model_flag = st.checkbox("💾 Modell mentése")
    custom_input = st.checkbox("🎛️ Egyéni input kipróbálása tanítás után")
    user_note = st.text_area("📝 Megjegyzésed", value=note)

    # 📥 Adatok betöltése fájlból vagy alapértelmezett XOR-ból
    df = get_uploaded_data(required_columns=["Input1", "Input2", "Target"])
    use_uploaded = False

    if df is not None:
        show_data_overview(df)
        X = df[["Input1", "Input2"]].values.astype(np.float32)
        y = df[["Target"]].values.astype(np.float32)
        st.success("✅ Feltöltött adat használatban.")
        use_uploaded = True
    else:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([[0], [1], [1], [0]], dtype=np.float32)
        st.info("ℹ️ Alapértelmezett XOR adat használatban.")

    # 🌪️ Zaj hozzáadása
    X_noisy = add_noise(X, noise_level)
    X_tensor = torch.tensor(X_noisy, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # 🧠 Háló létrehozása
    model = XORNet(hidden_size=hidden_size)
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
            progress_text.text(f"Tanítás... {percent}%")

    train_time = time.time() - start_time

    # 📉 Veszteség görbe
    st.markdown("### 📉 Veszteség alakulása")
    st.line_chart(losses)

    # 🌊 3D predikciós tér
    st.markdown("### 🌊 Interaktív predikciós tér (3D)")
    grid_size = 100
    x_vals = np.linspace(0, 1, grid_size)
    y_vals = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

    with torch.no_grad():
        zz = model(grid_tensor).numpy().reshape(xx.shape)

    fig_3d = go.Figure(data=[go.Surface(z=zz, x=xx, y=yy, colorscale='Viridis')])
    fig_3d.update_layout(title="Predikciós felület", height=500)
    st.plotly_chart(fig_3d, use_container_width=True)

    # 📊 Kiértékelés
    accuracy, predictions = evaluate(model, X_tensor, y_tensor)
    st.success(f"✅ Pontosság: {accuracy * 100:.2f}%")
    st.info(f"🕒 Tanítás ideje: {train_time:.2f} másodperc")

    # 🧮 Konfúziós mátrix
    st.markdown("### 🧮 Konfúziós mátrix")
    cm = confusion_matrix(y_tensor.numpy(), predictions.numpy())
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    ax.set_xlabel("Predikció")
    ax.set_ylabel("Valós érték")
    st.pyplot(fig_cm)

    # 📤 Exportálás
    if export_results:
        results_df = pd.DataFrame({
            "Input1": X[:, 0],
            "Input2": X[:, 1],
            "Zaj": noise_level,
            "Valós kimenet": y.flatten(),
            "Predikció": predictions.numpy().flatten()
        })
        csv_path = "xor_results.csv"
        results_df.to_csv(csv_path, index=False)
        with open(csv_path, "rb") as f:
            st.download_button("📁 CSV letöltése", data=f, file_name="xor_results.csv", mime="text/csv")

    # 💾 Mentés
    if save_model_flag:
        save_model(model)
        st.success("💾 Modell elmentve `xor_model.pth` néven.")

    # 🧪 Egyéni input
    if custom_input:
        st.markdown("### 🧪 Egyéni input kipróbálása")
        input1 = st.slider("Input 1", 0.0, 1.0, 0.0)
        input2 = st.slider("Input 2", 0.0, 1.0, 0.0)
        input_tensor = torch.tensor([[input1, input2]], dtype=torch.float32)
        with torch.no_grad():
            prediction = model(input_tensor).item()
        st.write(f"🔮 Predikció: {prediction:.4f} → {'1' if prediction > 0.5 else '0'}")

    # 📝 Jegyzet
    if user_note:
        st.markdown("### 📝 Megjegyzés")
        st.write(user_note)
