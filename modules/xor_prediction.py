def run(hidden_size=4, learning_rate=0.1, epochs=1000, note=""):
    st.subheader("🔁 XOR predikció neurális hálóval")

    st.markdown("Ez a modul egy egyszerű neurális háló segítségével tanítja meg az XOR függvényt, zajjal, mentéssel és exporttal kiegészítve.")

    # 📋 Beállítások
    noise_level = st.slider("Zaj szintje", 0.0, 0.5, 0.1, 0.01)
    export_results = st.checkbox("📤 Eredmények exportálása CSV-be")
    save_model_flag = st.checkbox("💾 Modell mentése")
    custom_input = st.checkbox("🎛️ Egyéni input kipróbálása tanítás után")
    user_note = st.text_area("📝 Megjegyzésed", value=note)

    # 🧠 Eszköz választás (GPU ha elérhető)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ÚJ

    # 🔢 Adatok
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    y = np.array([[0],[1],[1],[0]], dtype=np.float32)

    X_noisy = add_noise(X, noise_level)
    X_tensor = torch.tensor(X_noisy).to(device)  # ÚJ
    y_tensor = torch.tensor(y).to(device)        # ÚJ

    # 📐 Modell, veszteség, optimalizáló
    model = XORNet(hidden_size=hidden_size).to(device)  # ÚJ
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ⏱️ Tanítás idő mérése
    start_time = time.time()
    progress = st.progress(0)
    progress_text = st.empty()
    losses = []  # ÚJ: loss gyűjtés

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())  # ÚJ

        if epoch % max(1, epochs // 100) == 0 or epoch == epochs - 1:
            percent = int((epoch+1)/epochs * 100)
            progress.progress((epoch+1)/epochs)
            progress_text.text(f"Tanítás folyamatban... {percent}%")

    train_time = time.time() - start_time

    # 📈 Loss görbe megjelenítése
    st.markdown("### 📉 Veszteség alakulása")
    st.line_chart(losses)  # ÚJ

    # 📈 Eredmények
    accuracy, predictions = evaluate(model, X_tensor, y_tensor)
    st.success(f"✅ Tanítás kész! Pontosság: {accuracy * 100:.2f}%")
    st.info(f"🕒 Tanítás ideje: {train_time:.2f} másodperc")

    # 📤 Exportálás
    if export_results:
        results_df = pd.DataFrame({
            "Input1": X[:,0],
            "Input2": X[:,1],
            "Zaj": noise_level,
            "Valós kimenet": y.flatten(),
            "Predikció": predictions.cpu().numpy().flatten()  # ÚJ: CPU-ra vissza
        })
        csv_path = "xor_results.csv"
        results_df.to_csv(csv_path, index=False)
        with open(csv_path, "rb") as f:
            st.download_button("📁 CSV letöltése", data=f, file_name="xor_results.csv", mime="text/csv")

    # 💾 Modell mentés
    if save_model_flag:
        save_model(model)
        st.success("💾 Modell elmentve `xor_model.pth` néven.")

    # 🎛️ Egyéni input kipróbálása
    if custom_input:
        st.markdown("### 🧪 Egyéni input kipróbálása")
        input1 = st.slider("Input 1", 0.0, 1.0, 0.0)
        input2 = st.slider("Input 2", 0.0, 1.0, 0.0)
        input_tensor = torch.tensor([[input1, input2]], dtype=torch.float32).to(device)
        with torch.no_grad():
            prediction = model(input_tensor).item()
        st.write(f"🔮 Predikció valószínűség: {prediction:.4f}")  # ÚJ
        st.write(f"🧾 Kategória: {'1' if prediction > 0.5 else '0'}")

    # 📎 Jegyzet megjelenítése
    if user_note:
        st.markdown("### 📝 Felhasználói megjegyzés")
        st.write(user_note)
