def run():
    st.title("📈 Echo State Network (ESN) predikció")

    st.markdown("""
    Ez a modul bemutatja, hogyan lehet Echo State Network-öt alkalmazni Lorenz-rendszer előrejelzésére **vagy saját feltöltött adatokon való tanulásra**.
    A feltöltött adatnak legalább 3 oszlopos idősornak kell lennie (pl. x, y, z).
    """)

    steps = st.slider("Adatpontok száma", 500, 3000, 1000)
    train_fraction = st.slider("Tanítási arány", 0.1, 0.9, 0.5)
    reservoir_size = st.slider("Reservoir méret", 50, 500, 100)

    # ✅ Session-alapú adatkezelés
    if "uploaded_df" not in st.session_state:
        st.session_state["uploaded_df"] = get_uploaded_data()
    uploaded_df = st.session_state["uploaded_df"]

    use_uploaded = False

    if uploaded_df is not None:
        if uploaded_df.shape[1] >= 3:
            st.success("✅ Feltöltött adat sikeresen betöltve.")
            show_data_overview(uploaded_df)
            data = uploaded_df.iloc[:steps, :3].values
            use_uploaded = True
        else:
            st.warning("⚠️ A feltöltött adatnak legalább 3 oszloposnak kell lennie (pl. x, y, z).")

    if not use_uploaded:
        st.info("ℹ️ Lorenz-szimulációs adat kerül felhasználásra.")
        xs, ys, zs = generate_lorenz_data(steps)
        data = np.column_stack([xs, ys, zs])

    # 🌊 Tanító és cél adatok előkészítése
    X = data[:-1]
    y = data[1:, 0]  # Csak az x komponens predikciója

    split = int(train_fraction * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    esn = EchoStateNetwork(n_inputs=3, n_reservoir=reservoir_size)
    esn.fit(X_train, y_train)
    prediction = esn.predict(X_test)

    # 📈 Eredmény ábra
    fig, ax = plt.subplots()
    ax.plot(range(len(y_test)), y_test, label="Valós X")
    ax.plot(range(len(prediction)), prediction, label="Predikció", linestyle="--")
    ax.set_title("ESN előrejelzés Lorenz-rendszerre")
    ax.set_xlabel("Időlépések")
    ax.set_ylabel("X érték")
    ax.legend()
    st.pyplot(fig)
