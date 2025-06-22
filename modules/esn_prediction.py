import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Egyszerű Echo State Network modell
class ESN:
    def __init__(self, n_inputs, n_reservoir=100, spectral_radius=0.95, sparsity=0.1, seed=42):
        np.random.seed(seed)
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.Win = np.random.uniform(-1, 1, (n_reservoir, n_inputs))
        W = np.random.rand(n_reservoir, n_reservoir) - 0.5
        # Sparsity
        W[np.random.rand(*W.shape) > sparsity] = 0
        # Spectral radius normalization
        eigvals = np.linalg.eigvals(W)
        W *= spectral_radius / np.max(np.abs(eigvals))
        self.W = W
        self.Wout = None

    def _update(self, state, u):
        return np.tanh(np.dot(self.Win, u) + np.dot(self.W, state))

    def fit(self, X, y, washout=50, ridge=1e-6):
        n_samples = X.shape[0]
        states = np.zeros((n_samples, self.n_reservoir))
        state = np.zeros(self.n_reservoir)

        for t in range(n_samples):
            state = self._update(state, X[t])
            states[t] = state

        # Elhagyjuk a washout periódust
        states = states[washout:]
        y_target = y[washout:]

        # Ridge regression
        self.Wout = np.dot(np.linalg.pinv(states.T @ states + ridge * np.eye(self.n_reservoir)) @ states.T, y_target)

    def predict(self, X, initial_state=None):
        n_samples = X.shape[0]
        predictions = []
        state = np.zeros(self.n_reservoir) if initial_state is None else initial_state

        for t in range(n_samples):
            state = self._update(state, X[t])
            y_pred = np.dot(self.Wout, state)
            predictions.append(y_pred)

        return np.array(predictions)


def run():
    st.subheader("📈 Echo State Network (ESN) predikció")
    st.markdown("Ez a modul egy visszacsatolt neurális hálózatot (ESN) használ dinamikus időbeli minták előrejelzésére. Feltölthetsz saját adatot, vagy használhatod az alapértelmezett szinusz jelet.")

    # Adatfeltöltés
    uploaded_file = st.file_uploader("📤 CSV fájl feltöltése (opcionális)", type=["csv"])
    use_uploaded = False

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        use_uploaded = True

        input_cols = st.multiselect("Válaszd ki a bemeneti oszlop(oka)t", df.columns.tolist())
        output_col = st.selectbox("Válaszd ki a cél (output) oszlopot", df.columns.tolist())
    else:
        st.info("⚠️ Nem töltöttél fel adatot. Alapértelmezett szinusz hullámot használunk.")
        x = np.linspace(0, 20 * np.pi, 1000)
        y = np.sin(x).reshape(-1, 1)
        df = pd.DataFrame({'input': x, 'target': y.flatten()})
        input_cols = ['input']
        output_col = 'target'

    # Paraméterek
    n_reservoir = st.slider("Reservoir méret", 10, 500, 100)
    spectral_radius = st.slider("Spektrális sugár", 0.1, 1.5, 0.95)
    sparsity = st.slider("Sparsity", 0.01, 1.0, 0.1)
    washout = st.slider("Washout periódus", 0, 100, 50)

    # Előkészítés
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X = scaler_x.fit_transform(df[input_cols])
    y = scaler_y.fit_transform(df[[output_col]])

    esn = ESN(n_inputs=X.shape[1], n_reservoir=n_reservoir, spectral_radius=spectral_radius, sparsity=sparsity)
    esn.fit(X, y, washout=washout)
    y_pred = esn.predict(X)

    # Invertálás vissza az eredeti skálára
    y_orig = scaler_y.inverse_transform(y)
    y_pred_orig = scaler_y.inverse_transform(y_pred)

    # Vizualizáció
    st.markdown("### 🔍 Előrejelzés vs. Valós érték")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_orig, label="Valós", linewidth=2)
    ax.plot(y_pred_orig, label="ESN előrejelzés", linestyle="--")
    ax.legend()
    ax.set_title("Predikció összehasonlítás")
    st.pyplot(fig)

    # Hibametrika
    mse = mean_squared_error(y_orig, y_pred_orig)
    st.markdown(f"**📉 MSE (átl. négyzetes hiba):** `{mse:.6f}`")
