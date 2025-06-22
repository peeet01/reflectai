import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Adatfeltöltés modul
def upload_data():
    st.sidebar.markdown("### 📁 Adatfeltöltés")
    uploaded_file = st.sidebar.file_uploader("Tölts fel CSV fájlt predikcióhoz", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Feltöltött adat (első 5 sor):")
        st.write(df.head())
        return df
    return None

# Echo State Network implementáció
class ESN:
    def __init__(self, n_inputs, n_reservoir=100, spectral_radius=0.95, sparsity=0.2, random_state=42):
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.random_state = random_state
        self._init_weights()

    def _init_weights(self):
        np.random.seed(self.random_state)
        self.W_in = np.random.uniform(-0.5, 0.5, (self.n_reservoir, self.n_inputs))
        W = np.random.rand(self.n_reservoir, self.n_reservoir) - 0.5
        mask = np.random.rand(*W.shape) < self.sparsity
        W *= mask
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (self.spectral_radius / radius)

    def _update(self, state, input_data):
        pre_activation = np.dot(self.W_in, input_data) + np.dot(self.W, state)
        return np.tanh(pre_activation)

    def fit(self, inputs, outputs, washout=50, ridge_param=1e-6):
        states = np.zeros((inputs.shape[0], self.n_reservoir))
        state = np.zeros(self.n_reservoir)
        for t in range(inputs.shape[0]):
            state = self._update(state, inputs[t])
            states[t] = state
        self.scaler = StandardScaler().fit(states[washout:])
        states_scaled = self.scaler.transform(states[washout:])
        self.ridge = Ridge(alpha=ridge_param)
        self.ridge.fit(states_scaled, outputs[washout:])

    def predict(self, inputs):
        states = np.zeros((inputs.shape[0], self.n_reservoir))
        state = np.zeros(self.n_reservoir)
        for t in range(inputs.shape[0]):
            state = self._update(state, inputs[t])
            states[t] = state
        states_scaled = self.scaler.transform(states)
        return self.ridge.predict(states_scaled)

# Streamlit futtató
def run():
    st.subheader("📈 Echo State Network (ESN) predikció feltöltött adatokkal")

    df = upload_data()
    if df is None:
        st.warning("⚠️ Kérlek, tölts fel adatot a kezdéshez.")
        return

    input_cols = st.multiselect("Válaszd ki a bemeneti oszlopokat", df.columns.tolist())
    output_col = st.selectbox("Válaszd ki a predikciós célt", df.columns.tolist())

    if not input_cols or not output_col:
        st.info("ℹ️ Válassz ki bemenetet és célt a folytatáshoz.")
        return

    X = df[input_cols].values
    y = df[output_col].values.reshape(-1, 1)

    n_reservoir = st.slider("Reservoir méret", 10, 300, 100, 10)
    radius = st.slider("Spektrális sugár", 0.1, 1.5, 0.95)
    sparsity = st.slider("Ritkaság", 0.0, 1.0, 0.2)

    esn = ESN(n_inputs=len(input_cols), n_reservoir=n_reservoir, spectral_radius=radius, sparsity=sparsity)
    esn.fit(X, y)

    preds = esn.predict(X)
    st.line_chart({"Valós érték": y.flatten(), "Predikció": preds.flatten()})
