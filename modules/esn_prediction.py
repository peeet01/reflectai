import streamlit as st import numpy as np import matplotlib.pyplot as plt

def generate_lorenz(T=100, dt=0.01, sigma=10.0, rho=28.0, beta=8/3): N = int(T / dt) x = np.zeros(N) y = np.zeros(N) z = np.zeros(N) x[0], y[0], z[0] = 1.0, 1.0, 1.0 for i in range(1, N): x[i] = x[i-1] + dt * sigma * (y[i-1] - x[i-1]) y[i] = y[i-1] + dt * (x[i-1] * (rho - z[i-1]) - y[i-1]) z[i] = z[i-1] + dt * (x[i-1] * y[i-1] - beta * z[i-1]) return x, y, z

def esn_train_predict(data, reservoir_size=500, spectral_radius=0.9, leaking_rate=0.3): from sklearn.linear_model import Ridge N = len(data) in_size = 1 out_size = 1 Win = np.random.rand(reservoir_size, in_size) * 2 - 1 W = np.random.rand(reservoir_size, reservoir_size) - 0.5 rho_W = np.max(np.abs(np.linalg.eigvals(W))) W *= spectral_radius / rho_W

X = np.zeros((reservoir_size, N))
x = np.zeros((reservoir_size,))

for t in range(1, N):
    u = data[t - 1]
    x = (1 - leaking_rate) * x + leaking_rate * np.tanh(np.dot(Win, u) + np.dot(W, x))
    X[:, t] = x

ridge = Ridge(alpha=1e-6)
ridge.fit(X[:, :-1].T, data[1:])
Wout = ridge.coef_.reshape(1, -1)
pred = Wout @ X
return pred.flatten()

def run(): st.title("📈 Echo State Network (ESN) predikció") T = st.slider("Szimuláció hossza (időegység)", 10, 200, 100) reservoir_size = st.slider("Reservoir méret", 100, 1000, 500, step=100) spectral_radius = st.slider("Spektrális sugár", 0.1, 1.5, 0.9) leaking_rate = st.slider("Szivárgási ráta", 0.0, 1.0, 0.3)

x, _, _ = generate_lorenz(T)
prediction = esn_train_predict(x, reservoir_size, spectral_radius, leaking_rate)

fig, ax = plt.subplots()
ax.plot(x, label="Eredeti", linewidth=2)
ax.plot(prediction, label="ESN predikció", linestyle='--')
ax.set_title("Lorenz idősor ESN predikcióval")
ax.legend()
st.pyplot(fig)

