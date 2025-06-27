import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Streamlit UI
st.title("🔁 XOR Prediction – Scientific Neural Network Playground")
st.markdown("""
Ez a modul egy mesterséges neurális hálózat segítségével jósolja meg az XOR logikai művelet eredményét.  
A hálózat működése valós időben követhető, beleértve a tanulási folyamatot, vizuális visszajelzésekkel és egy 3D-s reprezentációval.
""")

# Model setup
hidden_layer_sizes = st.slider("Hidden layer size", 2, 10, 4)
max_iter = st.slider("Maximum iterations", 100, 1000, 300, 50)
show_3d = st.checkbox("Show 3D surface visualization")

# Progress bar
progress_bar = st.progress(0)
status_text = st.empty()

model = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), activation='tanh', solver='adam', max_iter=1, warm_start=True)

losses = []

# Training loop with progress
for i in range(max_iter):
    model.fit(X_train, y_train)
    losses.append(model.loss_)
    progress_bar.progress((i+1)/max_iter)
    status_text.text(f"Iteration {i+1}/{max_iter} - Loss: {model.loss_:.4f}")

# Results
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.success(f"✅ Model Accuracy: {acc:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
fig_cm, ax_cm = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
disp.plot(ax=ax_cm)
st.pyplot(fig_cm)

# Loss curve
fig_loss, ax_loss = plt.subplots()
ax_loss.plot(losses)
ax_loss.set_title("Loss Curve")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
st.pyplot(fig_loss)

# 3D Visualization
if show_3d:
    xx, yy = np.meshgrid(np.linspace(0,1,50), np.linspace(0,1,50))
    zz = np.array([model.predict([[x,y]])[0] for x,y in zip(np.ravel(xx), np.ravel(yy))])
    zz = zz.reshape(xx.shape)

    fig3d = go.Figure(data=[go.Surface(z=zz, x=xx, y=yy, colorscale='Viridis')])
    fig3d.update_layout(title='3D Prediction Surface', autosize=True)
    st.plotly_chart(fig3d)

# ReflectAI integráció
app = lambda: None
app.__code__ = run.__code__
