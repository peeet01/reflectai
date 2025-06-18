
import streamlit as st
from modules.xor_prediction import run_xor_prediction_with_mlp

st.title("🧠 XOR predikciós tanulás")

fig, acc = run_xor_prediction_with_mlp()
st.pyplot(fig)
st.success(f"Predikciós pontosság: {acc * 100:.2f}%")
