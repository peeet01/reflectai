import streamlit as st

st.set_page_config(page_title="ReflectAI App", layout="wide")
st.title("🧠 ReflectAI App Teszt")

try:
    from modules import kuramoto_sim
    st.sidebar.success("✅ kuramoto_sim betöltve")
except ImportError as e:
    st.sidebar.error(f"❌ kuramoto_sim hiba: {e}")

try:
    from modules import kuramoto_hebbian_sim
    st.sidebar.success("✅ kuramoto_hebbian_sim betöltve")
except ImportError as e:
    st.sidebar.error(f"❌ kuramoto_hebbian_sim hiba: {e}")

try:
    from modules import hebbian_learning
    st.sidebar.success("✅ hebbian_learning betöltve")
except ImportError as e:
    st.sidebar.error(f"❌ hebbian_learning hiba: {e}")

try:
    from modules import hebbian_learning_visual
    st.sidebar.success("✅ hebbian_learning_visual betöltve")
except ImportError as e:
    st.sidebar.error(f"❌ hebbian_learning_visual hiba: {e}")

try:
    from modules import xor_prediction
    st.sidebar.success("✅ xor_prediction betöltve")
except ImportError as e:
    st.sidebar.error(f"❌ xor_prediction hiba: {e}")
