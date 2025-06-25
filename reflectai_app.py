import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from utils.metadata_loader import load_metadata
from modules.modules_registry import MODULES

# --- Konfiguráció betöltése ---
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# --- Autentikáció ---
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config.get('preauthorized', {})
)

# --- Login (csak helyes paraméter: 'main' vagy 'sidebar') ---
name, authentication_status, username = authenticator.login("main")

# --- Hitelesítés állapot ---
if authentication_status is False:
    st.error("❌ Hibás felhasználónév vagy jelszó.")
elif authentication_status is None:
    st.warning("⚠️ Kérlek jelentkezz be.")
elif authentication_status:
    st.set_page_config(page_title="ReflectAI", layout="wide")
    st.sidebar.success(f"✅ Bejelentkezve mint: {name} ({username})")

    st.title("🧠 ReflectAI – Scientific Reflection")
    st.markdown("Válassz egy modult a bal oldali menüből.")

    st.sidebar.title("📂 Modulválasztó")
    selected_module_name = st.sidebar.radio("Modul kiválasztása:", list(MODULES.keys()))

    st.text_input("📝 Megfigyelés vagy jegyzet címe:", key="metadata_title")

    module_func = MODULES.get(selected_module_name)
    if module_func:
        module_func()
    else:
        st.error("❌ A kiválasztott modul nem található.")
