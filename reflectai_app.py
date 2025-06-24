import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import importlib
import os
from utils.metadata_loader import load_metadata

# Streamlit oldalbeállítás
st.set_page_config(page_title="ReflectAI", page_icon="🧠", layout="wide")

# Konfiguráció betöltése
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

# Hitelesítő inicializálása
authenticator = stauth.Authenticate(
    credentials=config["credentials"],
    cookie_name=config["cookie"]["name"],
    key=config["cookie"]["key"],
    cookie_expiry_days=config["cookie"]["expiry_days"],
    preauthorized=config.get("preauthorized", {})
)

# Bejelentkezési felület
name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status is False:
    st.error("Hibás felhasználónév vagy jelszó.")
elif authentication_status is None:
    st.warning("⚠️ Kérlek jelentkezz be.")
elif authentication_status:
    authenticator.logout("Kijelentkezés", "sidebar")
    st.sidebar.markdown(f"👤 **{name}** bejelentkezve")

    st.title("ReflectAI - Modulválasztó")

    # Modulok listázása
    module_files = [
        f for f in os.listdir("modules")
        if f.endswith(".py") and not f.startswith("__")
    ]

    module_keys = [f.replace(".py", "") for f in module_files]
    selected_module = st.sidebar.selectbox("📚 Modul kiválasztása", module_keys)

    # Metaadat betöltés
    metadata = load_metadata(selected_module)

    st.header(f"🧩 {metadata.get('title', selected_module)}")
    st.markdown(metadata.get("description", "Nincs leírás."))

    # Modul betöltése
    try:
        module = importlib.import_module(f"modules.{selected_module}")
        module.run()
    except Exception as e:
        st.error(f"Hiba a modul betöltése közben: {e}")
