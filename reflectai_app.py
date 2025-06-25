import streamlit as st 
import yaml 
from yaml.loader 
import SafeLoader 
import streamlit_authenticator as stauth from utils.metadata_loader 
import load_metadata from modules.modules_registry 
import MODULES

--- Konfiguráció betöltése ---

with open("config.yaml") as file: config = yaml.load(file, Loader=SafeLoader)

--- Autentikáció ---

authenticator = stauth.Authenticate( credentials=config['credentials'], cookie_name=config['cookie']['name'], key=config['cookie']['key'], cookie_expiry_days=config['cookie']['expiry_days'], preauthorized=config.get('preauthorized', {}) )

--- Bejelentkezés ---

name, authentication_status, username = authenticator.login("Bejelentkezés", "main")

--- Hitelesítés kezelése ---

if authentication_status is False: st.error("❌ Hibás felhasználónév vagy jelszó.") elif authentication_status is None: st.warning("⚠️ Kérlek jelentkezz be.") elif authentication_status: st.set_page_config(page_title="ReflectAI App", layout="wide") st.sidebar.success(f"✅ Bejelentkezve mint: {name} ({username})")

st.title(":brain: ReflectAI – Scientific Research Modules")
st.markdown("Válassz egy modult a bal oldali menüben az indításhoz.")

# Modulválasztó
st.sidebar.title(":file_folder: Modulok")
module_names = list(MODULES.keys())
selected_module = st.sidebar.radio("Válassz modult:", module_names)

# Modul betöltése
if selected_module:
    st.header(f"Modul: {selected_module}")

    # Metaadatok betöltése
    metadata_key = selected_module.replace(" ", "_").lower()
    metadata = load_metadata(metadata_key)

    with st.expander("📃 Modul metaadatai", expanded=False):
        st.json(metadata)

    # Modul futtatása
    MODULES[selected_module]()

