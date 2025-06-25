import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from utils.metadata_loader import load_metadata
from modules.modules_registry import MODULES

# Oldalbeállítás (csak akkor, ha már hitelesített a felhasználó)
st.set_page_config(page_title="Neurolab AI", layout="wide")

# Konfiguráció betöltése
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

# Hitelesítő példány
authenticator = stauth.Authenticate(
    credentials=config["credentials"],
    cookie_name=config["cookie"]["name"],
    key=config["cookie"]["key"],
    cookie_expiry_days=config["cookie"]["expiry_days"],
    preauthorized=config.get("preauthorized", {})
)

# Bejelentkezés – helyesen: először location, aztán a név
name, authentication_status, username = authenticator.login("main", "Bejelentkezés")

# Állapotkezelés
if authentication_status is False:
    st.error("❌ Hibás felhasználónév vagy jelszó.")
elif authentication_status is None:
    st.warning("⚠️ Kérlek jelentkezz be.")
elif authentication_status:
    st.sidebar.success(f"✅ Bejelentkezve mint: {name} ({username})")

    st.title("🧠 Neurolab AI – Scientific Reflection")
    st.markdown("Válassz egy modult a bal oldali menüből.")

    # Modulválasztó
    st.sidebar.title("📂 Modulválasztó")
    selected_module_name = st.sidebar.radio("Modul kiválasztása:", list(MODULES.keys()))

    # Metaadat mező
    st.text_input("📝 Megfigyelés vagy jegyzet címe:", key="metadata_title")

    # Modul betöltés
    module_func = MODULES.get(selected_module_name)
    if module_func:
        module_func()
    else:
        st.error("❌ A kiválasztott modul nem található.")
