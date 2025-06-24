import streamlit as st
import yaml
import streamlit_authenticator as stauth
from modules.modules_registry import MODULES, safe_run

# Autentikáció konfiguráció betöltése
with open("config.yaml") as file:
    config = yaml.safe_load(file)

# Streamlit Authenticator példányosítás (preauthorized NÉLKÜL)
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Bejelentkezési felület
name, authentication_status, username = authenticator.login("Bejelentkezés", "main")

if authentication_status is False:
    st.error("Hibás felhasználónév vagy jelszó.")

elif authentication_status is None:
    st.warning("Kérlek add meg a bejelentkezési adataidat.")

elif authentication_status:
    authenticator.logout("Kijelentkezés", "sidebar")
    st.sidebar.success(f"Szia, {name}!")

    # Modulválasztó
    st.title("ReflectAI - Modulválasztó")
    module_name = st.sidebar.selectbox("Válassz modult:", list(MODULES.keys()))

    if module_name:
        module_function = MODULES.get(module_name)
        if module_function:
            safe_run(module_function)
        else:
            st.error("Nem található a kiválasztott modul.")
