import streamlit as st 
import yaml
import streamlit_authenticator as stauth 
import os 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import logging

from modules.hebbian_learning import run_hebbian_learning from modules.kohonen_som import run_kohonen_som from modules.reflection_module import run_reflection_module from modules.context_modeling import run_context_model

Naplózás beállítása

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') logger = logging.getLogger(name)

Hitelesítési konfiguráció betöltése

with open("config.yaml") as file: config = yaml.safe_load(file)

authenticator = stauth.Authenticate( config['credentials'], config['cookie']['name'], config['cookie']['key'], config['cookie']['expiry_days'] )

Beléptetés

name, authentication_status, username = authenticator.login("Bejelentkezés", "main")

if authentication_status is False: st.error("❌ Hibás felhasználónév vagy jelszó.") elif authentication_status is None: st.warning("⚠️ Kérlek add meg a bejelentkezési adatokat.") elif authentication_status: authenticator.logout("Kijelentkezés", "sidebar") st.sidebar.title(f"🚀 Üdv, {name}!")

st.title("ReflectAI - Moduláris tanulási rendszer")

# Modul választás
modul_valasztas = st.sidebar.selectbox(
    "Válassz modult:",
    (
        "Hebbian tanulás",
        "Kohonen SOM",
        "Reflexiós modul",
        "Kontextus modell"
    )
)

try:
    if modul_valasztas == "Hebbian tanulás":
        run_hebbian_learning()
    elif modul_valasztas == "Kohonen SOM":
        run_kohonen_som()
    elif modul_valasztas == "Reflexiós modul":
        run_reflection_module()
    elif modul_valasztas == "Kontextus modell":
        run_context_model()
    else:
        st.warning("⚠️ Válassz modult a bal oldali menüben.")
except Exception as e:
    st.error(f"Hiba történt a modul futtatása közben: {e}")
    logger.exception("Modulhiba")

else: st.warning("Bejelentkezés szükséges a folytatáshoz.")

