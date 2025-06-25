reflectai_app.py

import streamlit as st
import yaml 
import streamlit_authenticator as stauth from modules import *  # Feltételezve, hogy minden modul ide van rendezve from utils import *

-- Betöltjük a configot --

with open('config.yaml') as file: config = yaml.safe_load(file)

-- Authentikáció beállítása --

authenticator = stauth.Authenticate( config['credentials'], config['cookie']['name'], config['cookie']['key'], config['cookie']['expiry_days'], config['preauthorized'] )

-- Login mező --

authentication_status = None name, authentication_status, username = authenticator.login( fields={"Form name": "Bejelentkezés"} )

-- Felhasználói állapot kezelése --

if authentication_status: authenticator.logout("Kijelentkezés", "sidebar") st.sidebar.success(f"Bejelentkezve: {username}")

st.title("ReflectAI")
oldal = st.sidebar.selectbox("Válassz modult", [
    "Kuramoto", "Lorenz", "MLP előrejelzés", "Lyapunov", "Topológia",
    "Memória", "Plaszticitás", "XOR", "Zaj-tűrés"])

if oldal == "Kuramoto":
    kuramoto.main()
elif oldal == "Lorenz":
    lorenz.main()
elif oldal == "MLP előrejelzés":
    mlp_predict_lorenz.main()
elif oldal == "Lyapunov":
    lyapunov_spectrum.main()
elif oldal == "Topológia":
    persistent_homology.main()
elif oldal == "Memória":
    memory_landscape.main()
elif oldal == "Plaszticitás":
    plasticity_dynamics.main()
elif oldal == "XOR":
    xor_prediction.main()
elif oldal == "Zaj-tűrés":
    noise_robustness.main()

elif authentication_status is False: st.error("❌ Hibás felhasználónév vagy jelszó.") elif authentication_status is None: st.warning("ℹ️ Kérlek jelentkezz be.")

