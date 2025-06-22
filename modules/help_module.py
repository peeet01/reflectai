import streamlit as st

def run():
    st.title("📘 Súgó – Neurolab AI Sandbox")

    st.markdown("""
    ## 🔍 Mi ez az alkalmazás?

    A **Neurolab AI** egy kutatásorientált sandbox platform, amely lehetővé teszi neurális hálózatok, szinkronizációs rendszerek,
    topológiai modellek és tanulási folyamatok vizualizációját és interaktív tesztelését.

    ---
    ## 🧭 Modulismertető

    ### 🔁 XOR predikció neurális hálóval
    - MLP tanítás zajjal, predikció, tanítási statisztikák, CSV export, 3D felület, konfúziós mátrix

    ### 🧭 Kuramoto szinkronizáció
    - Oszcillátorok kezdeti–végső fáziseloszlása, szinkronizációs index (r), 3D dendrites vizualizáció

    ### 🧠 Hebbian tanulás
    - Hebbian szabály, tanulási ráta beállítása, súlymátrix megjelenítés

    ### ⚡ Kuramoto–Hebbian hálózat
    - Dinamikus háló kombináció Hebbian és Kuramoto modellekből

    ### 🔒 Topológiai szinkronizáció
    - Topológia hatása szinkronizációra

    ### 🌀 Lorenz rendszer
    - Kaotikus szimuláció vizualizálása

    ### 🔮 Lorenz predikció
    - Neurális hálós előrejelzés kaotikus rendszerre

    ### 🧬 Zajtűrés
    - Rendszer viselkedése különböző zajszintek mellett

    ### 🧠 Insight Learning
    - Hirtelen belátás modellezése (problémamegoldás)

    ### 📈 ESN predikció
    - Echo State Network időfüggő adatokra

    ### 🔄 Hebbian plaszticitás
    - Tanulási súlyok időbeli változása

    ### 🧮 Fraktáldimenzió
    - Szinkronhálózat fraktáldimenziójának vizsgálata

    ### 🧠 Generatív Kuramoto
    - Automatikus hálógenerálás és szinkronizáció

    ### 🧭 Memória tájkép
    - Hálózat memóriaállapotainak feltérképezése

    ---
    ## 👤 Célközönség
    - **Kutatók:** gyors elméleti verifikációk
    - **Diákok:** saját modellek tesztelése
    - **Fejlesztők:** moduláris rendszerépítés
    - **Oktatók:** szemléltető példák tanításhoz
    """)
