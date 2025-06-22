
import streamlit as st

def run():
    st.title("❓ Súgó / Ismertető")
    st.markdown("""
    ## 🧠 Neurolab AI Scientific Playground Sandbox

    Ez az alkalmazás egy **interaktív tudományos szimulációs környezet**, amely segíti a **neurális hálók, szinkronizációs modellek, predikciós algoritmusok és topológiai jellemzők** mélyebb megértését.

    ### 🎯 Célja:
    - **Kísérletezés** neurális hálózatokkal és fizikai szinkronizációs modellekkel.
    - **Adatok exportálása** további analízisre (pl. kutatási célból).
    - **Interaktív megjelenítés**, 2D és 3D vizualizációval.
    - **Predikciók, tanulási folyamatok, fraktáldimenziók, topológiai értékek** szemléltetése.

    ### 🗂️ Elérhető modulok:
    - **Kuramoto szinkronizáció**
    - **Hebbian tanulás**
    - **XOR predikció** (zaj, tanítási idő, mentés, interaktív input, 3D felület)
    - **Generatív Kuramoto háló**
    - **Lorenz-rendszerek**
    - **Topológiai védettség / Berry görbület**
    - **Insight learning**, **Echo State Network**, stb.

    ### 🔧 Használat:
    1. Válaszd ki a kívánt modult a bal oldali sávban.
    2. Állítsd be a paramétereket (pl. tanulási ráta, zaj szint, epoch).
    3. Figyeld meg az **interaktív grafikonokat, predikciókat**.
    4. (Opcionálisan) exportálj CSV-t vagy mentsd a modellt.

    ### 🧪 Sandbox funkció:
    Ez az app **kísérleti laboratóriumként** is működik, ahol a felhasználó **saját interaktív bemenettel**, megjegyzésekkel, exportált adatokkal tesztelheti a szimulációkat.

    ### 📩 Visszajelzés
    Ha hibát találsz vagy javaslatod van, kérlek jelezd a projektgazdának.

    ---
    **Verzió:** 1.0  
    **Készítette:** Neurolab AI
    """)
