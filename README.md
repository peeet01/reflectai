# 🧠 Neurolab AI Sandbox

**Interaktív tudományos szimulációk idegtudomány, komplex rendszerek és gépi tanulás területén.**  
Ez a Streamlit-alapú platform lehetőséget ad kutatók, diákok és tanárok számára, hogy **vizualizálják**, **szimulálják** és **megértsék** a neurális és dinamikus rendszerek viselkedését.

---

## 🎯 Célkitűzés

A **Neurolab AI Sandbox** célja egy nyílt, moduláris környezet biztosítása az alábbiakhoz:

- ❇️ Neurális tanulási szabályok (pl. Hebbian, STDP, Oja, BCM) szemléltetése  
- 🔁 Dinamikus rendszerek (pl. Kuramoto, Lorenz, Hopfield) vizsgálata  
- 🌀 Kaotikus és szinkronizációs jelenségek modellezése  
- 📈 Gépi tanulási és predikciós modellek kipróbálása  
- 📊 Információelméleti mérések és topológiai analízis  

---

## 🚀 Elérhető modulok

| Modul                     | Leírás |
|---------------------------|--------|
| 🕸️ **Kuramoto Modell**             | Oszcillátor-hálózatok szinkronizációja |
| 🧠 **Hebbian Learning**            | Szinaptikus súlyok erősítése korrelált aktivitásra |
| 🔁 **Kuramoto–Hebbian Szimuláció**| Kollektív tanulás és szinkronizáció |
| 💡 **Insight Learning**           | Belátásos tanulás reprezentációalapú stratégiával |
| 🧠 **BCM Tanulás**                | Dinamikus aktivitásküszöb szerinti súlymódosítás |
| 🌪️ **Lorenz Szimuláció**         | Kaotikus rendszerek időbeli előrejelzése |
| 🔮 **Echo State Network (ESN)**   | Rezervoár-alapú idősoros predikció |
| 🧠 **Neural Entropy**            | Információtartalom és komplexitás mérése |
| ⚡ **Spiking Neural Network (SNN)** | LIF neuronmodell STDP tanulással |
| 🧠 **Hebbian Visualizer**         | Interaktív Hebbian súlymódosítás |
| 🧠 **Oja Learning**               | PCA-szerű tanulás súlynormalizálással |
| ❌ **XOR Predikció**              | Klasszikus nemlineáris probléma tanulása |
| 📶 **Noise Robustness**           | Predikciós modellek zajtűrése |
| 🔬 **Plasticity Dynamics**        | Különböző tanulási szabályok összehasonlítása |
| 🧠 **Memory Landscape**           | Hopfield-háló energiaviszonya és memóriái |
| 🌐 **Berry-görbület**             | Kvantum topológia vizualizáció |
| 🌋 **Criticality Explorer**       | Neurális rendszerek kritikus állapotai |
| 📉 **Lyapunov Spectrum**          | Kaotikus rendszerek stabilitása |
| 🌀 **Fraktál Explorer**           | Mandelbrot és Julia-halmazok |
| 🧮 **Fractal Dimension**          | Adatstruktúrák komplexitásának mérése |
| 🧮 **Graph Sync Analysis**        | Hálózati szinkronizáció és Laplace spektrum |
| 🧩 **Generative Kuramoto**        | Dinamikusan generált gráfok szimulációja |
| 🏔️ **Persistent Homology**        | Topológiai adatértelmezés és zajszűrés |

---

## 🔧 Telepítés

```bash
git clone https://github.com/<felhasznalo>/neurolab-ai.git
cd neurolab-ai
pip install -r requirements.txt
streamlit run app.py
