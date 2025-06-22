import streamlit as st

def run(): st.title("❓ Súgó – Neurolab AI Scientific Playground Sandbox")

st.markdown("""
## 🔍 Mi ez az alkalmazás?
A **Neurolab AI** egy interaktív kutatási platform, amely lehetővé teszi különféle mesterséges intelligencia, hálózati dinamika és matematikai modellek vizsgálatát. A cél egy nyitott, vizualizáció-alapú, moduláris sandbox létrehozása kutatók, hallgatók és oktatók számára.
""")

st.markdown("## 🧭 Modulismertető – Tudományos és történeti háttérrel")

st.markdown("### 🔁 XOR predikció neurális hálóval")
st.markdown("""
**Cél:** A klasszikus XOR logikai függvény megtanítása mesterséges neurális hálóval.  
**Tudományos háttér:** Az XOR probléma a 80-as években kulcsszerepet játszott a mély tanulás fejlődésében. Egyetlen lineáris réteg nem tudja megoldani, így legalább egy rejtett rétegre van szükség.

**Alkalmazott képlet:**  
$$
\hat{y} = \sigma\left(W_2 \cdot \tanh(W_1 \cdot x + b_1) + b_2\right)
$$

**Paraméterek:**
- Rejtett réteg mérete: A háló komplexitása
- Tanulási ráta: A súlyfrissítések mértéke
- Epochok: Tanítási ciklusok száma

**Alkalmazások:**
- Nemlineáris osztályozási problémák megoldása
- Gépi tanulás alapjainak bemutatása
""")

st.markdown("### 🧭 Kuramoto szinkronizáció")
st.markdown("""
**Cél:** Oszcillátorok kollektív szinkronizációjának vizsgálata.  
**Tudományos háttér:** A Kuramoto-modellt Yoshiki Kuramoto japán fizikus vezette be 1975-ben.

**Kuramoto-egyenlet:**
$$
\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^{N} \sin(\theta_j - \theta_i)
$$

**Order parameter:**
$$
r(t) = \left|\frac{1}{N} \sum_{j=1}^N e^{i \theta_j(t)} \right|
$$

**Paraméterek:**
- Kapcsolási erősség (K)
- Oszcillátorok száma

**Alkalmazások:**
- Idegi oszcillációk
- Ritmusgenerálás
- Szinkronizációs zavarok vizsgálata
""")

st.markdown("### 🧠 Hebbian tanulás")
st.markdown("""
**Cél:** A tanulás biológiai modellje – ha két neuron egyidejűleg aktiválódik, akkor kapcsolatuk erősödik.  
**Történeti háttér:** Donald Hebb 1949-ben fogalmazta meg ezt az elvet.

**Hebb-szabály:**  
$$
\Delta w_{ij} = \eta \, x_i \, y_j
$$

**Paraméterek:**
- Tanulási ráta (η)
- Neuronok száma

**Alkalmazások:**
- Szinaptikus plaszticitás
- Tanulási szabályok modellezése
- Asszociatív memória
""")

st.markdown("### ⚡ Kuramoto–Hebbian hálózat")
st.markdown("""
**Cél:** Dinamikus oszcillátorhálózatok és adaptív tanulási szabály összekapcsolása.  
**Magyarázat:** A Kuramoto-dinamika hat a kapcsolat súlyaira, miközben a Hebbian-szabály az összekötések erősségét adaptálja.

$$
\Delta w_{ij}(t) \propto \cos(\theta_i(t) - \theta_j(t))
$$

**Alkalmazások:**
- Szinaptikus adaptáció és ritmusmodellezés kombinálása
- Biológiai inspirációjú komplex rendszerek vizsgálata
""")

st.markdown("### 🌀 Lorenz rendszer")
st.markdown("""
**Cél:** A kaotikus viselkedés vizsgálata determinisztikus rendszerben.  
**Történeti háttér:** Edward Lorenz 1963-as meteorológiai modellje a káoszelmélet alapját képezte.

**Lorenz-egyenletek:**
$$
\begin{aligned}
\frac{dx}{dt} &= \sigma(y - x) \\
\frac{dy}{dt} &= x(\rho - z) - y \\
\frac{dz}{dt} &= xy - \beta z
\end{aligned}
$$

**Paraméterek:**
- σ, ρ, β: Rendszerkonstansok, amelyek a dinamika jellegét határozzák meg

**Alkalmazások:**
- Káoszdetekció
- Idősorok elemzése
""")

st.markdown("### 🔮 Lorenz predikció")
st.markdown("""
**Cél:** Mély neurális hálózat segítségével előrejelezni a Lorenz-rendszer jövőbeli állapotait.  
**Magyarázat:** Idősoros tanulás egy nemlineáris, determinisztikus rendszer alapján.

$$
\hat{x}_{t+1} = f(x_t, x_{t-1}, ...)
$$

**Alkalmazások:**
- Idősor előrejelzés
- Kaotikus rendszerek tanulása
""")

st.markdown("### 🧬 Zajtűrés és robusztusság")
st.markdown("""
**Cél:** A modellek érzékenységének mérése véletlenszerű zajra.  
**Motiváció:** A biológiai rendszerek gyakran robusztusak a hibák ellen, ezt modellezzük mesterséges rendszerekben.

**Alkalmazások:**
- Robusztus MI rendszerek fejlesztése
- Szinaptikus zajtűrés vizsgálata
""")

st.markdown("### 🧩 Chern–szám analízis")
st.markdown("""
**Cél:** Kvantumtopológiai jellemzők numerikus vizsgálata (pl. Berry-görbület).  

**Képlet:**
$$
C = \frac{1}{2\pi} \int_{BZ} F(k) \, d^2k
$$
ahol $F(k)$ a Berry-görbület, $BZ$ a Brillouin-zóna.

**Alkalmazások:**
- Topológiai izolátorok modellezése
- Kvantummechanikai hálózatelemzés
""")

st.markdown("### 📈 Echo State Network")
st.markdown("""
**Cél:** Dinamikus idősorok tanulása visszacsatolt hálóval.

**Képlet:**
$$
x(t+1) = \tanh(W_{res} \, x(t) + W_{in} \, u(t))
$$

**Alkalmazások:**
- Recurrent neural network (RNN) alapú tanulás
- Idősorok előrejelzése
- Időbeli mintázatok azonosítása
""")

st.markdown("### 🧠 Insight Learning")
st.markdown("""
**Cél:** Tanulás, amely hirtelen felismerésből következik, nem fokozatos fejlődésből.  
**Pszichológiai alap:** Köhler és a Gestalt-pszichológia elveiből származik.

**Alkalmazások:**
- Kognitív pszichológia modellezése
- Tanulási áttörések szimulációja
""")

st.markdown("### 🧠 Generatív Kuramoto hálózat")
st.markdown("""
**Cél:** Új gráfstruktúrák generálása és szinkronizációjuk vizsgálata Kuramoto-modell segítségével.

**Alkalmazások:**
- Gráf-generálás dinamikai célokra
- Hálózati adaptáció szimulációja
""")

st.markdown("### 🧭 Memória tájkép")
st.markdown("""
**Cél:** Neurális hálók stabil állapotainak (memória pontok) feltérképezése.  
**Elmélet:** A tájkép lokális minimumai stabil állapotokként viselkednek.

**Alkalmazások:**
- Energiaalapú hálók (pl. Hopfield-hálózatok)
- Attractor-analízis
""")

st.markdown("### 📊 Lyapunov spektrum")
st.markdown("""
**Cél:** Egy rendszer kaotikusságának kvantitatív jellemzése a Lyapunov-exponensek segítségével.

**Definíció:**
A pozitív Lyapunov-exponens a káosz egyik fő jellemzője:
$$
\lambda = \lim_{t \to \infty} \frac{1}{t} \ln \frac{\delta(t)}{\delta(0)}
$$

**Alkalmazások:**
- Káoszdetekció
- Rendszerstabilitás elemzése
""")

st.markdown("---")
st.markdown("## 👥 Célcsoport (Átfogó leírás)")
st.markdown("""
Az alkalmazás célcsoportja a tudományos és oktatási közösség széles spektrumát lefedi:

- **Kutatók:** Lehetőség van komplex rendszerek gyors tesztelésére, hipotézisek vizsgálatára és vizualizáció-alapú kutatásra.
- **Oktatók:** Az egyes modulok segítségével szemléletes módon lehet bemutatni matematikai modelleket és tanulási mechanizmusokat.
- **Hallgatók:** Interaktív környezetben kísérletezhetnek különböző paraméterekkel, mélyebb megértést szerezve a dinamikai rendszerekről és MI elvekről.
- **Fejlesztők:** Moduláris felépítése miatt könnyen bővíthető, módosítható, új kísérletek vagy vizualizációk beillesztésére alkalmas.
""")

