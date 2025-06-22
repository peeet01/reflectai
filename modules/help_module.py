import streamlit as st

def run():
    st.title("❓ Súgó – Neurolab AI Scientific Playground Sandbox")

    st.markdown("""
    ## 🔍 Mi ez az alkalmazás?

    A **Neurolab AI** egy nyílt kutatási sandbox, amely lehetővé teszi mesterséges intelligencia modellek, neurális hálók, dinamikai rendszerek és topológiai szimulációk vizsgálatát.  
    A célközönség: **kutatók, hallgatók, oktatók, fejlesztők**, akik vizuálisan és kvantitatívan szeretnék megérteni az MI-alapú rendszerek működését.

    ---

    ## 🧭 Modulismertető (Tudományos + Matematikai kiegészítésekkel)
    """)

    st.markdown("### 🔁 XOR predikció")
    st.markdown("""
    **Tudományos háttér:** Az XOR probléma nemlineáris, ezért nem tanítható meg egyetlen rétegű perceptronnal. Az MLP (Multi-Layer Perceptron) képes erre rejtett rétegek használatával.

    **Képlet:**  
    $$ \hat{y} = \sigma(W_2 \cdot \sigma(W_1 \cdot x + b_1) + b_2) $$
    """)

    st.markdown("### 🧭 Kuramoto szinkronizáció")
    st.markdown(r"""
    **Tudományos háttér:** A Kuramoto-modell oszcillátorok közötti fáziskoherenciát vizsgál.

    **Képlet:**  
    $$ \frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^{N} \sin(\theta_j - \theta_i) $$
    """)

    st.markdown("### 🧠 Hebbian tanulás")
    st.markdown(r"""
    **Tudományos háttér:** A tanulás biológiai elve: a szinapszis erősödik, ha a pre- és posztszinaptikus neuron együtt aktiválódik.

    **Képlet:**  
    $$ \Delta w_{ij} = \eta \cdot x_i \cdot y_j $$
    """)

    st.markdown("### ⚡ Kuramoto–Hebbian háló")
    st.markdown(r"""
    **Tudományos háttér:** A szinkronizáció és a tanulás dinamikája egyesítve: a kapcsolatok módosulnak a fáziskülönbségek alapján.

    **Képlet:**  
    $$ \frac{d\theta_i}{dt} = \omega_i + \sum_j K_{ij}(t) \sin(\theta_j - \theta_i) $$
    $$ \frac{dK_{ij}}{dt} = \eta \cos(\theta_i - \theta_j) $$
    """)

    st.markdown("### 🔒 Topológiai szinkronizáció")
    st.markdown(r"""
    **Tudományos háttér:** A hálózat gráfstruktúrája befolyásolja a szinkronizációs stabilitást.

    **Képlet:**  
    $$ \lambda_2 > 0 \Rightarrow \text{alapfeltétel a globális szinkronizációhoz (Algebraic connectivity)} $$
    """)

    st.markdown("### 🌀 Lorenz rendszer")
    st.markdown(r"""
    **Tudományos háttér:** Kaotikus viselkedésű háromdimenziós rendszer, amely érzékeny a kezdeti feltételekre.

    **Képlet:**  
    $$
    \begin{aligned}
    \dot{x} &= \sigma(y - x) \\
    \dot{y} &= x(\rho - z) - y \\
    \dot{z} &= xy - \beta z
    \end{aligned}
    $$
    """)

    st.markdown("### 🔮 Lorenz predikció")
    st.markdown(r"""
    **Tudományos háttér:** A Lorenz-rendszer idősorainak predikciója MLP vagy RNN segítségével.

    **Képlet:**  
    $$ \hat{x}_{t+1} = f(x_t, x_{t-1}, ..., x_{t-n}) $$
    """)

    st.markdown("### 🧬 Zajtűrés és robusztusság")
    st.markdown(r"""
    **Tudományos háttér:** Vizsgáljuk, hogy a hálózat mennyire stabil zajjal szemben.

    **Képlet:**  
    $$ x_{\text{noisy}} = x + \mathcal{N}(0, \sigma^2) $$
    """)

    st.markdown("### 🧩 Topológiai Chern–szám")
    st.markdown(r"""
    **Tudományos háttér:** Kvantált topológiai szám, amely leírja a rendszer globális szerkezetét.

    **Képlet:**  
    $$ C = \frac{1}{2\pi} \int_{\text{BZ}} \mathcal{F}(k) \, d^2k $$
    """)

    st.markdown("### 🧠 Insight Learning")
    st.markdown("""
    **Tudományos háttér:** A tanulás egy hirtelen felismerésen (aha-élmény) alapul, nem fokozatos próbálkozásokon.

    (A modul szimuláció alapú, matematikai modell nincs hozzárendelve.)
    """)

    st.markdown("### 📈 Echo State Network (ESN)")
    st.markdown(r"""
    **Tudományos háttér:** A visszacsatolt neurális hálók egyik típusa, memóriaeffektusokkal.

    **Képlet:**  
    $$
    x(t+1) = \tanh(W_{res} \cdot x(t) + W_{in} \cdot u(t)) \\
    \hat{y}(t) = W_{out} \cdot x(t)
    $$
    """)

    st.markdown("### 🔄 Hebbian plaszticitás")
    st.markdown(r"""
    **Tudományos háttér:** A tanulási súlyok időbeli alakulása Hebbian szabály szerint.

    **Képlet:**  
    $$ \frac{dW_{ij}}{dt} = \eta x_i y_j $$
    """)

    st.markdown("### 🧮 Szinkronfraktál dimenzióanalízis")
    st.markdown(r"""
    **Tudományos háttér:** A szinkronizáció által létrejövő fraktálszerkezetek dimenziójának meghatározása.

    **Képlet (box-counting):**  
    $$ D = \lim_{\epsilon \to 0} \frac{\log N(\epsilon)}{\log(1/\epsilon)} $$
    """)

    st.markdown("### 🧠 Generatív Kuramoto háló")
    st.markdown(r"""
    **Tudományos háttér:** Random gráfok generálása és szinkronizációs elemzése Kuramoto-dinamikával.

    (Paraméterfüggő sztochasztikus hálózat, explicit képlet nincs.)
    """)

    st.markdown("### 🧭 Memória tájkép")
    st.markdown(r"""
    **Tudományos háttér:** Az állapottér topográfiája és az energiafelszínek feltérképezése.

    **Képlet (Hopfield-féle energia):**  
    $$ E = -\frac{1}{2} \sum_{i,j} w_{ij} s_i s_j $$
    """)

    st.markdown("---")
    st.markdown("## 📦 Exportálás és mentés")
    st.markdown("""
    - CSV export: predikciós eredményekhez  
    - Modellmentés: `.pth` formátumban  
    - Jegyzetek mentése `.txt` fájlba
    """)

    st.markdown("## 👥 Célközönség")
    st.markdown("""
    - **Kutatók:** elméleti modellvalidálás  
    - **Oktatók:** szemléltető eszköz  
    - **Diákok:** tanulási kísérletek  
    - **Fejlesztők:** moduláris bővíthetőség
    """)
