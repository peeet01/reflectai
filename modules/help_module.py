
import streamlit as st

def run():
    st.title("❓ Súgó – Neurolab AI Scientific Playground Sandbox")

    st.markdown("""
    ## 🔍 Mi ez az alkalmazás?

    A **Neurolab AI** egy kutatásorientált sandbox platform, amely lehetővé teszi neurális hálózatok, dinamikus rendszerek és szinkronizációs modellek interaktív vizsgálatát.

    ---
    ## 🧭 Modulismertető
    """)

    with st.expander("🔁 XOR predikció neurális hálóval"):
        st.markdown("""
        **Cél:** Az XOR logikai művelet megtanítása egy több rétegű perceptronnal (MLP).

        **Tudományos háttér:** Az XOR nemlineáris probléma, amely egy rejtett réteggel oldható meg. A háló tanulása visszaterjesztéssel történik.
        """)
        st.latex(r"y = \sigma\left(W_2 \cdot \tanh(W_1 \cdot x + b_1) + b_2\right)")
        st.markdown("ahol \(\sigma\) a szigmoid aktivációs függvény.")

    with st.expander("🧭 Kuramoto szinkronizáció"):
        st.markdown("""
        **Cél:** Szinkronizáció modellezése oszcillátorok között.

        **Képlet:**
        """)
        st.latex(r"\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^{N} \sin(\theta_j - \theta_i)")
        st.markdown("Itt \(\theta_i\) az i-edik oszcillátor fázisa, \(\omega_i\) a természetes frekvencia, K a kapcsolási erősség.")

    with st.expander("🧠 Hebbian tanulás"):
        st.markdown("""
        **Cél:** A Hebb-elv szimulációja.

        **Képlet:**
        """)
        st.latex(r"\Delta w_{ij} = \eta \cdot x_i \cdot x_j")

    with st.expander("⚡ Kuramoto–Hebbian hálózat"):
        st.markdown("""
        **Képlet:**
        """)
        st.latex(r"\frac{d\theta_i}{dt} = \omega_i + \sum_j w_{ij} \sin(\theta_j - \theta_i)")
        st.latex(r"\Delta w_{ij} = \eta \cos(\theta_j - \theta_i)")

    with st.expander("🔒 Topológiai szinkronizáció"):
        st.markdown("A gráf Laplace-mátrixának spektruma meghatározza a szinkronizáció stabilitását.")

    with st.expander("🌀 Lorenz rendszer (szimuláció)"):
        st.latex(r"""
        \begin{cases}
        \dot{x} = \sigma(y - x) \\
        \dot{y} = x(\rho - z) - y \\
        \dot{z} = xy - \beta z
        \end{cases}
        """)

    with st.expander("🔮 Lorenz predikció"):
        st.markdown("Predikciós modell egy kaotikus rendszerre neurális hálóval.")

    with st.expander("🧬 Zajtűrés és szinkronizációs robusztusság"):
        st.latex(r"\omega_i \rightarrow \omega_i + \xi(t)")
        st.markdown("ahol \(\xi(t)\) időfüggő zajkomponens.")

    with st.expander("🧩 Chern–szám analízis"):
        st.latex(r"C = \frac{1}{2\pi} \int_{\mathcal{M}} \mathcal{F}_{xy} \, dx \, dy")

    with st.expander("📈 Echo State Network (ESN)"):
        st.latex(r"y(t) = W_{\text{out}} \cdot x(t)")

    with st.expander("🔄 Hebbian plaszticitás dinamikája"):
        st.latex(r"\Delta w_{ij}(t) = \eta \cdot x_i(t) \cdot x_j(t)")

    with st.expander("🧮 Fraktáldimenzió analízis"):
        st.latex(r"D_2 = \lim_{\epsilon \to 0} \frac{\log C(\epsilon)}{\log \epsilon}")

    with st.expander("🧠 Belátás alapú tanulás"):
        st.markdown("A megoldás váratlanul történik, a próbálkozás után hirtelen belátás következik.")

    with st.expander("🧠 Generatív Kuramoto hálózat"):
        st.markdown("Véletlenszerű gráfokra alkalmazott Kuramoto dinamikák.")

    with st.expander("🧭 Memória tájkép"):
        st.markdown("Állapottér stabil pontjainak feltérképezése, energialeképezéssel.")

    st.success("✅ Tudományos és matematikai modulismertetők betöltve.")
