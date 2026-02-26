import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import json
import os
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

st.set_page_config(
    page_title="Analisis Sentimen Ulasan Gojek",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;0,14..32,800;1,14..32,400&display=swap');

html, body, [class*="css"], * {
    font-family: 'Inter', sans-serif !important;
}

.stApp { background: #0b0c14; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2.5rem 3rem 4rem !important;
    max-width: 1280px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0e101e !important;
    border-right: 1px solid #1e2235 !important;
    min-width: 220px !important;
    max-width: 220px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.8rem 1.4rem !important;
    width: 100% !important;
}

.brand {
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #1e2235;
}
.brand-name {
    font-size: 0.95rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.01em;
    white-space: nowrap;
}
.brand-sub {
    font-size: 0.7rem;
    color: #3d4466;
    margin-top: 2px;
    white-space: nowrap;
}

/* Nav radio — hide the circle, style the whole label */
.stRadio > label { display: none !important; }
.stRadio > div {
    gap: 2px !important;
    display: flex !important;
    flex-direction: column !important;
}
/* Target the clickable wrapper */
[data-testid="stSidebar"] [data-baseweb="radio"] {
    display: flex !important;
    align-items: center !important;
    padding: 0.5rem 0.8rem !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
    white-space: nowrap !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"]:hover {
    background: #161929 !important;
}
/* Hide the radio circle */
[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child {
    display: none !important;
}
/* Style the text */
[data-testid="stSidebar"] [data-baseweb="radio"] > div:last-child {
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    color: #4a5280 !important;
    letter-spacing: -0.01em !important;
    white-space: nowrap !important;
    padding: 0 !important;
    margin: 0 !important;
}
/* Active item */
[data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="true"] {
    background: #161929 !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="true"] > div:last-child {
    color: #6d83f3 !important;
    font-weight: 600 !important;
}

/* Page title area */
.page-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4361ee;
    margin-bottom: 0.35rem;
}
.page-title {
    font-size: 2rem;
    font-weight: 700;
    color: #dde3f5;
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin-bottom: 0.4rem;
}
.page-sub {
    font-size: 0.9rem;
    color: #3d4466;
    margin-bottom: 2rem;
}

/* Hero */
.hero {
    background: linear-gradient(140deg, #111428 0%, #16183a 55%, #0e1630 100%);
    border: 1px solid #1e2340;
    border-radius: 20px;
    padding: 2.8rem 3rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(67,97,238,0.07) 0%, transparent 60%);
    top: -150px; right: -100px;
    pointer-events: none;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #dde3f5;
    letter-spacing: -0.04em;
    line-height: 1.15;
    margin-bottom: 0.8rem;
}
.hero-title span {
    background: linear-gradient(90deg, #4361ee, #4cc9f0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-desc {
    font-size: 0.95rem;
    color: #3d4466;
    line-height: 1.7;
    max-width: 540px;
    margin-bottom: 1.6rem;
}
.tag {
    display: inline-block;
    padding: 0.28rem 0.75rem;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 500;
    margin-right: 0.4rem;
    background: #161929;
    color: #4a5280;
    border: 1px solid #1e2235;
}

/* Stat cards */
.stat-card {
    background: #0e101e;
    border: 1px solid #1e2235;
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    transition: border-color 0.2s, transform 0.2s;
}
.stat-card:hover {
    border-color: #2d3566;
    transform: translateY(-2px);
}
.stat-val {
    font-size: 1.9rem;
    font-weight: 800;
    color: #dde3f5;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.stat-lbl {
    font-size: 0.73rem;
    font-weight: 500;
    color: #3d4466;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Section label */
.sec-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #3d4466;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e2235;
}

/* Pipeline cards */
.pipe-card {
    background: #0e101e;
    border: 1px solid #1e2235;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin-bottom: 0.6rem;
}
.pipe-n {
    font-size: 1.1rem;
    font-weight: 800;
    color: #4361ee;
    letter-spacing: -0.04em;
    line-height: 1;
    flex-shrink: 0;
    min-width: 22px;
}
.pipe-t {
    font-size: 0.83rem;
    font-weight: 600;
    color: #8892c8;
    margin-bottom: 0.1rem;
}
.pipe-d { font-size: 0.75rem; color: #3d4466; line-height: 1.5; }

/* Sentiment result */
.res-box {
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.res-pos { background: #071810; border: 1px solid #0d4a2a; }
.res-neg { background: #170a0a; border: 1px solid #4a1010; }
.res-neu { background: #181208; border: 1px solid #4a3608; }

.res-lbl { font-size: 0.72rem; color: #3d4466; text-transform: uppercase; letter-spacing: 0.1em; }
.res-val { font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; margin-top: 0.25rem; }
.c-pos { color: #34d399; } .c-neg { color: #f87171; } .c-neu { color: #fbbf24; }

/* Empty state */
.empty-state {
    background: #0e101e;
    border: 1px dashed #1e2235;
    border-radius: 14px;
    padding: 3rem 2rem;
    text-align: center;
    color: #2a2f52;
    font-size: 0.9rem;
}

/* Input */
.stTextArea textarea {
    background: #0e101e !important;
    color: #dde3f5 !important;
    border: 1px solid #1e2235 !important;
    border-radius: 12px !important;
    font-size: 0.9rem !important;
}
.stTextArea textarea:focus {
    border-color: #4361ee !important;
    box-shadow: 0 0 0 3px rgba(67,97,238,0.1) !important;
}

/* Buttons */
.stButton > button {
    background: #4361ee !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.2rem !important;
    width: 100% !important;
    letter-spacing: -0.01em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #3451d1 !important;
    transform: translateY(-1px) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0e101e !important;
    border-radius: 10px !important;
    padding: 3px !important;
    border: 1px solid #1e2235 !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #3d4466 !important;
    font-weight: 500 !important;
    font-size: 0.83rem !important;
    padding: 0.45rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    background: #4361ee !important;
    color: white !important;
}

/* Cards */
[data-testid="stMetric"] {
    background: #0e101e !important;
    border: 1px solid #1e2235 !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: #3d4466 !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #dde3f5 !important; font-weight: 700 !important; }

/* Progress */
.stProgress > div > div {
    background: linear-gradient(90deg, #4361ee, #4cc9f0) !important;
    border-radius: 999px !important;
}
.stProgress > div { background: #1e2235 !important; border-radius: 999px !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2235 !important;
    border-radius: 12px !important;
}

/* Expander */
details {
    background: #0e101e !important;
    border: 1px solid #1e2235 !important;
    border-radius: 12px !important;
}

/* Selectbox / slider */
.stSelectbox label, .stSlider label {
    color: #3d4466 !important;
    font-size: 0.78rem !important;
}

/* Info table */
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #1e2235;
    font-size: 0.83rem;
}
.info-row .k { color: #3d4466; }
.info-row .v { color: #8892c8; font-weight: 500; }

/* Lib card */
.lib-card {
    background: #0e101e;
    border: 1px solid #1e2235;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
}
.lib-name { font-size: 0.83rem; font-weight: 600; color: #4361ee; }
.lib-desc { font-size: 0.74rem; color: #3d4466; margin-top: 1px; }

/* Step list */
.step-row {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    padding: 0.65rem 0;
    border-bottom: 1px solid #1e2235;
}
.step-num {
    width: 22px; height: 22px;
    border-radius: 50%;
    background: #161929;
    border: 1px solid #2d3566;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.65rem; font-weight: 700;
    color: #4361ee;
    flex-shrink: 0;
}
.step-t { font-size: 0.83rem; font-weight: 600; color: #8892c8; }
.step-d { font-size: 0.74rem; color: #3d4466; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def muat_model():
    m = joblib.load("model/model_sentimen.pkl")
    v = joblib.load("model/vektorizer_tfidf.pkl")
    return m, v

@st.cache_data
def muat_data():
    return pd.read_csv("data/ulasan_gojek_bersih.csv")

@st.cache_data
def muat_metadata():
    with open("model/metadata_model.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def muat_nlp():
    sw = StopWordRemoverFactory().create_stop_word_remover()
    st_ = StemmerFactory().create_stemmer()
    return sw, st_

SLANG = {
    "gk":"tidak","ga":"tidak","gak":"tidak","ngga":"tidak","nggak":"tidak",
    "yg":"yang","dgn":"dengan","sy":"saya","sdh":"sudah","udh":"sudah",
    "udah":"sudah","blm":"belum","blom":"belum","gmn":"bagaimana",
    "gimana":"bagaimana","krn":"karena","karna":"karena","tp":"tapi",
    "tpi":"tapi","bgt":"banget","banget":"sangat","aja":"saja","jg":"juga",
    "lg":"lagi","klo":"kalau","kalo":"kalau","bs":"bisa","dr":"dari",
    "utk":"untuk","tdk":"tidak","km":"kamu","lo":"kamu","lu":"kamu",
    "gue":"saya","gw":"saya","abis":"habis","lbh":"lebih","susah":"sulit",
}

def preprocess(teks, sw, st_):
    teks = str(teks).lower()
    teks = re.sub(r"http\S+|www\.\S+", "", teks)
    teks = re.sub(r"@\w+|#\w+", "", teks)
    teks = teks.encode("ascii", "ignore").decode("ascii")
    teks = re.sub(r"\d+", "", teks)
    teks = re.sub(r"[^\w\s]", " ", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    teks = " ".join([SLANG.get(k, k) for k in teks.split()])
    teks = sw.remove(teks)
    teks = st_.stem(teks)
    return teks


# ── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-name">Analisis Sentimen</div>
        <div class="brand-sub">Ulasan Gojek · Google Play</div>
    </div>
    """, unsafe_allow_html=True)

    halaman = st.radio(
        "nav",
        ["Beranda", "Cek Sentimen", "Statistik", "Data Ulasan", "Tentang Model"],
        label_visibility="collapsed"
    )


# ── Guard ─────────────────────────────────────────────
if not (os.path.exists("model/model_sentimen.pkl") and os.path.exists("data/ulasan_gojek_bersih.csv")):
    st.error("Model atau data belum tersedia. Jalankan skrip preprocessing dan modeling terlebih dahulu.")
    st.stop()

model, vektorizer = muat_model()
data = muat_data()
meta = muat_metadata()
sw, st_ = muat_nlp()


# ══════════════════════════════════════════
#  BERANDA
# ══════════════════════════════════════════
if halaman == "Beranda":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Analisis Sentimen<br><span>Ulasan Gojek</span></div>
        <div class="hero-desc">
            Implementasi NLP Pipeline menggunakan ulasan nyata dari Google Play Store.
            Dataset dikumpulkan langsung, diproses, dan dimodelkan untuk klasifikasi sentimen otomatis.
        </div>
        <span class="tag">Logistic Regression</span>
        <span class="tag">TF-IDF</span>
        <span class="tag">PySastrawi</span>
        <span class="tag">Google Play Scraper</span>
    </div>
    """, unsafe_allow_html=True)

    akurasi = meta.get("akurasi", 0)
    total   = meta.get("jumlah_data", 0)
    fitur   = meta.get("jumlah_fitur", 0)
    dist    = meta.get("distribusi", {})
    pos_pct = round(dist.get("Positif", 0) / total * 100, 1) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, f"{akurasi*100:.1f}%", "Akurasi Model"),
        (c2, f"{total:,}",          "Total Ulasan"),
        (c3, f"{fitur:,}",          "Fitur TF-IDF"),
        (c4, f"{pos_pct}%",         "Ulasan Positif"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">{val}</div>
                <div class="stat-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Pipeline</div>', unsafe_allow_html=True)

    pipe = [
        ("01", "Data Acquisition",    "Scraping 2.000 ulasan dari Google Play Store menggunakan google-play-scraper."),
        ("02", "Text Cleaning",        "Menghapus URL, mention, emoji, angka, dan karakter khusus dari teks."),
        ("03", "Pre-processing",       "Normalisasi kata slang, stopword removal, dan stemming dengan PySastrawi."),
        ("04", "Feature Engineering",  "Konversi teks ke vektor numerik menggunakan TF-IDF (5.000 fitur, unigram+bigram)."),
        ("05", "Modeling",             "Logistic Regression dengan split data 80/20 dan stratified sampling."),
        ("06", "Evaluation",           "Pengukuran akurasi, precision, recall, F1-score, dan confusion matrix."),
    ]
    ca, cb = st.columns(2)
    for i, (n, t, d) in enumerate(pipe):
        col = ca if i % 2 == 0 else cb
        with col:
            st.markdown(f"""
            <div class="pipe-card">
                <div class="pipe-n">{n}</div>
                <div>
                    <div class="pipe-t">{t}</div>
                    <div class="pipe-d">{d}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Distribusi Dataset</div>', unsafe_allow_html=True)
    wmap = {"Positif": "#34d399", "Negatif": "#f87171", "Netral": "#fbbf24"}
    col_l, col_r = st.columns([2, 3])
    with col_l:
        for lbl, jml in dist.items():
            pct = jml / total * 100 if total else 0
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;font-size:0.82rem;
                        color:#3d4466;margin-bottom:4px;">
                <span>{lbl}</span><span>{jml:,}</span>
            </div>
            """, unsafe_allow_html=True)
            st.progress(pct / 100)
    with col_r:
        fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor="none")
        colors = [wmap.get(l, "#3d4466") for l in dist.keys()]
        wedges, _, autotexts = ax.pie(
            list(dist.values()), labels=list(dist.keys()),
            colors=colors, autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="#0b0c14", linewidth=2.5),
            textprops=dict(color="#8892c8", fontsize=10)
        )
        for at in autotexts:
            at.set_color("#0b0c14")
            at.set_fontweight("bold")
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        st.pyplot(fig, transparent=True)
        plt.close()


# ══════════════════════════════════════════
#  CEK SENTIMEN
# ══════════════════════════════════════════
elif halaman == "Cek Sentimen":
    st.markdown('<div class="page-eyebrow">Real-time</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Cek Sentimen Ulasan</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Masukkan teks ulasan Gojek untuk dianalisis sentimennya.</div>', unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        ulasan_input = st.text_area(
            "Teks ulasan",
            placeholder="Tulis atau tempel ulasan di sini...",
            height=150,
        )

        analisis = st.button("Analisis", type="primary")

    with col_out:
        if analisis and ulasan_input.strip():
            with st.spinner("Memproses..."):
                bersih = preprocess(ulasan_input, sw, st_)
                fitur  = vektorizer.transform([bersih])
                hasil  = model.predict(fitur)[0]
                proba  = model.predict_proba(fitur)[0]
                kelas  = model.classes_

            css_m = {"Positif": "res-pos", "Negatif": "res-neg", "Netral": "res-neu"}
            c_m   = {"Positif": "c-pos",   "Negatif": "c-neg",   "Netral": "c-neu"}

            st.markdown(f"""
            <div class="res-box {css_m[hasil]}">
                <div class="res-lbl">Hasil Prediksi</div>
                <div class="res-val {c_m[hasil]}">{hasil}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='font-size:0.75rem;color:#3d4466;margin:0.8rem 0 0.4rem;text-transform:uppercase;letter-spacing:0.08em;'>Kepercayaan Model</div>", unsafe_allow_html=True)
            emap = {"Positif": "Positif", "Negatif": "Negatif", "Netral": "Netral"}
            for i, k in enumerate(kelas):
                p = proba[i]
                st.progress(p, text=f"{emap.get(k, k)}  {p*100:.1f}%")

            with st.expander("Detail pre-processing"):
                st.markdown("**Teks asli:**")
                st.code(ulasan_input, language=None)
                st.markdown("**Setelah pre-processing:**")
                st.code(bersih, language=None)

        elif analisis:
            st.warning("Isi teks ulasan terlebih dahulu.")
        else:
            st.markdown("""
            <div class="empty-state">
                Belum ada teks yang dianalisis.<br>Masukkan ulasan dan klik <strong>Analisis</strong>.
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════
#  STATISTIK
# ══════════════════════════════════════════
elif halaman == "Statistik":
    st.markdown('<div class="page-eyebrow">Eksplorasi Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Statistik & Grafik</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Distribusi sentimen, word cloud, dan evaluasi performa model.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Distribusi Sentimen", "Word Cloud", "Evaluasi Model"])

    with tab1:
        if os.path.exists("gambar/distribusi_sentimen.png"):
            st.image("gambar/distribusi_sentimen.png", use_container_width=True)
        else:
            st.info("Jalankan `03_modeling_evaluasi.py` untuk menghasilkan grafik ini.")

    with tab2:
        if os.path.exists("gambar/wordcloud_sentimen.png"):
            st.image("gambar/wordcloud_sentimen.png", use_container_width=True)
            st.caption("Ukuran kata mencerminkan frekuensi kemunculannya dalam kategori sentimen tersebut.")
        else:
            st.info("Jalankan `03_modeling_evaluasi.py` untuk menghasilkan grafik ini.")

    with tab3:
        if os.path.exists("gambar/confusion_matrix.png"):
            col_cm, col_m = st.columns([3, 2])
            with col_cm:
                st.image("gambar/confusion_matrix.png", use_container_width=True)
            with col_m:
                st.metric("Akurasi", f"{meta.get('akurasi', 0)*100:.2f}%")
                st.markdown("<br>", unsafe_allow_html=True)
                for snt in ["Positif", "Negatif", "Netral"]:
                    with st.expander(snt):
                        ca2, cb2, cc2 = st.columns(3)
                        ca2.metric("Presisi",  f"{meta.get(f'presisi_{snt.lower()}', 0)*100:.1f}%")
                        cb2.metric("Recall",   f"{meta.get(f'recall_{snt.lower()}',  0)*100:.1f}%")
                        cc2.metric("F1",       f"{meta.get(f'f1_{snt.lower()}',      0)*100:.1f}%")
        else:
            st.info("Jalankan `03_modeling_evaluasi.py` untuk menghasilkan grafik ini.")


# ══════════════════════════════════════════
#  DATA ULASAN
# ══════════════════════════════════════════
elif halaman == "Data Ulasan":
    st.markdown('<div class="page-eyebrow">Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Data Ulasan</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Jelajahi dataset ulasan Gojek yang digunakan untuk pelatihan model.</div>', unsafe_allow_html=True)

    cf1, cf2, cf3 = st.columns(3)
    with cf1:
        f_sent  = st.selectbox("Sentimen", ["Semua", "Positif", "Netral", "Negatif"])
    with cf2:
        f_bint  = st.selectbox("Bintang",  ["Semua", 1, 2, 3, 4, 5])
    with cf3:
        n_rows  = st.slider("Jumlah baris", 10, 200, 50)

    df_f = data.copy()
    if f_sent != "Semua":
        df_f = df_f[df_f["sentimen"] == f_sent]
    if f_bint != "Semua":
        df_f = df_f[df_f["bintang"] == f_bint]

    st.markdown(f"<div style='font-size:0.8rem;color:#3d4466;margin:1rem 0 0.5rem;'>Menampilkan {min(n_rows, len(df_f)):,} dari {len(df_f):,} baris</div>", unsafe_allow_html=True)

    tampil = df_f[["isi_ulasan", "bintang", "sentimen", "tanggal_ulasan"]].head(n_rows)
    tampil.columns = ["Ulasan", "Bintang", "Sentimen", "Tanggal"]
    st.dataframe(tampil, use_container_width=True, height=420)

    csv = df_f.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("Download CSV", csv, file_name="ulasan_gojek.csv", mime="text/csv")


# ══════════════════════════════════════════
#  TENTANG MODEL
# ══════════════════════════════════════════
elif halaman == "Tentang Model":
    st.markdown('<div class="page-eyebrow">Dokumentasi</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Tentang Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Spesifikasi model, konfigurasi, dan pipeline pre-processing yang digunakan.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sec-label">Konfigurasi Model</div>', unsafe_allow_html=True)
        params = [
            ("Algoritma",         "Logistic Regression"),
            ("Feature Extraction","TF-IDF"),
            ("Jumlah Fitur",      "5.000"),
            ("N-gram Range",      "(1, 2) — Unigram & Bigram"),
            ("Min Doc. Frequency","2"),
            ("Sublinear TF",      "True"),
            ("Split Data",        "80% latih / 20% uji"),
            ("Stratified Split",  "Ya"),
            ("Bahasa",            "Indonesia"),
            ("Stemmer",           "PySastrawi"),
        ]
        for k, v in params:
            st.markdown(f"""
            <div class="info-row">
                <span class="k">{k}</span>
                <span class="v">{v}</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="sec-label">Pipeline Pre-processing</div>', unsafe_allow_html=True)
        steps = [
            ("1",  "Lowercase",           "Konversi seluruh teks ke huruf kecil"),
            ("2",  "Hapus URL & Mention", "Menghapus link, @user, dan #hashtag"),
            ("3",  "Hapus Non-ASCII",     "Menghapus emoji dan karakter unicode"),
            ("4",  "Hapus Angka",         "Menghapus semua token numerik"),
            ("5",  "Hapus Tanda Baca",    "Menghapus simbol dan tanda baca"),
            ("6",  "Normalisasi Slang",   "Mengubah kata gaul ke bentuk baku"),
            ("7",  "Stopword Removal",    "Menghapus kata-kata umum tidak bermakna"),
            ("8",  "Stemming",            "Mengubah kata ke bentuk dasar"),
            ("9",  "TF-IDF Vectorizer",   "Mengubah teks ke representasi vektor"),
        ]
        for n, t, d in steps:
            st.markdown(f"""
            <div class="step-row">
                <div class="step-num">{n}</div>
                <div>
                    <div class="step-t">{t}</div>
                    <div class="step-d">{d}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Library yang Digunakan</div>', unsafe_allow_html=True)
    libs = [
        ("google-play-scraper", "Scraping ulasan dari Google Play Store"),
        ("PySastrawi",          "Stemmer dan stopword Bahasa Indonesia"),
        ("scikit-learn",        "TF-IDF vectorizer dan Logistic Regression"),
        ("streamlit",           "Framework untuk membangun web app"),
        ("wordcloud",           "Membuat visualisasi word cloud"),
        ("matplotlib / seaborn","Visualisasi grafik statistik"),
    ]
    la, lb = st.columns(2)
    for i, (nm, ds) in enumerate(libs):
        col = la if i % 2 == 0 else lb
        with col:
            st.markdown(f"""
            <div class="lib-card">
                <div class="lib-name">{nm}</div>
                <div class="lib-desc">{ds}</div>
            </div>
            """, unsafe_allow_html=True)
