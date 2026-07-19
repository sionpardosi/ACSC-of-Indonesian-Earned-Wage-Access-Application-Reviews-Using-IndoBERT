# =============================================================================
# Dashboard Analisis Sentimen ACSC - Ulasan Aplikasi EWA Indonesia
# Model utama : IndoBERT (indobenchmark/indobert-base-p1), Sentence-Pair
# Baseline    : TF-IDF + SVM (lower-bound konvensional)
# Data        : 2.266 baris ACSC final (2.031 ulasan unik, 6 aplikasi, 2020-2026)
#
# Catatan implementasi:
# 1) Teks masukan pada panel Analisis Ulasan Baru melewati praproses yang SAMA
#    dengan pipeline pelatihan (case folding, penghapusan noise, normalisasi
#    karakter berulang, normalisasi kata non-baku dengan kamus + protected
#    words) sebelum diinferensikan, agar representasi input konsisten.
# 2) Model ACSC mengasumsikan aspek sudah diketahui (a priori). Untuk ulasan
#    baru, relevansi aspek ditentukan lebih dulu secara heuristik leksikon
#    (fungsi deteksi kategori aspek), sehingga sentimen hanya diprediksi pada
#    aspek yang relevan. Mode prediksi seluruh aspek tetap tersedia.
# 3) Seluruh angka pada dashboard mengikuti hasil akhir penelitian:
#    IndoBERT Macro-F1 = 0,8762 +/- 0,0137 (rata-rata 5 seed);
#    checkpoint terbaik 0,8843; baseline SVM 0,8454; data uji 454 baris.
# aplikasi sangat mudah digunakan dan bagus, baik tetapi biaya mahal tidak bagus
# =============================================================================

import re
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Dashboard ACSC EWA Indonesia",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Gaya tampilan --------------------------------
st.markdown("""
<style>
  .block-container {padding-top: 1.4rem;}
  .judul-utama {font-size: 26px; font-weight: 700; color: #1f2937; margin-bottom: 2px;}
  .sub-judul   {font-size: 14px; color: #6b7280; margin-bottom: 12px;}
  .kartu {background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px;
          padding: 14px 16px; text-align: center;}
  .kartu .angka {font-size: 22px; font-weight: 700;}
  .kartu .label {font-size: 12px; color: #374151; font-weight: 600;}
  .kartu .ket   {font-size: 11px; color: #9ca3af;}
  .seksi {font-size: 16px; font-weight: 700; color: #111827;
          border-left: 4px solid #2563eb; padding-left: 8px; margin: 14px 0 8px 0;}
  .lencana {display: inline-block; padding: 2px 10px; border-radius: 12px;
            font-size: 12px; font-weight: 700; color: white;}
  .l-pos {background: #16a34a;} .l-neg {background: #dc2626;} .l-net {background: #6b7280;}
  .l-abu {background: #d1d5db; color: #374151;}
  .kotak-hasil {border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px 12px;
                margin-bottom: 8px; background: #fafafa;}
</style>
""", unsafe_allow_html=True)

def seksi(judul):
    st.markdown(f'<div class="seksi">{judul}</div>', unsafe_allow_html=True)

def kartu(angka, label, ket="", warna="#2563eb"):
    return (f'<div class="kartu"><div class="angka" style="color:{warna}">{angka}</div>'
            f'<div class="label">{label}</div><div class="ket">{ket}</div></div>')

# ----------------------------- Konstanta hasil ------------------------------
ASPEK = ["Kecepatan Pencairan", "Biaya/Potongan", "Kemudahan Penggunaan",
         "Customer Service", "Keandalan Sistem"]
ASPEK_INPUT = ["kecepatan pencairan", "biaya/potongan", "kemudahan penggunaan",
               "customer service", "keandalan sistem"]
LABEL = ["Positif", "Negatif", "Netral"]
WARNA = {"Positif": "#2563eb", "Negatif": "#dc2626", "Netral": "#9ca3af"}

# Distribusi dataset final (2.266 baris ACSC)
DIST_ASPEK = {
    "Kemudahan Penggunaan": {"Positif": 935, "Negatif": 244, "Netral": 19},
    "Keandalan Sistem":     {"Positif": 66,  "Negatif": 375, "Netral": 23},
    "Kecepatan Pencairan":  {"Positif": 93,  "Negatif": 168, "Netral": 35},
    "Biaya/Potongan":       {"Positif": 51,  "Negatif": 118, "Netral": 27},
    "Customer Service":     {"Positif": 22,  "Negatif": 62,  "Netral": 28},
}
DIST_GLOBAL = {"Positif": 1167, "Negatif": 967, "Netral": 132}

# Distribusi per platform (baris ACSC final) + interval kepercayaan Wilson 95%
PLATFORM = pd.DataFrame({
    "Platform": ["VENTENY", "Paywatch", "Mekari Flex", "GajiGesa", "Wagely", "AyoKasbon"],
    "n":        [588, 172, 572, 326, 573, 35],
    "PersenPositif": [67.2, 69.8, 49.8, 41.4, 38.4, 34.3],
    "Margin":        [3.8, 6.8, 4.1, 5.3, 4.0, 15.0],
})

# Performa akhir (data uji 454 baris, bebas kebocoran)
INDOBERT = {
    "MacroF1_mean": 0.8762, "MacroF1_std": 0.0137,
    "MacroF1_best": 0.8843, "Accuracy_best": 0.9075,
    "per_kelas": {
        "Positif": {"P": 0.9489, "R": 0.9028, "F1": 0.9253, "n": 247},
        "Negatif": {"P": 0.8711, "R": 0.9185, "F1": 0.8942, "n": 184},
        "Netral":  {"P": 0.8000, "R": 0.8696, "F1": 0.8333, "n": 23},
    },
}
SVM = {
    "MacroF1": 0.8454, "Accuracy": 0.8789,
    "per_kelas": {
        "Positif": {"P": 0.9057, "R": 0.8947, "F1": 0.9002, "n": 247},
        "Negatif": {"P": 0.8519, "R": 0.8750, "F1": 0.8633, "n": 184},
        "Netral":  {"P": 0.8095, "R": 0.7391, "F1": 0.7727, "n": 23},
    },
}
STABILITAS = pd.DataFrame({
    "Epoch": [1, 2, 3, 4],
    "Mean":  [0.9137, 0.9082, 0.9089, 0.9141],
    "Std":   [0.0083, 0.0159, 0.0191, 0.0107],
})
F1_PER_ASPEK = pd.DataFrame({
    "Aspek": ASPEK,
    "n_uji": [55, 42, 256, 18, 83],
    "IndoBERT": [0.787, 0.800, 0.940, 0.827, 0.694],
    "SVM":      [0.721, 0.806, 0.912, 0.725, 0.733],
})
ALUR_DATA = [
    ("Ulasan mentah", 3628), ("Setelah deduplication tahap 1", 3018),
    ("Kandidat anotasi", 2648), ("Ulasan teranotasi", 2075),
    ("Baris ACSC (dekomposisi)", 2311), ("Baris ACSC final", 2266),
]

# ----------------------- Praproses (identik pelatihan) ----------------------
PROTECTED_WORDS = {"venteny", "wagely", "paywatch", "gajigesa", "mekari",
                   "error", "login", "otp", "pin", "limit", "maintenance"}

KAMUS_BAWAAN = {
    "gak": "tidak", "ga": "tidak", "gk": "tidak", "nggak": "tidak", "ngga": "tidak",
    "tdk": "tidak", "bgt": "banget", "bener": "benar", "udah": "sudah", "udh": "sudah",
    "blm": "belum", "sdh": "sudah", "makasi": "terima kasih", "makasih": "terima kasih",
    "trims": "terima kasih", "thx": "terima kasih", "gmn": "bagaimana", "bgs": "bagus",
    "aplikasinya": "aplikasinya", "cepet": "cepat", "lemot": "lambat", "ribet": "rumit",
    "gampang": "mudah", "duit": "uang", "bs": "bisa", "krn": "karena", "dgn": "dengan",
    "yg": "yang", "sy": "saya", "sm": "sama", "tp": "tapi", "jg": "juga", "utk": "untuk",
}

@st.cache_resource(show_spinner=False)
def muat_kamus(path_kamus):
    """Memuat kamus normalisasi 4.975 entri dari file; jika tidak tersedia,
    memakai kamus bawaan ringkas dengan pemberitahuan kepada pengguna."""
    try:
        df = pd.read_excel(path_kamus)
        kolom = [c.lower() for c in df.columns]
        df.columns = kolom
        k_slang = "slang" if "slang" in kolom else kolom[0]
        k_baku = "baku" if "baku" in kolom else kolom[1]
        kamus = dict(zip(df[k_slang].astype(str).str.lower(),
                         df[k_baku].astype(str).str.lower()))
        return kamus, True
    except Exception:
        return dict(KAMUS_BAWAAN), False

def praproses_teks(teks, kamus):
    """Pipeline praproses yang sama dengan pelatihan (NB02):
    case folding, penghapusan noise, normalisasi karakter berulang (3+ menjadi 2),
    dan normalisasi kata non-baku dengan protected words."""
    t = teks.lower()                                          # case folding
    t = re.sub(r"http\S+|www\.\S+", " ", t)                   # URL
    t = re.sub(r"\S+@\S+", " ", t)                            # email
    t = re.sub(r"[@#]\w+", " ", t)                            # mention / hashtag
    t = re.sub(r"[^0-9a-z\s/]", " ", t)                       # emoji & simbol non-teks
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)                      # karakter berulang 3+ -> 2
    hasil = []
    for kata in t.split():
        if kata in PROTECTED_WORDS:
            hasil.append(kata)
        else:
            hasil.append(kamus.get(kata, kata))
    return re.sub(r"\s+", " ", " ".join(hasil)).strip()

# ------------------- Deteksi relevansi aspek (heuristik) --------------------
LEKSIKON_ASPEK = {
    "Kecepatan Pencairan": [
        "cair", "pencairan", "dicairkan", "cepat", "lama", "lambat", "proses",
        "transfer", "masuk", "menit", "jam", "hari", "dana", "tarik", "penarikan",
        "instan", "langsung",
    ],
    "Biaya/Potongan": [
        "biaya", "potongan", "dipotong", "admin", "fee", "mahal", "murah",
        "gratis", "tarif", "bunga", "persen", "charge",
    ],
    "Kemudahan Penggunaan": [
        "mudah", "gampang", "ribet", "rumit", "simpel", "praktis", "membantu",
        "daftar", "pendaftaran", "registrasi", "tampilan", "antarmuka",
        "navigasi", "digunakan", "pakai", "fitur", "user friendly",
    ],
    "Customer Service": [
        "cs", "customer", "service", "layanan", "respon", "respons", "dibalas",
        "hubungi", "dihubungi", "kontak", "komplain", "keluhan", "tanggap",
        "tanggapan", "bantuan", "dilayani",
    ],
    "Keandalan Sistem": [
        "error", "eror", "gangguan", "maintenance", "gagal", "crash", "bug",
        "lemot", "lambat", "macet", "server", "sistem", "login", "otp",
        "verifikasi", "force close", "keluar sendiri", "tidak bisa dibuka",
    ],
}

def deteksi_aspek(teks_bersih):
    """Deteksi kategori aspek yang relevan (Aspect Category Detection heuristik
    berbasis leksikon). Model ACSC mengasumsikan aspek sudah diketahui, sehingga
    untuk ulasan baru langkah deteksi ini diperlukan sebelum klasifikasi sentimen."""
    relevan = []
    for aspek, kata_kunci in LEKSIKON_ASPEK.items():
        for kk in kata_kunci:
            if kk in teks_bersih:
                relevan.append(aspek)
                break
    return relevan

# ------------------------------- Sidebar ------------------------------------
with st.sidebar:
    st.markdown("### Dashboard ACSC EWA")
    st.caption("Aspect Category Sentiment Classification pada ulasan enam "
               "aplikasi Earned Wage Access Indonesia menggunakan IndoBERT.")
    halaman = st.radio(
        "Navigasi",
        ["Ikhtisar Penelitian", "Dataset dan Pipeline", "Distribusi Sentimen",
         "Performa Model", "Analisis Ulasan Baru"],
    )
    st.markdown("---")
    with st.expander("Tentang sistem ini"):
        st.caption(
            "Prediksi sentimen dihasilkan oleh model IndoBERT yang di-fine-tune "
            "pada penelitian ini (bukan layanan AI eksternal). Prediksi bersifat "
            "probabilistik dan dapat keliru; nilai keyakinan ditampilkan sebagai "
            "bahan pertimbangan. Deteksi relevansi aspek untuk ulasan baru "
            "dilakukan secara heuristik berbasis leksikon kata kunci."
        )

# =============================== HALAMAN 1 ==================================
if halaman == "Ikhtisar Penelitian":
    st.markdown('<div class="judul-utama">Analisis Sentimen Berbasis Kategori Aspek '
                'pada Ulasan Aplikasi EWA Indonesia</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-judul">Aspect Category Sentiment Classification '
                '(ACSC) menggunakan IndoBERT | Kerangka kerja CRISP-DM | '
                'Enam platform, periode Januari 2020 - Mei 2026</div>',
                unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(kartu("2.266", "Baris Dataset ACSC", "dari 2.031 ulasan unik", "#7c3aed"),
                unsafe_allow_html=True)
    k2.markdown(kartu("6", "Aplikasi EWA", "Play Store 2020-2026", "#0891b2"),
                unsafe_allow_html=True)
    k3.markdown(kartu("5", "Kategori Aspek", "ditetapkan a priori", "#16a34a"),
                unsafe_allow_html=True)
    k4.markdown(kartu("0,8762", "Macro-F1 IndoBERT", "rata-rata 5 seed (+/- 0,0137)", "#dc2626"),
                unsafe_allow_html=True)
    k5.markdown(kartu("+0,0308", "Uplift vs Baseline", "SVM = 0,8454; uji N=454", "#f59e0b"),
                unsafe_allow_html=True)

    seksi("Ringkasan Penelitian")
    st.markdown(
        "Penelitian ini menerapkan fine-tuning IndoBERT dengan pendekatan "
        "Sentence-Pair Classification untuk mengklasifikasikan sentimen "
        "(positif, negatif, netral) terhadap lima kategori aspek layanan EWA: "
        "Kecepatan Pencairan, Biaya/Potongan, Kemudahan Penggunaan, Customer "
        "Service, dan Keandalan Sistem. Dataset dibangun dari 3.628 ulasan "
        "Google Play Store yang melalui praproses delapan tahap dan anotasi "
        "manual berpanduan, menghasilkan 2.266 baris data final. Pembagian data "
        "menggunakan GroupShuffleSplit berbasis identitas ulasan sehingga bebas "
        "kebocoran, dan pelatihan dijalankan lima kali dengan random seed "
        "berbeda untuk menjamin stabilitas hasil."
    )

    seksi("Kriteria Keberhasilan")
    st.markdown(
        "Kriteria yang ditetapkan adalah Macro-F1 minimal 0,80 pada data uji. "
        "Hasil akhir: rata-rata lima seed 0,8762 dan checkpoint terbaik 0,8843, "
        "keduanya melampaui kriteria. Baseline TF-IDF + SVM mencapai 0,8454, "
        "sehingga uplift arsitektur kontekstual sebesar +0,0308 pada data uji "
        "yang identik."
    )

# =============================== HALAMAN 2 ==================================
elif halaman == "Dataset dan Pipeline":
    st.markdown('<div class="judul-utama">Dataset dan Pipeline</div>',
                unsafe_allow_html=True)

    seksi("Alur Penyusutan Data")
    df_alur = pd.DataFrame(ALUR_DATA, columns=["Tahap", "Jumlah"])
    fig = go.Figure(go.Funnel(
        y=df_alur["Tahap"], x=df_alur["Jumlah"],
        textinfo="value", marker={"color": "#2563eb"}))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Deduplication tahap 1 menghapus 593 duplikat dan 17 baris kosong; "
        "penyaringan menandai 365 ulasan terlalu pendek dan 5 ambigu; anotasi "
        "manual men-skip 573 ulasan tanpa aspek; finalisasi membuang 44 "
        "near-duplicate pasca-normalisasi dan menyelesaikan 1 konflik label."
    )

    seksi("Pembagian Data (GroupShuffleSplit, bebas kebocoran)")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kartu("1.627", "Pelatihan", "71,8 persen", "#2563eb"), unsafe_allow_html=True)
    c2.markdown(kartu("185", "Validasi", "pemilihan checkpoint", "#0891b2"), unsafe_allow_html=True)
    c3.markdown(kartu("454", "Pengujian", "dikunci, evaluasi akhir", "#16a34a"), unsafe_allow_html=True)
    c4.markdown(kartu("0", "Kebocoran", "irisan teks train-test", "#dc2626"), unsafe_allow_html=True)
    st.caption(
        "Seluruh baris dari satu ulasan (review_id) dijamin berada pada subset "
        "yang sama. Baseline dilatih pada gabungan pelatihan dan validasi "
        "(1.812 baris) karena memakai validasi silang internal."
    )

    seksi("Sumber Data per Aplikasi")
    df_src = pd.DataFrame({
        "Aplikasi": ["Mekari Flex", "Wagely", "VENTENY", "GajiGesa", "Paywatch", "AyoKasbon"],
        "Ulasan mentah": [1101, 1002, 662, 498, 313, 52],
        "Proporsi": ["30,3%", "27,6%", "18,2%", "13,7%", "8,6%", "1,4%"],
        "File CSV": ["raw_mekari_flex.csv", "raw_wagely.csv", "raw_venteny.csv",
                     "raw_gajigesa.csv", "raw_paywatch.csv", "raw_ayokasbon.csv"],
    })
    st.dataframe(df_src, use_container_width=True, hide_index=True)

# =============================== HALAMAN 3 ==================================
elif halaman == "Distribusi Sentimen":
    st.markdown('<div class="judul-utama">Distribusi Sentimen</div>',
                unsafe_allow_html=True)

    seksi("Distribusi Global (2.266 baris ACSC)")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        fig = go.Figure(go.Pie(
            labels=list(DIST_GLOBAL.keys()), values=list(DIST_GLOBAL.values()),
            marker=dict(colors=[WARNA[k] for k in DIST_GLOBAL]), hole=0.45))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        seksi_kecil = st.container()
        df_a = pd.DataFrame(DIST_ASPEK).T[["Positif", "Negatif", "Netral"]]
        fig2 = go.Figure()
        for lab in LABEL:
            fig2.add_bar(name=lab, x=df_a.index, y=df_a[lab],
                         marker_color=WARNA[lab])
        fig2.update_layout(barmode="stack", height=300,
                           margin=dict(l=10, r=10, t=10, b=10),
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Kemudahan Penggunaan didominasi sentimen positif (78 persen), "
        "sedangkan Keandalan Sistem didominasi sentimen negatif (81 persen). "
        "Keluhan utama pengguna berpusat pada stabilitas teknis, biaya, dan "
        "kecepatan pencairan."
    )

    seksi("Sentimen Positif per Platform dengan Interval Kepercayaan Wilson 95 persen")
    fig3 = go.Figure()
    for _, r in PLATFORM.iterrows():
        warna = "#dc2626" if r["n"] < 100 else "#2563eb"
        fig3.add_trace(go.Scatter(
            x=[r["PersenPositif"]], y=[r["Platform"]],
            error_x=dict(type="data", array=[r["Margin"]], color=warna),
            mode="markers", marker=dict(size=10, color=warna),
            name=r["Platform"], showlegend=False))
    fig3.update_layout(height=330, xaxis_title="Persen sentimen positif (baris ACSC)",
                       xaxis=dict(range=[0, 100]),
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "AyoKasbon (n=35) memiliki margin kesalahan +/- 15,0 poin persentase "
        "(ditandai merah), hampir empat kali lipat margin VENTENY (+/- 3,8 pada "
        "n=588), sehingga perbandingan langsung dengan platform besar tidak "
        "setara secara statistik."
    )

# =============================== HALAMAN 4 ==================================
elif halaman == "Performa Model":
    st.markdown('<div class="judul-utama">Performa Model</div>',
                unsafe_allow_html=True)

    seksi("Ringkasan Kontekstualisasi (data uji identik, N = 454)")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kartu("0,8762 +/- 0,0137", "IndoBERT Macro-F1",
                      "rata-rata 5 seed (klaim utama)", "#2563eb"), unsafe_allow_html=True)
    c2.markdown(kartu("0,8843", "IndoBERT (checkpoint terbaik)",
                      "Accuracy 0,9075", "#0891b2"), unsafe_allow_html=True)
    c3.markdown(kartu("0,8454", "TF-IDF + SVM", "Accuracy 0,8789", "#6b7280"),
                unsafe_allow_html=True)
    c4.markdown(kartu("+0,0308", "Uplift (rata-rata)",
                      "+0,0389 pada checkpoint terbaik", "#16a34a"), unsafe_allow_html=True)
    st.caption(
        "Baseline berperan sebagai lower-bound konvensional untuk mengukur "
        "uplift arsitektur kontekstual. IndoBERT dilatih pada 1.627 baris, "
        "baseline pada 1.812 baris; keunggulan dicapai meskipun data latih "
        "lebih sedikit."
    )

    seksi("Laporan Klasifikasi per Kelas")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**IndoBERT (checkpoint terbaik)**")
        df_b = pd.DataFrame(INDOBERT["per_kelas"]).T
        df_b.columns = ["Precision", "Recall", "F1-Score", "Support"]
        st.dataframe(df_b.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}",
                                        "F1-Score": "{:.4f}", "Support": "{:.0f}"}),
                     use_container_width=True)
    with c2:
        st.markdown("**TF-IDF + SVM**")
        df_s = pd.DataFrame(SVM["per_kelas"]).T
        df_s.columns = ["Precision", "Recall", "F1-Score", "Support"]
        st.dataframe(df_s.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}",
                                        "F1-Score": "{:.4f}", "Support": "{:.0f}"}),
                     use_container_width=True)
    st.caption(
        "Peningkatan terbesar IndoBERT terjadi pada kelas netral "
        "(F1 0,8333 berbanding 0,7727), kelas paling ambigu yang paling "
        "membutuhkan pemahaman konteks. Kelas netral hanya 23 sampel sehingga "
        "estimasinya sensitif; inilah alasan pelaporan multi-seed."
    )

    seksi("Stabilitas Pelatihan Multi-Seed (Val Macro-F1 per Epoch, 5 seed)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=STABILITAS["Epoch"], y=STABILITAS["Mean"],
        error_y=dict(type="data", array=STABILITAS["Std"]),
        mode="lines+markers", marker=dict(size=9, color="#2563eb"),
        name="Val Macro-F1"))
    fig.update_layout(height=300, xaxis_title="Epoch",
                      yaxis_title="Val Macro-F1 (mean +/- std)",
                      xaxis=dict(dtick=1),
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Interval antar epoch saling tumpang tindih; epoch 1 (0,9137 +/- "
        "0,0083) paling stabil. Pemilihan checkpoint epoch awal terjustifikasi "
        "secara empiris dan bukan artefak satu kali pelatihan. Kriteria "
        "checkpoint adalah Val Weighted-F1 karena data validasi hanya memuat "
        "11 sampel netral dari 185."
    )

    seksi("Macro-F1 per Kategori Aspek")
    fig2 = go.Figure()
    fig2.add_bar(name="IndoBERT", x=F1_PER_ASPEK["Aspek"], y=F1_PER_ASPEK["IndoBERT"],
                 marker_color="#2563eb")
    fig2.add_bar(name="SVM", x=F1_PER_ASPEK["Aspek"], y=F1_PER_ASPEK["SVM"],
                 marker_color="#9ca3af")
    fig2.update_layout(barmode="group", height=320,
                       yaxis=dict(range=[0, 1]),
                       margin=dict(l=10, r=10, t=10, b=10),
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "IndoBERT unggul terbesar pada Customer Service (+0,102) dan Kecepatan "
        "Pencairan (+0,066), aspek yang penilaiannya bergantung pada konteks. "
        "Pada Keandalan Sistem baseline sedikit lebih unggul karena dominasi "
        "sentimen negatif 81 persen membuat strategi base-rate cukup efektif."
    )

# =============================== HALAMAN 5 ==================================
elif halaman == "Analisis Ulasan Baru":
    st.markdown('<div class="judul-utama">Analisis Ulasan Baru</div>',
                unsafe_allow_html=True)
    st.caption(
        "Teks masukan melewati praproses yang sama dengan pipeline pelatihan "
        "sebelum diinferensikan. Relevansi aspek ditentukan lebih dulu secara "
        "heuristik leksikon; sentimen hanya diprediksi pada aspek yang relevan "
        "(mode prediksi seluruh aspek tersedia sebagai pilihan)."
    )

    if "model_siap" not in st.session_state:
        st.session_state.model_siap = False
    if "riwayat" not in st.session_state:
        st.session_state.riwayat = []

    seksi("Pemuatan Model dan Kamus")
    with st.expander("Pengaturan", expanded=not st.session_state.model_siap):
        path_model = st.text_input("Path checkpoint model (best_model_indobert_revisi.pt)",
                                   value="best_model_indobert_revisi.pt")
        path_kamus = st.text_input("Path kamus normalisasi (kamuskatabaku.xlsx)",
                                   value="kamuskatabaku.xlsx")
        if st.button("Muat model", type="primary", use_container_width=True):
            import os
            if not os.path.exists(path_model):
                st.error(f"File model tidak ditemukan: {path_model}")
            else:
                import torch
                from transformers import BertTokenizer, BertForSequenceClassification

                @st.cache_resource(show_spinner=True)
                def _muat(path):
                    tok = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
                    mdl = BertForSequenceClassification.from_pretrained(
                        "indobenchmark/indobert-base-p1", num_labels=3)
                    s = torch.load(path, map_location="cpu")
                    if isinstance(s, dict) and "model_state_dict" in s:
                        s = s["model_state_dict"]
                    if isinstance(s, dict) and any(k.startswith("module.") for k in s):
                        s = {k.replace("module.", ""): v for k, v in s.items()}
                    mdl.load_state_dict(s)
                    mdl.eval()
                    return tok, mdl

                tok, mdl = _muat(path_model)
                st.session_state.tokenizer = tok
                st.session_state.model = mdl
                st.session_state.model_siap = True
                st.rerun()

    kamus, kamus_lengkap = muat_kamus(
        st.session_state.get("path_kamus_aktif", "kamuskatabaku.xlsx"))
    if not kamus_lengkap:
        st.info(
            "Kamus 4.975 entri tidak ditemukan pada path yang diberikan; "
            "digunakan kamus bawaan ringkas. Untuk kesetaraan penuh dengan "
            "pipeline pelatihan, sediakan file kamuskatabaku.xlsx."
        )

    if st.session_state.model_siap:
        st.success("Model IndoBERT siap digunakan (checkpoint terbaik, "
                   "kriteria pemilihan Val Weighted-F1).")

        def prediksi(tok, mdl, teks_bersih, aspek):
            import torch
            enc = tok(teks_bersih, aspek, max_length=128, truncation=True,
                      padding="max_length", return_tensors="pt")
            with torch.no_grad():
                logits = mdl(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1).numpy().tolist()
            pred = int(np.argmax(probs))
            return pred, probs[pred], probs

        seksi("Masukkan Ulasan")
        ulasan = st.text_area(
            "Teks ulasan", height=110,
            placeholder="Contoh: pencairan cepat sekali tapi biaya adminnya mahal")
        mode_semua = st.checkbox(
            "Prediksi seluruh lima aspek tanpa penyaringan relevansi "
            "(mode evaluasi penuh)", value=False)

        if st.button("Analisis sentimen", type="primary",
                     disabled=(not ulasan.strip()), use_container_width=True):
            teks_bersih = praproses_teks(ulasan, kamus)

            seksi("Hasil Praproses")
            c1, c2 = st.columns(2)
            c1.markdown("**Teks asli**")
            c1.code(ulasan.strip(), language=None)
            c2.markdown("**Teks setelah praproses (input model)**")
            c2.code(teks_bersih, language=None)

            aspek_relevan = deteksi_aspek(teks_bersih)
            if mode_semua:
                target = ASPEK
            else:
                target = aspek_relevan

            seksi("Deteksi Relevansi Aspek")
            baris_lencana = ""
            for a in ASPEK:
                kelas = "l-pos" if a in aspek_relevan else "l-abu"
                baris_lencana += f'<span class="lencana {kelas}">{a}</span> '
            st.markdown(baris_lencana, unsafe_allow_html=True)
            if not aspek_relevan:
                st.warning(
                    "Tidak ada aspek yang terdeteksi relevan dari leksikon. "
                    "Ulasan kemungkinan bersifat umum. Aktifkan mode evaluasi "
                    "penuh untuk tetap memprediksi seluruh aspek.")
            if not target:
                st.stop()

            seksi("Hasil Klasifikasi Sentimen per Aspek")
            hasil = []
            for a in target:
                idx = ASPEK.index(a)
                p, c, probs = prediksi(
                    st.session_state.tokenizer, st.session_state.model,
                    teks_bersih, ASPEK_INPUT[idx])
                hasil.append((a, p, c, probs))

            for a, p, c, probs in hasil:
                kelas_l = ["l-pos", "l-neg", "l-net"][p]
                st.markdown(
                    f'<div class="kotak-hasil"><b>{a}</b> &nbsp; '
                    f'<span class="lencana {kelas_l}">{LABEL[p]}</span> '
                    f'<span style="font-size:12px;color:#374151;"> '
                    f'keyakinan {c:.1%}</span><br>'
                    f'<span style="font-size:11px;color:#9ca3af;">'
                    f'positif: {probs[0]:.2f} | negatif: {probs[1]:.2f} | '
                    f'netral: {probs[2]:.2f}</span></div>',
                    unsafe_allow_html=True)

            st.caption(
                "Format input model: [CLS] teks_bersih [SEP] kategori_aspek "
                "[SEP]. Prediksi bersifat probabilistik; nilai keyakinan "
                "adalah probabilitas softmax kelas terpilih."
            )
            st.session_state.riwayat.append(
                {"teks": ulasan.strip(), "bersih": teks_bersih,
                 "target": [h[0] for h in hasil]})

        if st.session_state.riwayat:
            with st.expander(f"Riwayat analisis ({len(st.session_state.riwayat)})"):
                for i, r in enumerate(reversed(st.session_state.riwayat[-10:]), 1):
                    st.markdown(f"{i}. {r['teks']} — aspek: "
                                f"{', '.join(r['target']) if r['target'] else 'tidak ada'}")
    else:
        st.info("Muat model terlebih dahulu melalui panel Pengaturan di atas.")
