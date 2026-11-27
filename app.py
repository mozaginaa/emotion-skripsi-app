# ============================================================
# app.py - TAMPILAN APLIKASI (Streamlit)
# ============================================================
import altair as alt
import pandas as pd
import streamlit as st

from evaluasi import baca_csv, tebak_kolom, hitung_metrik, CsvTidakValid, KANDIDAT_TEKS, KANDIDAT_LABEL

st.set_page_config(page_title="Klasifikasi Emosi RoBERTa", layout="wide",
                   initial_sidebar_state="collapsed")

PRIMER, PINK = "#D96868", "#FFDCDC"
NAMA = {"sadness": "Sedih (sadness)", "joy": "Senang (joy)", "love": "Cinta (love)",
        "anger": "Marah (anger)", "fear": "Takut (fear)", "surprise": "Terkejut (surprise)"}
KIRI, KANAN = ["sadness", "joy", "love"], ["anger", "fear", "surprise"]          # grafik dibagi 3 + 3
CONTOH = [("Sadness",    "i feel lonely and sad since she left"),
          ("Joy",   "i feel so happy and thankful today"),
          ("Love",    "i love you in every situation"),
          ("Anger",    "i am so angry that he lied to me again"),
          ("Fear",    "i am terrified of walking home alone at night"),
          ("Surprise", "i feel shocked and amazed that i actually won the prize")]
PESAN_BATAS = ("⚠️Hanya menggunakan kalimat berbahasa Inggris dengan setiap kalimat diklasifikasikan ke dalam "
               "satu dari enam kategori emosi, termasuk kalimat netral dan kalimat yang mengandung lebih dari "
               "satu emosi⚠️")
MAKS_BARIS = 5000                                                                 # batas baris per unggahan


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def ambil_model():
    from inferensi import muat_model          
    return muat_model()


# ------------------------------------------------------------
# FUNGSI BANTU
# ------------------------------------------------------------
def persen(p):
    """Probabilitas -> teks persen berkoma, tanpa pernah menampilkan '100%'."""
    if p >= 0.9995: return "> 99,9%"
    if p < 0.0005:  return "< 0,1%"
    return f"{p * 100:.1f}%".replace(".", ",")

def angka(x, desimal=4):
    return f"{x:.{desimal}f}".replace(".", ",")

def isi_contoh(teks):
    st.session_state["teks_input"] = teks        

def gambar_grafik(prob, kelompok, label_utama):
    df = pd.DataFrame({"emosi": [NAMA[k] for k in kelompok], "probabilitas": [prob[k] for k in kelompok],
                       "utama": [k == label_utama for k in kelompok], "persen": [persen(prob[k]) for k in kelompok]})
    dasar = alt.Chart(df).encode(
        y=alt.Y("emosi:N", sort=[NAMA[k] for k in kelompok], title=None,
                axis=alt.Axis(ticks=False, domain=False, labelFontSize=18, labelColor="#1F2937", labelLimit=300, labelPadding=8)),
        x=alt.X("probabilitas:Q", scale=alt.Scale(domain=[0, 1.5]), axis=None))
    batang = dasar.mark_bar(cornerRadiusEnd=6, size=28, stroke=PRIMER, strokeWidth=1).encode(
        color=alt.condition(alt.datum.utama, alt.value(PRIMER), alt.value(PINK)))
    teks = dasar.mark_text(align="left", dx=6, fontSize=18, color="#1F2937").encode(text="persen:N")
    return (batang + teks).properties(height=46 * len(kelompok)).configure_view(stroke=None)


# ------------------------------------------------------------
# GAYA TAMBAHAN
# ------------------------------------------------------------
st.markdown("""
<style>
footer {visibility: hidden;}
[data-testid="stMainBlockContainer"] {max-width: 1500px; padding-top: 1.5rem; padding-bottom: 1.5rem;}
[data-testid="stMainBlockContainer"] {max-width: 1500px; padding-top: 1.5rem; padding-bottom: 1.5rem;}
/* pengunggah file: sembunyikan tombol "+" (cukup satu file) */
[data-testid="stFileUploaderDropzone"] button[aria-label="Add files"] {display: none;}
/* tabel: sembunyikan ikon "Download as CSV" bawaan Streamlit */
[data-testid="stElementToolbarButton"]:has(button[aria-label="Download as CSV"]) {display: none;}
/* tombol Prediksi: besar, warna D96868 */
button[data-testid="stBaseButton-primary"] {min-height: 4rem; border: none;}
button[data-testid="stBaseButton-primary"] p {font-size: 1.5rem; font-weight: 700; color: #FFFFFF;}
button[data-testid="stBaseButton-primary"]:hover {background: #C95A5A;}
@media (max-width: 640px) {h1 {font-size: 1.7rem !important;}}      /* judul lebih kecil di layar HP */
/* kotak centang: garisnya dibuat jelas */
[data-testid="stCheckbox"] label > div:first-of-type {border: 2px solid #D96868 !important;}
/* label kotak yang dikunci (disabled) tetap gelap dan terbaca */
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {color: #1F2937 !important; opacity: 1 !important;}
/* tombol contoh: pink FFDCDC */
button[data-testid="stBaseButton-secondary"] {background: #FFDCDC; border: 1px solid #FFDCDC; color: #1F2937;}
button[data-testid="stBaseButton-secondary"]:hover {background: #FFC9C9; border-color: #D96868; color: #1F2937;}
/* kotak yang tidak bisa diedit tetap terbaca jelas */
textarea:disabled {color: #1F2937 !important; -webkit-text-fill-color: #1F2937 !important; opacity: 1 !important;}
/* kartu hasil */
.kartu {background: #FFDCDC; border-radius: 24px; padding: 1.4rem 1.8rem; margin: 1rem 0 1.2rem 0;}
.kartu-judul {font-size: .8rem; letter-spacing: .1em; text-transform: uppercase; color: #7A2E2E; font-weight: 700;}
.kartu-emosi {font-size: 2.6rem; font-weight: 800; color: #8C2F2F; line-height: 1.2; margin: .2rem 0;}
.kartu-conf {color: #4B2A2A; font-size: 1.05rem;}
/* penanda area hasil sebelum ada prediksi */
.kosong {border: 2px dashed #D96868; border-radius: 24px; padding: 4rem 1.5rem; text-align: center; color: #7A2E2E;}
/* kotak peringatan merah */
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# POP-UP PERINGATAN
# ------------------------------------------------------------
@st.dialog("Peringatan")
def popup_tidak_valid():
    st.write("Harap memasukkan kalimat dengan benar")


# ------------------------------------------------------------
# TAMPILAN HASIL
# ------------------------------------------------------------
def tampil_hasil_satu(hasil):
    st.markdown(
        f"""<div class="kartu"><div class="kartu-judul">Emosi terprediksi</div>
        <div class="kartu-emosi">{NAMA[hasil['label']]}</div>
        <div class="kartu-conf">Keyakinan model (confidence): <b>{persen(hasil['confidence'])}</b></div></div>""",
        unsafe_allow_html=True)
    if hasil["terpotong"]:
        from inferensi import MAX_LENGTH
        st.caption(f"Kalimat melebihi batas {MAX_LENGTH} token, sehingga bagian akhirnya tidak ikut dianalisis.")
    st.markdown("**Probabilitas tiap emosi**")
    st.altair_chart(gambar_grafik(hasil["prob"], KIRI + KANAN, hasil["label"]), width="stretch")


def tampil_hasil_banyak(out):
    ada_label = "Hasil" in out.columns
    tidak_valid = int((out["Status"] == "Tidak valid").sum())
    m = st.columns(5 if ada_label else 3)
    m[0].metric("Jumlah baris", f"{len(out):,}".replace(",", "."))
    m[1].metric("Dianalisis", f"{len(out) - tidak_valid:,}".replace(",", "."))
    m[2].metric("Tidak valid", f"{tidak_valid:,}".replace(",", "."))
    if ada_label:
        dinilai = out[out["Hasil"] != "-"]
        if len(dinilai):
            akurasi, makro_f1 = hitung_metrik(dinilai["Label sebenarnya"], dinilai["Prediksi"])
            m[3].metric("Akurasi", angka(akurasi))
            m[4].metric("Macro-F1", angka(makro_f1))
            st.caption(f"Akurasi dan macro-F1 dihitung dari {len(dinilai):,} baris yang valid "
                       f"({int((dinilai['Hasil'] == 'Salah').sum()):,} salah).".replace(",", "."))
        else:
            st.warning("Tidak ada baris yang bisa dinilai: tidak ada baris yang kalimatnya valid sekaligus berlabel dikenali. "
                       "Label harus berupa nama emosi (sadness, joy, love, anger, fear, surprise) atau angka 0-5.")
    if ada_label:
        tak_dikenali = int(((out["Status"] == "Valid") & out["Label sebenarnya"].isna()).sum())
        if tak_dikenali:
            st.caption(f"{tak_dikenali} baris valid tidak ikut dinilai karena label aslinya tidak dikenali.")

    if tidak_valid:
        st.caption(f"{tidak_valid} baris tidak valid (kosong, atau hanya angka dan simbol setelah dibersihkan) tidak diprediksi.")

    tabel = out
    if ada_label and st.checkbox("Tampilkan hanya prediksi yang salah"):
        tabel = out[out["Hasil"] == "Salah"]
    # tabel layar diringkas supaya muat (kalimat asli dan status lengkap ada di file CSV unduhan)
    tampil = tabel.drop(columns=["Kalimat asli"]).copy()
    tampil["Prediksi"] = tampil["Prediksi"].where(tampil["Status"] == "Valid", "(tidak valid)")
    tampil = tampil.drop(columns=["Status"])
    gaya = tampil.style.format({"Confidence": lambda v: "-" if pd.isna(v) else f"{v:.3f}".replace(".", ",")})
    if ada_label:
        gaya = gaya.map(lambda v: "color: #B42318; font-weight: 700" if v == "Salah"
                        else ("color: #067647; font-weight: 700" if v == "Benar" else ""), subset=["Hasil"])
    st.dataframe(gaya, hide_index=True, width="stretch", height=480)


# ------------------------------------------------------------
# HALAMAN UTAMA
# ------------------------------------------------------------
st.markdown("# KLASIFIKASI EMOSI <span style='white-space:nowrap'>MULTI-KELAS</span> MENGGUNAKAN ROBERTA",
            unsafe_allow_html=True)       # "MULTI-KELAS" dijaga tetap satu baris
st.caption("Model memprediksi satu dari enam emosi: sadness, joy, love, anger, fear, surprise.  \n" + PESAN_BATAS)

kolom_utama = st.container()  # wadah satu kolom penuh (menggantikan pembagian 3 : 1 yang lama)

with kolom_utama:
    with st.spinner("Memuat model, mohon tunggu"):
        tokenizer, model = ambil_model()
    from inferensi import MAX_LENGTH, clean_text, kalimat_valid, klasifikasi, proses_data_banyak    # sudah termuat, jadi cepat

    tab_satu, tab_banyak = st.tabs(["Tulis Kalimat", "Unggah Data"])

    # ---------------- TAB 1: SATU KALIMAT (kiri: masukan, kanan: hasil) ----------------
    with tab_satu:
        kol_masuk, kol_hasil = st.columns(2, gap="large")
        with kol_masuk:
            teks = st.text_area("Masukkan satu kalimat berbahasa Inggris", key="teks_input", height=130,
                                placeholder="Contoh: i feel so happy today")
            bersih = clean_text(teks)
            st.session_state["teks_bersih_tampil"] = bersih
            st.text_area("Kalimat setelah dibersihkan", key="teks_bersih_tampil", height=90, disabled=True)
            if kalimat_valid(bersih) and len(tokenizer(bersih, truncation=True, max_length=MAX_LENGTH + 1)["input_ids"]) > MAX_LENGTH:
                st.caption(f"Kalimat ini lebih panjang dari {MAX_LENGTH} token. Model hanya membaca {MAX_LENGTH} token pertama "
                           "(kurang lebih 40 kata); bagian sesudahnya tidak ikut dianalisis.")

            st.caption("Atau coba salah satu contoh:")
            for baris in (CONTOH[:3], CONTOH[3:]):
                kolom = st.columns(3)
                for kol, (nama_tombol, kalimat) in zip(kolom, baris):
                    kol.button(nama_tombol, on_click=isi_contoh, args=(kalimat,), width="stretch")

            klik = st.button("Prediksi", type="primary", width="stretch", key="btn_satu")

        with kol_hasil:
            if klik and not kalimat_valid(bersih):
                popup_tidak_valid()
            elif klik:
                tampil_hasil_satu(klasifikasi(bersih, tokenizer, model))
            else:
                st.markdown('<div class="kosong">Hasil prediksi akan tampil di sini.</div>', unsafe_allow_html=True)

    # ---------------- TAB 2: DATA BANYAK ----------------
    with tab_banyak:
        st.markdown("Unggah file **CSV** berisi kalimat berbahasa Inggris")
        berkas = st.file_uploader("Pilih file CSV", type=["csv"])
        if berkas is not None:
            try:
                df = baca_csv(berkas)
            except CsvTidakValid as e:
                df = None
                st.error(str(e))
            except Exception:
                df = None
                st.error("File tidak dapat dibaca sebagai CSV. Pastikan baris pertama berisi nama kolom.")
            if df is not None and len(df) == 0:
                df = None
                st.error("File tidak berisi baris data.")
            if df is not None:
                if len(df) > MAKS_BARIS:
                    st.warning(f"File berisi {len(df):,} baris. Hanya {MAKS_BARIS:,} baris pertama yang diproses "
                               "(batas aplikasi agar proses tidak terlalu lama).".replace(",", "."))
                    df = df.head(MAKS_BARIS)
                daftar = list(df.columns)
                # tampilan saja: kolom bernama "text" ditulis "teks" (nilai aslinya tetap "text")
                nama_tampil = lambda k: "teks" if (k == "text" and "teks" not in daftar) else k
                mirip_kalimat = [k for k in daftar if len(k.split()) >= 4]
                if mirip_kalimat:
                    st.warning(f"Nama kolom \"{mirip_kalimat[0][:50]}\" terlihat seperti kalimat. Jika file tidak punya baris judul "
                               "kolom, baris pertama dibaca sebagai judul dan tidak ikut diprediksi. Tambahkan baris judul "
                               "(misalnya teks,label) lalu unggah ulang.")
                tebak_t, tebak_l = tebak_kolom(daftar, KANDIDAT_TEKS), tebak_kolom(daftar, KANDIDAT_LABEL)
                c1, c2 = st.columns(2)
                kol_teks = c1.selectbox("Kolom kalimat", daftar, index=daftar.index(tebak_t) if tebak_t else 0, format_func=nama_tampil)
                pilihan = ["(tidak ada label)"] + daftar
                kol_label = c2.selectbox("Kolom label asli", pilihan, index=pilihan.index(tebak_l) if tebak_l else 0, format_func=nama_tampil)
                st.caption(f"{len(df):,} baris terbaca dan akan diprediksi. Berikut pratinjau 5 baris pertama ".replace(",", "."))
                st.dataframe(df.head(5).rename(columns=nama_tampil), hide_index=True, width="stretch")

                if st.button("Prediksi", type="primary", width="stretch", key="btn_banyak"):
                    bar = st.progress(0.0, text="Memproses...")
                    out = proses_data_banyak(df, kol_teks, None if kol_label == "(tidak ada label)" else kol_label,
                                             tokenizer, model,
                                             progress=lambda f: bar.progress(f, text=f"Memproses... {f * 100:.0f}%"))
                    bar.empty()
                    st.session_state["hasil_banyak"] = ((berkas.file_id, kol_teks, kol_label), out)

                tersimpan = st.session_state.get("hasil_banyak")          # hasil hanya tampil bila file dan kolom pilihan masih sama
                if tersimpan and tersimpan[0] == (berkas.file_id, kol_teks, kol_label):
                    tampil_hasil_banyak(tersimpan[1])
