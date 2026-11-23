# ============================================================
# inferensi.py - LOGIKA PREDIKSI (tanpa kode Streamlit)
# ============================================================
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from evaluasi import petakan_label

FOLDER_MODEL = Path(__file__).parent / "model"    # folder "model" di sebelah file ini (dipakai di laptop)
REPO_MODEL = "moddzierr/emotion-skripsi"          # model di Hugging Face (dipakai bila folder model/ tidak ada, mis. di server)
MAX_LENGTH = 57                                    # sama dengan CELL B9 di notebook training
UKURAN_BATCH = 32                                  # jumlah kalimat per kelompok untuk data banyak


# ------------------------------------------------------------
# 4a. MEMUAT MODEL
# ------------------------------------------------------------
def muat_model():
    """Memuat tokenizer dan model. Memakai folder model/ bila ada; kalau tidak, mengunduh dari Hugging Face.
    Mengembalikan (tokenizer, model)."""
    sumber = str(FOLDER_MODEL) if (FOLDER_MODEL / "config.json").exists() else REPO_MODEL
    tokenizer = AutoTokenizer.from_pretrained(sumber)
    model = AutoModelForSequenceClassification.from_pretrained(sumber)
    model.eval()                                   # mode prediksi (dropout dimatikan)
    return tokenizer, model


# ------------------------------------------------------------
# 4b. PEMBERSIHAN TEKS (salinan persis CELL B4 notebook training)
# ------------------------------------------------------------
ATURAN = [
    ("1 casefolding (huruf kecil)",      lambda s: s.lower()),
    ("2 hapus URL",                      lambda s: re.sub(r"https?://\S+|www\.\S+", " ", s)),
    ("3 hapus mention (@nama)",          lambda s: re.sub(r"@\w+", " ", s)),
    ("4 hapus tanda # (kata tetap)",     lambda s: s.replace("#", "")),
    ("5 underscore jadi spasi",          lambda s: s.replace("_", " ")),
    ("6 hapus sisa link HTML / elipsis", lambda s: re.sub(r"\b(?:a href|href|https?|www|hellip|img src)\b", " ", s)),
    ("7 rapikan spasi",                  lambda s: re.sub(r"\s+", " ", s).strip()),
]

def clean_text(teks):
    """Menerapkan semua aturan ke SATU teks."""
    for _, fungsi in ATURAN:
        teks = fungsi(teks)
    return teks


def kalimat_valid(bersih):
    """VALIDASI MASUKAN (khusus aplikasi, BUKAN bagian pembersihan di eksperimen):
    kalimat layak dianalisis bila memuat minimal satu kata berhuruf (2 huruf a-z atau lebih berurutan).
    Contoh tidak layak: kosong, '123', '!!!???', '123#' (angka dan simbol saja)."""
    return re.search(r"[a-z]{2,}", bersih) is not None


# ------------------------------------------------------------
# 4c. PREDIKSI
# ------------------------------------------------------------
def klasifikasi(bersih, tokenizer, model):
    """Memprediksi SATU teks yang SUDAH bersih. Mengembalikan dict hasil prediksi."""
    jumlah_token = len(tokenizer(bersih, truncation=True, max_length=MAX_LENGTH + 1)["input_ids"])   # > MAX_LENGTH berarti terpotong                          # panjang asli (dengan token khusus)
    enc = tokenizer(bersih, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():                                                       # tanpa menyimpan gradien
        logits = model(**enc).logits[0]                                         # 6 skor mentah
    prob = torch.softmax(logits, dim=-1).tolist()                               # skor -> probabilitas (jumlah = 1)

    id2label = {int(k): v for k, v in model.config.id2label.items()}            # nomor -> nama emosi
    hasil = {id2label[i]: p for i, p in enumerate(prob)}
    label = max(hasil, key=hasil.get)
    return {"teks_bersih": bersih, "label": label, "confidence": hasil[label],
            "prob": hasil, "terpotong": jumlah_token > MAX_LENGTH}


def prediksi(teks, tokenizer, model):
    """Teks mentah dari pengguna -> bersihkan -> validasi -> klasifikasi. None bila tidak layak."""
    bersih = clean_text(teks)
    if not kalimat_valid(bersih):
        return None
    return klasifikasi(bersih, tokenizer, model)


# ------------------------------------------------------------
# 5. DATA BANYAK
# ------------------------------------------------------------
def klasifikasi_batch(daftar_bersih, tokenizer, model, ukuran_batch=UKURAN_BATCH, progress=None):
    """Memprediksi banyak teks yang SUDAH bersih, per kelompok (lebih cepat daripada satu per satu).
    Mengembalikan daftar dict {label, confidence, prob} dengan urutan sama seperti masukan."""
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    urutan = sorted(range(len(daftar_bersih)), key=lambda i: len(daftar_bersih[i]))   # teks pendek dikelompokkan
    hasil = [None] * len(daftar_bersih)
    for mulai in range(0, len(urutan), ukuran_batch):
        idx = urutan[mulai:mulai + ukuran_batch]
        enc = tokenizer([daftar_bersih[i] for i in idx], truncation=True, max_length=MAX_LENGTH,
                        padding=True, return_tensors="pt")
        with torch.no_grad():
            prob = torch.softmax(model(**enc).logits, dim=-1).tolist()
        for i, p in zip(idx, prob):
            d = {id2label[k]: v for k, v in enumerate(p)}
            lab = max(d, key=d.get)
            hasil[i] = {"label": lab, "confidence": d[lab], "prob": d}
        if progress:
            progress(min(mulai + ukuran_batch, len(urutan)) / len(urutan))
    return hasil


def proses_data_banyak(df, kol_teks, kol_label, tokenizer, model, progress=None):
    """Memprediksi satu tabel kalimat. kol_label boleh None. Mengembalikan tabel hasil (DataFrame)."""
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    asli = df[kol_teks].astype(str).tolist()
    bersih = [clean_text(t) for t in asli]
    valid = [kalimat_valid(b) for b in bersih]
    idx_valid = [i for i, v in enumerate(valid) if v]
    hasil_valid = klasifikasi_batch([bersih[i] for i in idx_valid], tokenizer, model, progress=progress)

    prediksi_, conf = [None] * len(df), [np.nan] * len(df)
    for i, h in zip(idx_valid, hasil_valid):
        prediksi_[i], conf[i] = h["label"], h["confidence"]

    out = pd.DataFrame({"No": range(1, len(df) + 1), "Kalimat asli": asli, "Kalimat bersih": bersih,
                        "Status": ["Valid" if v else "Tidak valid" for v in valid],
                        "Prediksi": prediksi_, "Confidence": conf})
    if kol_label:
        out["Label sebenarnya"] = petakan_label(df[kol_label], id2label).values
        ada = out["Prediksi"].notna() & out["Label sebenarnya"].notna()
        out["Hasil"] = np.where(~ada, "-", np.where(out["Prediksi"] == out["Label sebenarnya"], "Benar", "Salah"))
    return out


# ------------------------------------------------------------
# UJI CEPAT: jalankan  python inferensi.py
# ------------------------------------------------------------
if __name__ == "__main__":
    import time
    t0 = time.time()
    tokenizer, model = muat_model()
    print(f"Model dimuat dalam {time.time() - t0:.1f} detik")
    print("id2label:", model.config.id2label)
    print()

    contoh = [
        "i feel so happy today",
        "I Feel #Happy @budi https://t.co/x so_much  today",
        "the meeting starts at three pm",
        "i feel " + "very " * 80 + "tired",                  # teks panjang: harus terpotong
        "https://t.co/abc @budi",                              # kosong setelah dibersihkan
        "123##",                                               # hanya angka dan simbol: tidak layak
    ]
    for t in contoh:
        h = prediksi(t, tokenizer, model)
        if h is None:
            print("(tidak layak) |", t[:60])
        else:
            print(f"{h['label']:9s} {h['confidence']:.3f} | terpotong={h['terpotong']} | {h['teks_bersih'][:60]}")