# ============================================================
# evaluasi.py - DATA BANYAK: baca CSV, petakan label, hitung metrik
# (hanya pandas dan numpy, tanpa torch, jadi ringan dimuat)
# ============================================================
import csv
import io

import numpy as np
import pandas as pd

KANDIDAT_TEKS = ("text", "teks", "kalimat", "sentence", "tweet", "komentar", "review")
KANDIDAT_LABEL = ("label", "label_asli", "emotion", "emosi", "kelas", "class")


class CsvTidakValid(ValueError):
    """Masalah pada isi file CSV; pesannya aman dan jelas untuk ditampilkan ke pengguna."""


def baca_csv(berkas):
    """Membaca file CSV unggahan. Pemisah koma, titik koma, atau tab; UTF-8 atau latin-1."""
    data = berkas.getvalue()
    try:
        isi = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        isi = data.decode("latin-1")
    if "\x00" in isi:
        raise CsvTidakValid("Isi file bukan teks biasa (mungkin file Excel atau file biner yang hanya diganti namanya "
                            "menjadi .csv). Simpan ulang sebagai CSV.")
    baris_pertama = isi.splitlines()[0] if isi.strip() else ""
    pemisah = max([",", ";", "\t"], key=baris_pertama.count)                 # tebak dari baris judul kolom

    # Pemeriksaan: baris yang kolomnya LEBIH banyak dari judul (dan kelebihannya tidak kosong) hampir pasti
    # kalimat berkoma tanpa tanda kutip. Tanpa pemeriksaan ini pandas membaca file secara salah dan diam-diam.
    pembaca = csv.reader(io.StringIO(isi), delimiter=pemisah)
    jumlah_kolom = None
    for baris in pembaca:
        if not baris:
            continue
        if jumlah_kolom is None:
            jumlah_kolom = len(baris)
        elif len(baris) > jumlah_kolom and any(x.strip() for x in baris[jumlah_kolom:]):
            raise CsvTidakValid(f"Baris {pembaca.line_num} punya {len(baris)} kolom, padahal judul kolom hanya {jumlah_kolom}. "
                                "Biasanya penyebabnya kalimat yang mengandung koma tidak diapit tanda kutip (\"...\").")
    return pd.read_csv(io.StringIO(isi), sep=pemisah, dtype=str, keep_default_na=False, index_col=False)


def tebak_kolom(daftar_kolom, kandidat):
    """Mengembalikan nama kolom pertama yang namanya cocok dengan kandidat, atau None."""
    for k in daftar_kolom:
        if k.strip().lower() in kandidat:
            return k
    return None


def petakan_label(seri, id2label):
    """Label unggahan -> nama emosi. Menerima nama (sadness, ...) atau angka (0-5). Tak dikenali -> None."""
    nama = set(id2label.values())

    def satu(v):
        v = str(v).strip().lower()
        if v in nama:
            return v
        try:
            f = float(v)
        except ValueError:
            return None
        return id2label.get(int(f)) if f.is_integer() else None

    return seri.map(satu)


def hitung_metrik(y_benar, y_prediksi):
    """Akurasi dan macro-F1. Macro-F1 dirata-ratakan atas kelas yang muncul di label ATAU prediksi
    (sama seperti sklearn.metrics.f1_score(average='macro'))."""
    yb, yp = pd.Series(list(y_benar)), pd.Series(list(y_prediksi))
    akurasi = float((yb == yp).mean())
    f1_kelas = []
    for k in sorted(set(yb) | set(yp)):
        tp = int(((yb == k) & (yp == k)).sum())
        fp = int(((yb != k) & (yp == k)).sum())
        fn = int(((yb == k) & (yp != k)).sum())
        f1_kelas.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return akurasi, float(np.mean(f1_kelas))