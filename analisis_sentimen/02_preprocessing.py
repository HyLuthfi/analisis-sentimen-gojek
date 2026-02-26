import pandas as pd
import re
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

INPUT = "data/ulasan_gojek_mentah.csv"
OUTPUT = "data/ulasan_gojek_bersih.csv"

df = pd.read_csv(INPUT)
print(f"Data dimuat: {len(df)} baris")

pabrik_sw = StopWordRemoverFactory()
hapus_sw = pabrik_sw.create_stop_word_remover()

pabrik_st = StemmerFactory()
stemmer = pabrik_st.create_stemmer()

kamus_slang = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak",
    "ngga": "tidak", "nggak": "tidak", "g": "tidak",
    "yg": "yang", "yng": "yang", "drpd": "daripada",
    "dgn": "dengan", "dg": "dengan", "sy": "saya",
    "sdh": "sudah", "udh": "sudah", "udah": "sudah",
    "blm": "belum", "blom": "belum", "blum": "belum",
    "dpt": "dapat", "gmn": "bagaimana", "gimana": "bagaimana",
    "hrs": "harus", "krn": "karena", "karna": "karena",
    "tp": "tapi", "tpi": "tapi", "ttg": "tentang",
    "dl": "dulu", "dlu": "dulu", "bnyk": "banyak",
    "byk": "banyak", "msh": "masih", "masi": "masih",
    "bgt": "banget", "banget": "sangat", "aja": "saja",
    "aj": "saja", "jg": "juga", "spt": "seperti",
    "ok": "oke", "lg": "lagi", "klo": "kalau",
    "kalo": "kalau", "bs": "bisa", "bsa": "bisa",
    "dr": "dari", "utk": "untuk", "u": "untuk",
    "tdk": "tidak", "pd": "pada", "km": "kamu",
    "lo": "kamu", "lu": "kamu", "gue": "saya",
    "gw": "saya", "abis": "habis", "lbh": "lebih",
    "lbih": "lebih", "susah": "sulit",
    "aplikasinya": "aplikasi", "appnya": "aplikasi",
}


def normalisasi_slang(teks):
    return " ".join([kamus_slang.get(k, k) for k in teks.split()])


def bersihkan_teks(teks):
    if pd.isna(teks) or str(teks).strip() == "":
        return ""
    teks = str(teks).lower()
    teks = re.sub(r"http\S+|www\.\S+", "", teks)
    teks = re.sub(r"@\w+|#\w+", "", teks)
    teks = teks.encode("ascii", "ignore").decode("ascii")
    teks = re.sub(r"\d+", "", teks)
    teks = re.sub(r"[^\w\s]", " ", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    teks = normalisasi_slang(teks)
    teks = hapus_sw.remove(teks)
    teks = stemmer.stem(teks)
    return teks.strip()


print("Memproses teks...")
total = len(df)
hasil_bersih = []

for i, baris in df.iterrows():
    hasil_bersih.append(bersihkan_teks(baris["isi_ulasan"]))
    if (i + 1) % 200 == 0:
        print(f"  {i + 1}/{total} selesai...")

df["teks_bersih"] = hasil_bersih
df = df[df["teks_bersih"].str.strip() != ""]
df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"Selesai. {len(df)} baris disimpan ke {OUTPUT}")
