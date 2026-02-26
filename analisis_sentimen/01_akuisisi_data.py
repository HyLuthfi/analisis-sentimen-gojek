import pandas as pd
from google_play_scraper import reviews, Sort
import time

APP_ID = "com.gojek.app"
TARGET = 2000
OUTPUT = "data/ulasan_gojek_mentah.csv"

semua_ulasan = []
token = None

try:
    while len(semua_ulasan) < TARGET:
        sisa = TARGET - len(semua_ulasan)
        batch = min(200, sisa)

        hasil, token = reviews(
            APP_ID,
            lang="id",
            country="id",
            sort=Sort.NEWEST,
            count=batch,
            continuation_token=token
        )

        if not hasil:
            break

        semua_ulasan.extend(hasil)
        print(f"Mengambil ulasan: {len(semua_ulasan)}/{TARGET}")

        if token is None:
            break

        time.sleep(1)

except Exception as e:
    print(f"Error: {e}")

daftar_data = []
for ulasan in semua_ulasan:
    daftar_data.append({
        "id_ulasan": ulasan.get("reviewId", ""),
        "nama_pengguna": ulasan.get("userName", ""),
        "isi_ulasan": ulasan.get("content", ""),
        "bintang": ulasan.get("score", 0),
        "tanggal_ulasan": ulasan.get("at", ""),
        "jumlah_like": ulasan.get("thumbsUpCount", 0),
    })

df = pd.DataFrame(daftar_data)

def tentukan_sentimen(bintang):
    if bintang >= 4:
        return "Positif"
    elif bintang == 3:
        return "Netral"
    else:
        return "Negatif"

df["sentimen"] = df["bintang"].apply(tentukan_sentimen)
df = df[df["isi_ulasan"].str.strip() != ""]
df = df.dropna(subset=["isi_ulasan"])

df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"Selesai. Total: {len(df)} ulasan disimpan ke {OUTPUT}")
print(df["sentimen"].value_counts().to_string())
