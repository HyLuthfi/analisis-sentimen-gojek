import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/ulasan_gojek_bersih.csv")
df = df.dropna(subset=["teks_bersih", "sentimen"])
df = df[df["teks_bersih"].str.strip() != ""]
print(f"Data: {len(df)} baris")

distribusi = df["sentimen"].value_counts()
print(distribusi.to_string())

X = df["teks_bersih"]
y = df["sentimen"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vektorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vektorizer.fit_transform(X_train)
X_test_tfidf = vektorizer.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver="lbfgs",
    multi_class="multinomial",
    random_state=42
)
model.fit(X_train_tfidf, y_train)

prediksi = model.predict(X_test_tfidf)
akurasi = accuracy_score(y_test, prediksi)
print(f"\nAkurasi: {akurasi * 100:.2f}%")
print(classification_report(y_test, prediksi, target_names=model.classes_))

laporan = classification_report(y_test, prediksi, target_names=model.classes_, output_dict=True)
pd.DataFrame(laporan).transpose().to_csv("data/laporan_evaluasi.csv", encoding="utf-8-sig")

label_kelas = model.classes_
cm = confusion_matrix(y_test, prediksi, labels=label_kelas)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_kelas, yticklabels=label_kelas,
            linewidths=0.5, ax=ax)
ax.set_title("Confusion Matrix - Analisis Sentimen Ulasan Gojek", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Prediksi", fontsize=12)
ax.set_ylabel("Aktual", fontsize=12)
plt.tight_layout()
plt.savefig("gambar/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

palet = {"Positif": "#2ecc71", "Netral": "#f39c12", "Negatif": "#e74c3c"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Distribusi Sentimen Ulasan Gojek", fontsize=14, fontweight="bold")

warna_bar = [palet.get(k, "#95a5a6") for k in distribusi.index]
batang = ax1.bar(distribusi.index, distribusi.values, color=warna_bar, edgecolor="white", linewidth=1.5)
ax1.set_title("Jumlah Ulasan per Sentimen")
ax1.set_xlabel("Sentimen")
ax1.set_ylabel("Jumlah Ulasan")
for p in batang:
    ax1.annotate(f"{p.get_height():,}",
                 (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha="center", va="bottom", fontweight="bold")

ax2.pie(distribusi.values, labels=distribusi.index, colors=warna_bar,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2))
ax2.set_title("Persentase Sentimen")
plt.tight_layout()
plt.savefig("gambar/distribusi_sentimen.png", dpi=150, bbox_inches="tight")
plt.close()

fig, sumbu = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Word Cloud Ulasan Gojek per Sentimen", fontsize=14, fontweight="bold")

wc_config = {
    "Positif": {"colormap": "Greens", "ax": sumbu[0]},
    "Netral": {"colormap": "Oranges", "ax": sumbu[1]},
    "Negatif": {"colormap": "Reds", "ax": sumbu[2]},
}

for label, cfg in wc_config.items():
    teks = " ".join(df[df["sentimen"] == label]["teks_bersih"].tolist())
    if teks.strip():
        wc = WordCloud(width=500, height=300, background_color="white",
                       colormap=cfg["colormap"], max_words=80,
                       collocations=False).generate(teks)
        cfg["ax"].imshow(wc, interpolation="bilinear")
    cfg["ax"].set_title(f"Sentimen: {label}", fontsize=12, fontweight="bold")
    cfg["ax"].axis("off")

plt.tight_layout()
plt.savefig("gambar/wordcloud_sentimen.png", dpi=150, bbox_inches="tight")
plt.close()

joblib.dump(model, "model/model_sentimen.pkl")
joblib.dump(vektorizer, "model/vektorizer_tfidf.pkl")

metadata = {
    "akurasi": akurasi,
    "jumlah_data": len(df),
    "jumlah_latih": len(X_train),
    "jumlah_uji": len(X_test),
    "jumlah_fitur": X_train_tfidf.shape[1],
    "kelas": list(model.classes_),
    "distribusi": distribusi.to_dict(),
    "presisi_positif": laporan.get("Positif", {}).get("precision", 0),
    "recall_positif": laporan.get("Positif", {}).get("recall", 0),
    "f1_positif": laporan.get("Positif", {}).get("f1-score", 0),
    "presisi_negatif": laporan.get("Negatif", {}).get("precision", 0),
    "recall_negatif": laporan.get("Negatif", {}).get("recall", 0),
    "f1_negatif": laporan.get("Negatif", {}).get("f1-score", 0),
    "presisi_netral": laporan.get("Netral", {}).get("precision", 0),
    "recall_netral": laporan.get("Netral", {}).get("recall", 0),
    "f1_netral": laporan.get("Netral", {}).get("f1-score", 0),
}

with open("model/metadata_model.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"\nModel, vektorizer, dan metadata berhasil disimpan.")
print(f"Akurasi akhir: {akurasi * 100:.2f}%")
